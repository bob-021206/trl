# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import load_dataset

from trl import DistillationConfig, DistillationTrainer, GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

from .testing_utils import TrlTestCase


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
# A rollout starts by waking the weights to push them, then the KV cache to generate
WAKE_AND_SYNC = ["wake_up:weights", "load_weights", "wake_up:kv_cache"]


def varied_reward(completions, **kwargs):
    # Rewards that differ within a group give a non-zero loss, so that the weights actually change
    return [float(i) for i in range(len(completions))]


def make_rollout_func(num_turns, between_turns=None):
    def rollout_func(prompts, trainer):
        for turn in range(num_turns):
            trainer.vllm_generation.generate([[2]], images=None, num_generations=1)
            if between_turns is not None and turn < num_turns - 1:
                between_turns(trainer)
        batch_size = len(prompts)
        return {
            "prompt_ids": [[2]] * batch_size,
            # Distinct completions, or the advantages of a group would cancel out and leave the weights unchanged
            "completion_ids": [[3 + i, 4] for i in range(batch_size)],
            "logprobs": [[-0.1, -0.1]] * batch_size,
        }

    return rollout_func


def record_engine(trainer, llm):
    """
    Records what the rollout asks of the engine, in order, along with the forward passes of the trained model.

    Returns:
        `tuple` of:
            - `calls` (`list[str]`):
                The ordered calls. Consecutive `load_weights` are merged, as vLLM gets one parameter per call.
            - `pushed_weights` (`list[dict[str, torch.Tensor]]`):
                The weights received by the engine, one dict per synchronization.
    """
    calls = []
    pushed_weights = []

    def load_weights(named_params):
        if not calls or calls[-1] != "load_weights":
            calls.append("load_weights")
            pushed_weights.append({})
        # `_iter_named_params` passes the training parameters themselves: clone them, or the optimizer step that
        # follows would also change what was recorded
        pushed_weights[-1].update((name, param.clone()) for name, param in named_params)

    def generate(prompts, sampling_params, use_tqdm):
        calls.append("generate")
        return [
            MagicMock(
                prompt_token_ids=prompt["prompt_token_ids"], outputs=[MagicMock(token_ids=[3, 4], logprobs=None)]
            )
            for prompt in prompts
        ]

    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights.side_effect = load_weights
    llm.generate.side_effect = generate
    llm.wake_up.side_effect = lambda tags: calls.append(f"wake_up:{tags[0]}")
    llm.sleep.side_effect = lambda level: calls.append(f"sleep:{level}")
    trainer.model.register_forward_pre_hook(lambda *args: calls.append("forward"))
    return calls, pushed_weights


def without_forward(calls):
    return [call for call in calls if call != "forward"]


@pytest.fixture
def llm():
    # Stand in for the vLLM engine, which cannot run here
    with (
        patch("trl.generation.vllm_generation.is_vllm_available", return_value=True),
        patch("trl.generation.vllm_generation.LLM", create=True) as llm_class,
        patch("trl.generation.vllm_generation.SamplingParams", create=True),
    ):
        yield llm_class.return_value


@patch.dict(os.environ, {"TRL_EXPERIMENTAL_SILENCE": "1"})
class TestVLLMSleep(TrlTestCase):
    def _make_grpo_trainer(self, rollout_func, **config_kwargs):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        return GRPOTrainer(
            model=MODEL_ID,
            reward_funcs=varied_reward,
            args=GRPOConfig(
                output_dir=self.tmp_dir,
                learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                num_generations=2,
                num_generations_eval=2,
                save_strategy="no",
                report_to="none",
                **config_kwargs,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            rollout_func=rollout_func,
        )

    def test_rollout_syncs_once_and_sleeps_before_scoring(self, llm):
        trainer = self._make_grpo_trainer(make_rollout_func(num_turns=2), max_steps=1)
        calls, _ = record_engine(trainer, llm)

        trainer.train()

        assert calls[: calls.index("forward")] == [*WAKE_AND_SYNC, "generate", "generate", "sleep:2"]
        assert without_forward(calls).count("sleep:2") == 1

    def test_next_rollout_pushes_the_updated_weights(self, llm):
        trainer = self._make_grpo_trainer(make_rollout_func(num_turns=2), max_steps=2)
        calls, pushed_weights = record_engine(trainer, llm)

        trainer.train()

        one_rollout = [*WAKE_AND_SYNC, "generate", "generate", "sleep:2"]
        assert without_forward(calls) == one_rollout * 2
        first, second = pushed_weights
        assert first.keys() == second.keys()
        assert any(not torch.equal(first[name], second[name]) for name in first)

    def test_rollout_at_the_same_step_reloads_the_discarded_weights(self, llm):
        # The trainer only syncs when the step changes, so `generate()` must notice that the weights were discarded
        trainer = self._make_grpo_trainer(make_rollout_func(num_turns=2))
        calls, pushed_weights = record_engine(trainer, llm)
        trainer.evaluate()
        num_rollouts = without_forward(calls).count("sleep:2")

        calls.clear()
        trainer.evaluate()

        assert num_rollouts > 0
        assert without_forward(calls).count("load_weights") == num_rollouts
        assert len(pushed_weights) == 2 * num_rollouts

    def test_sleeping_between_turns_reloads_the_weights(self, llm):
        # A tool that needs GPU memory can release the engine between two turns
        rollout_func = make_rollout_func(num_turns=2, between_turns=lambda trainer: trainer.vllm_generation.sleep())
        trainer = self._make_grpo_trainer(rollout_func, max_steps=1)
        calls, _ = record_engine(trainer, llm)

        trainer.train()

        one_turn = [*WAKE_AND_SYNC, "generate", "sleep:2"]
        assert without_forward(calls) == one_turn * 2

    def test_rollout_without_generation_still_sleeps(self, llm):
        # The trainer wakes the weights to sync them, but the KV cache is still asleep: the weights must not stay on
        # the GPU during training
        trainer = self._make_grpo_trainer(make_rollout_func(num_turns=0), max_steps=1)
        calls, _ = record_engine(trainer, llm)

        trainer.train()

        assert without_forward(calls) == ["wake_up:weights", "load_weights", "sleep:2"]

    def test_generation_after_a_partial_wake_and_sleep_wakes_everything(self, llm):
        trainer = self._make_grpo_trainer(make_rollout_func(num_turns=0))
        calls, _ = record_engine(trainer, llm)
        trainer.vllm_generation.sync_weights()
        trainer.vllm_generation.sleep()

        calls.clear()
        trainer.vllm_generation.generate([[2]], images=None, num_generations=1)

        assert calls == [*WAKE_AND_SYNC, "generate"]

    def test_sleep_does_not_sleep_an_asleep_engine(self, llm):
        trainer = self._make_grpo_trainer(make_rollout_func(num_turns=1))
        llm.reset_mock()  # ignore the sleep that ends the engine construction

        trainer.vllm_generation.sleep()

        llm.sleep.assert_not_called()

    def test_sleep_without_sleep_mode_is_a_noop(self, llm):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        trainer = GRPOTrainer(
            model=MODEL_ID,
            reward_funcs=varied_reward,
            args=GRPOConfig(output_dir=self.tmp_dir, use_vllm=True, vllm_mode="colocate", report_to="none"),
            train_dataset=dataset,
        )

        trainer.vllm_generation.sleep()

        llm.sleep.assert_not_called()

    def test_sleep_is_a_noop_in_server_mode(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        # Stand in for the vLLM server, which cannot run here
        with (
            patch("trl.generation.vllm_generation.is_vllm_available", return_value=True),
            patch("trl.generation.vllm_generation.VLLMClient"),
        ):
            trainer = GRPOTrainer(
                model=MODEL_ID,
                reward_funcs=varied_reward,
                args=GRPOConfig(
                    output_dir=self.tmp_dir,
                    use_vllm=True,
                    vllm_mode="server",
                    vllm_enable_sleep_mode=True,
                    report_to="none",
                ),
                train_dataset=dataset,
            )

        trainer.vllm_generation.sleep()  # must not reach for an engine that only exists in colocate mode

    def test_rloo_sleeps_before_scoring(self, llm):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        trainer = RLOOTrainer(
            model=MODEL_ID,
            reward_funcs=varied_reward,
            args=RLOOConfig(
                output_dir=self.tmp_dir,
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                per_device_train_batch_size=2,
                num_generations=2,
                max_steps=1,
                save_strategy="no",
                report_to="none",
            ),
            train_dataset=dataset,
        )
        calls, _ = record_engine(trainer, llm)

        trainer.train()

        assert calls[: calls.index("forward")] == [*WAKE_AND_SYNC, "generate", "sleep:2"]

    def test_distillation_sleeps_before_scoring(self, llm):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        trainer = DistillationTrainer(
            model=MODEL_ID,
            teacher_model=MODEL_ID,
            args=DistillationConfig(
                output_dir=self.tmp_dir,
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                per_device_train_batch_size=2,
                max_steps=1,
                save_strategy="no",
                report_to="none",
            ),
            train_dataset=dataset,
        )
        calls, _ = record_engine(trainer, llm)

        trainer.train()

        assert calls[: calls.index("forward")] == [*WAKE_AND_SYNC, "generate", "sleep:2"]
