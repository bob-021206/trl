"""Multi-turn GRPO benchmark for issue #7428 (colocated vLLM + sleep mode), built on HPML_WORK/data_prepare/smoke_test.py.

Run from the repository root, e.g.:

    python HPML_WORK/benchmark_7428/bench_multiturn.py HPML_WORK/data_prepare/out/w1/K8_L256_P256 \
        --label fixed --out-dir results/issue7428/train/fixed/w1_K8_L256_P256_1.7B_prefix-on

Runs a few GRPO steps on one W1/W2 cell and records, with wall-clock timestamps so they can be aligned:
- every vLLM engine call (sync_weights, llm.generate, llm.sleep, llm.wake_up, VLLMGeneration.sleep) with its duration,
  plus prompt / prefix-cached / completion token counts per llm.generate;
- the rollout (`_generate`, includes tool execution) and the full generate+score phase per step;
- step boundaries, and the trainer's log_history (reward, tool calls, completion lengths, ...);
- GPU memory used (NVML, whole device) sampled every --mem-interval seconds.

The instrumentation wraps public objects only, so the same script measures any TRL checkout. The TRL that is imported
is the one under test: point PYTHONPATH at a checkout of `main` for the baseline (see run_w1.sh), at this repository for
the fix. summary.json records the path and commit of the TRL actually imported.

Outputs in --out-dir: events.jsonl, gpu_mem.csv, log_history.json, steps.json (per-step aggregates), summary.json.
"""

import argparse
import functools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import TrainerCallback

import trl
import trl.generation.vllm_generation as vllm_generation_module
from trl import GRPOConfig, GRPOTrainer


TRL_ROOT = Path(trl.__file__).resolve().parents[1]  # checkout of the TRL under test (may differ from this repository)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PREPARE = REPO_ROOT / "HPML_WORK" / "data_prepare"
sys.path.insert(0, str(DATA_PREPARE))
T0 = time.perf_counter()


def now():
    return time.perf_counter() - T0


def workload(cell):
    name = cell.parent.name
    if name == "w1":
        from w1_env import ChainArithmeticEnv, answer_reward, steps_followed_reward

        return ChainArithmeticEnv, [answer_reward, steps_followed_reward], "steps_followed_reward"
    if name == "w2":
        from w2_env import KnowledgeBase, KnowledgeSearchEnv, answer_reward, hops_followed_reward

        kb = KnowledgeBase(cell.parent / "kb")
        return lambda: KnowledgeSearchEnv(kb), [answer_reward, hops_followed_reward], "hops_followed_reward"
    raise ValueError(f"Unsupported workload: {name}")


class MemorySampler(threading.Thread):
    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.rows = []
        self.stop_event = threading.Event()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self):
        while not self.stop_event.is_set():
            self.rows.append((round(now(), 3), pynvml.nvmlDeviceGetMemoryInfo(self.handle).used // 2**20))
            time.sleep(self.interval)


class Recorder:
    def __init__(self):
        self.events = []
        self.trainer = None

    def step(self):
        return self.trainer.state.global_step if self.trainer is not None else -1

    def add(self, event, start, **extra):
        self.events.append(
            {
                "event": event,
                "step": self.step(),
                "t_start": round(start, 4),
                "ms": round((now() - start) * 1000, 3),
                **extra,
            }
        )

    def wrap(self, owner, name, event, cuda_sync=False, summarize=None):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            start = now()
            result = original(*args, **kwargs)
            if cuda_sync:
                torch.cuda.synchronize()
            extra = summarize(args, kwargs, result) if summarize else {}
            self.add(event, start, **extra)
            return result

        setattr(owner, name, wrapper)


def summarize_llm_generate(args, kwargs, outputs):
    cached = [output.num_cached_tokens for output in outputs]
    return {
        "requests": len(outputs),
        "prompt_tokens": sum(len(output.prompt_token_ids) for output in outputs),
        "cached_tokens": None if any(c is None for c in cached) else sum(cached),
        "completion_tokens": sum(len(o.token_ids) for output in outputs for o in output.outputs),
    }


class StepCallback(TrainerCallback):
    def __init__(self, recorder):
        self.recorder = recorder
        self.started = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.started = now()

    def on_step_end(self, args, state, control, **kwargs):
        # global_step was already incremented: record the step that just ended
        self.recorder.events.append(
            {
                "event": "train_step",
                "step": state.global_step - 1,
                "t_start": round(self.started, 4),
                "ms": round((now() - self.started) * 1000, 3),
            }
        )


def git(*args):
    return subprocess.run(["git", *args], cwd=TRL_ROOT, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--label", required=True, help="e.g. baseline or fixed")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank; 0 trains all parameters.")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--followed-weight", type=float, default=0.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--mem-interval", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cell = args.cell.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_prefix_caching:
        vllm_generation_module.LLM = functools.partial(vllm_generation_module.LLM, enable_prefix_caching=False)

    recorder = Recorder()
    VLLMGeneration = vllm_generation_module.VLLMGeneration
    recorder.wrap(VLLMGeneration, "sync_weights", "sync_weights", cuda_sync=True)
    recorder.wrap(VLLMGeneration, "generate", "vllm_generation.generate")
    if hasattr(VLLMGeneration, "sleep"):  # only on the fix branch
        recorder.wrap(VLLMGeneration, "sleep", "vllm_generation.sleep")

    environment_factory, reward_funcs, followed_name = workload(cell)
    budget = json.loads((cell / "meta.json").read_text())["grpo_config"]
    num_prompts = args.generation_batch_size // args.num_generations
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=load_from_disk(str(cell)).select(range(num_prompts * args.max_steps)),
        environment_factory=environment_factory,
        reward_funcs=reward_funcs,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear")
        if args.lora_r
        else None,
        args=GRPOConfig(
            output_dir=str(args.out_dir / "trainer_tmp"),
            max_steps=args.max_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=args.generation_batch_size // 2,
            num_generations=args.num_generations,
            learning_rate=args.learning_rate,
            bf16=True,
            gradient_checkpointing=True,
            reward_weights=[1.0, args.followed_weight],
            use_vllm=True,
            vllm_mode="colocate",
            vllm_enable_sleep_mode=True,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            chat_template_kwargs={"enable_thinking": False},
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            **budget,
        ),
        callbacks=[],
    )
    recorder.trainer = trainer
    trainer.add_callback(StepCallback(recorder))
    llm = trainer.vllm_generation.llm
    recorder.wrap(llm, "generate", "llm.generate", summarize=summarize_llm_generate)
    recorder.wrap(llm, "sleep", "llm.sleep", summarize=lambda a, k, r: {"level": k.get("level")})
    recorder.wrap(llm, "wake_up", "llm.wake_up", summarize=lambda a, k, r: {"tags": k.get("tags")})
    recorder.wrap(trainer, "_generate", "rollout")
    recorder.wrap(trainer, "_generate_and_score_completions", "generate_and_score")

    sampler = MemorySampler(args.mem_interval)
    sampler.start()
    train_start = now()
    trainer.train()
    torch.cuda.synchronize()
    train_end = now()
    sampler.stop_event.set()
    sampler.join()

    # Per-step aggregates. Steps are counted by the generation that produced them (global_step at rollout time).
    log_steps = [logs for logs in trainer.state.log_history if "reward" in logs]
    steps = []
    for i in range(trainer.state.global_step):
        ev = [e for e in recorder.events if e["step"] == i]
        of = lambda name, ev=ev: [e for e in ev if e["event"] == name]  # noqa: E731
        gens = of("llm.generate")
        logs = log_steps[i] if i < len(log_steps) else {}
        step_ev = of("train_step")
        steps.append(
            {
                "step": i,
                "step_ms": step_ev[0]["ms"] if step_ev else None,
                "generate_and_score_ms": sum(e["ms"] for e in of("generate_and_score")),
                "rollout_ms": sum(e["ms"] for e in of("rollout")),
                "num_syncs": len(of("sync_weights")),
                "sync_ms": sum(e["ms"] for e in of("sync_weights")),
                "num_llm_sleeps": len(of("llm.sleep")),
                "llm_sleep_ms": sum(e["ms"] for e in of("llm.sleep")),
                "num_llm_wake_ups": len(of("llm.wake_up")),
                "llm_wake_up_ms": sum(e["ms"] for e in of("llm.wake_up")),
                "num_turns": len(gens),
                "llm_generate_ms": sum(e["ms"] for e in gens),
                "turn_ms": [e["ms"] for e in gens],
                "turn_requests": [e["requests"] for e in gens],
                "turn_prompt_tokens": [e["prompt_tokens"] for e in gens],
                "turn_cached_tokens": [e["cached_tokens"] for e in gens],
                "turn_completion_tokens": [e["completion_tokens"] for e in gens],
                "reward": logs.get("reward"),
                "followed_fraction": logs.get(f"rewards/{followed_name}/mean"),
                "tool_calls_per_sample": logs.get("tools/call_frequency"),
                "tool_failures_per_sample": logs.get("tools/failure_frequency"),
                "mean_completion_length": logs.get("completions/mean_length"),
                "clipped_ratio": logs.get("completions/clipped_ratio"),
                "loss": logs.get("loss"),
            }
        )

    import peft
    import transformers
    import vllm

    summary = {
        "label": args.label,
        "cell": f"{cell.parent.name}/{cell.name}",
        "budget": budget,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "trl_path": str(TRL_ROOT),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "has_vllm_generation_sleep": hasattr(VLLMGeneration, "sleep"),
        "gpu": torch.cuda.get_device_name(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "train_start_s": round(train_start, 3),
        "train_end_s": round(train_end, 3),
        "train_wall_s": round(train_end - train_start, 3),
        "completed_steps": trainer.state.global_step,
        "peak_gpu_used_mib": max(m for _, m in sampler.rows),
    }

    out = args.out_dir
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in recorder.events))
    (out / "gpu_mem.csv").write_text("t_s,used_mib\n" + "".join(f"{t},{m}\n" for t, m in sampler.rows))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (out / "steps.json").write_text(json.dumps(steps, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for s in steps:
        print(
            f"step {s['step']}: {s['step_ms']:.0f} ms | rollout {s['rollout_ms']:.0f} ms | turns {s['num_turns']} | "
            f"syncs {s['num_syncs']} ({s['sync_ms']:.0f} ms) | sleeps {s['num_llm_sleeps']} | "
            f"tool calls/sample {s['tool_calls_per_sample']} | followed {s['followed_fraction']}"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    main()
