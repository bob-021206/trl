"""CPU-only fake-engine fixture for testing plumbing. NEVER use its timings as benchmark results."""

import argparse
from pathlib import Path

from .instrumentation import instrument
from .recorder import Recorder, write_json


class FakeEngine:
    def generate(self, prompts):
        return [p + [7, 8] for p in prompts]

    def sleep(self, level=2):
        pass

    def wake_up(self, tags=None):
        pass

    def reset_prefix_cache(self):
        pass


class FakeGeneration:
    def __init__(self):
        self.llm = FakeEngine()

    def sync_weights(self):
        self.llm.wake_up(tags=["weights"])
        self.llm.reset_prefix_cache()

    def generate(self, prompts):
        self.sync_weights()
        self.llm.wake_up(tags=["kv_cache"])
        self.llm.generate(prompts)
        self.llm.sleep(level=2)
        return prompts, [[7, 8] for _ in prompts], None, None


class FakeTrainer:
    def __init__(self, recorder):
        self.recorder = recorder
        self.vllm_generation = FakeGeneration()

    def _generate(self, prompts):
        for _ in range(3):
            self.vllm_generation.generate(prompts)
        return (prompts,)

    def _get_per_token_logps_and_entropies(self):
        pass

    def _calculate_rewards(self):
        for i in range(2):
            with self.recorder.span("quality_sample", sample_id=str(i), correct=1, fully_followed=True, tool_calls=2):
                pass

    def _generate_and_score_completions(self):
        self._generate([[1], [2]])
        self._get_per_token_logps_and_entropies()
        self._calculate_rewards()

    def training_step(self):
        self._generate_and_score_completions()


def create_fixture(output):
    output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(output / "events.jsonl")
    trainer = FakeTrainer(recorder)
    with instrument(trainer, recorder):
        for step in range(1, 4):
            recorder.step = step
            with recorder.span("optimizer_step"):
                trainer.training_step()
    recorder.flush()
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "kind": "cpu_fixture",
            "status": "complete",
            "config": {
                "variant": "baseline",
                "seed": 0,
                "model": "FAKE",
                "cell": "FAKE",
                "warmup_steps": 1,
                "timing_mode": "throughput",
            },
            "data_meta": {"sha256": "FAKE"},
            "gpu": "NONE",
            "packages": {},
            "torch_profile": False,
            "nvtx": False,
        },
    )
    return recorder, trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_fixture(args.output)
    print(f"CPU FIXTURE ONLY (not performance data): {args.output}")


if __name__ == "__main__":
    main()
