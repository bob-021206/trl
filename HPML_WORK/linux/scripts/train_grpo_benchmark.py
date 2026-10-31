import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback

from trl import GRPOConfig, GRPOTrainer


HPML_WORK = Path(__file__).resolve().parents[2]
REPO_ROOT = HPML_WORK.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run a finite Qwen3-0.6B LoRA/GRPO training job.")
    parser.add_argument("--model", default=os.environ.get("HPML_MODEL", "models/Qwen3-0.6B"))
    parser.add_argument("--revision", default=os.environ.get("HPML_MODEL_REVISION"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("--vllm-gpu-memory-utilization must be between 0 and 1.")
    return args


def exact_answer_reward(completions, solution, **kwargs):
    rewards = []
    for completion, expected in zip(completions, solution, strict=True):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        match = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text)
        rewards.append(float(match is not None and match.group(1) == str(expected)))
    return rewards


class StepTimingCallback(TrainerCallback):
    def __init__(self):
        self.step_started_at = None
        self.step_times_ms = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started_at = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started_at is None:
            raise RuntimeError("Training step ended without a recorded start time.")
        self.step_times_ms.append((time.perf_counter() - self.step_started_at) * 1000)
        self.step_started_at = None
        return control


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The training benchmark requires a CUDA-enabled PyTorch installation.")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        repo_model_path = REPO_ROOT / model_path
        if repo_model_path.exists():
            model_path = repo_model_path
    model_source = str(model_path)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    metrics_output = args.metrics_output if args.metrics_output.is_absolute() else REPO_ROOT / args.metrics_output
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    additions = [(1, 1), (1, 2), (2, 3), (3, 5), (5, 8), (13, 21), (34, 55), (89, 144)]
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Compute {left} + {right}. Reply only as <answer>NUMBER</answer>.",
                    }
                ],
                "solution": str(left + right),
            }
            for left, right in additions
        ]
    )
    timing = StepTimingCallback()
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        save_strategy="no",
        report_to="none",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=64,
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=1e-5,
        gradient_checkpointing=True,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        model_init_kwargs={"torch_dtype": dtype, **({"revision": args.revision} if args.revision else {})},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=1024,
        logging_steps=1,
    )
    trainer = GRPOTrainer(
        model=model_source,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=exact_answer_reward,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
        callbacks=[timing],
    )
    if trainer.accelerator.device.type != "cuda":
        raise RuntimeError(f"GRPO training must run on CUDA, got device {trainer.accelerator.device}.")

    torch.cuda.reset_peak_memory_stats()
    training_started_at = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    training_duration_ms = (time.perf_counter() - training_started_at) * 1000
    trainer.save_model(str(output_dir / "final_adapter"))

    report = {
        "task": "Qwen3-0.6B LoRA/GRPO synthetic arithmetic training",
        "model": model_path.relative_to(REPO_ROOT).as_posix() if model_path.is_relative_to(REPO_ROOT) else args.model,
        "model_revision": args.revision,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "vllm_mode": "colocate",
        "vllm_enable_sleep_mode": True,
        "training": {
            "requested_steps": args.max_steps,
            "completed_steps": trainer.state.global_step,
            "duration_ms": round(training_duration_ms, 3),
            "median_step_ms": round(statistics.median(timing.step_times_ms), 3) if timing.step_times_ms else None,
            "step_times_ms": [round(value, 3) for value in timing.step_times_ms],
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "train_loss": train_result.training_loss,
        },
        "output_dir": output_dir.relative_to(REPO_ROOT).as_posix() if output_dir.is_relative_to(REPO_ROOT) else str(output_dir),
    }
    metrics_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
