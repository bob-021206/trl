import argparse
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.import_utils import is_vllm_available


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HPML_WORK = Path(__file__).resolve().parents[2]
REPO_ROOT = HPML_WORK.parent


class BenchmarkBoundaryReached(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Measure real colocated-vLLM rollout turn timing and reload counts.")
    parser.add_argument(
        "--model",
        default=os.environ.get("HPML_MODEL", "models/Qwen3-0.6B"),
    )
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--label", help="Run label for comparison plots, e.g. baseline or fixed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HPML_WORK/linux/results/linux_gpu_benchmark_results.json"),
        help="JSON output path (defaults inside HPML_WORK/linux/results).",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.turns < 2:
        parser.error("--turns must be at least 2.")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("--vllm-gpu-memory-utilization must be between 0 and 1.")
    return args


def main():
    args = parse_args()
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output
    if platform.system() != "Linux":
        raise RuntimeError("The live colocated-vLLM benchmark requires Linux.")
    if not torch.cuda.is_available():
        raise RuntimeError("The live benchmark requires a CUDA-enabled NVIDIA GPU.")
    if not is_vllm_available():
        raise RuntimeError(
            'vLLM is not installed. Install this repository with: python -m pip install -e ".[peft,vllm]"'
        )

    model_path = Path(args.model)
    if not model_path.is_absolute():
        repo_model_path = REPO_ROOT / model_path
        if repo_model_path.exists():
            model_path = repo_model_path
    model_source = str(model_path)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model_config = AutoConfig.from_pretrained(model_source, revision=args.revision)
    if model_config.model_type != "qwen3" or model_config.hidden_size != 1024 or model_config.num_hidden_layers != 28:
        raise ValueError(
            "This benchmark requires the proposal's Qwen3-0.6B architecture "
            "(model_type=qwen3, hidden_size=1024, num_hidden_layers=28)."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=args.revision)
    sync_times = []
    sleep_calls_inside_rollout = []
    sleep_calls_after_rollout = []
    current_run = {}
    rollout_active = False

    def rollout_func(prompts, trainer):
        nonlocal rollout_active
        rollout_active = True
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Reply with a short phrase."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            current_ids = prompt_ids
            completion_ids = []
            completion_logprobs = []
            env_mask = []
            turn_times = []
            token_counts = []
            tool_result_ids = tokenizer.encode("\nCPU tool result: ok.\n", add_special_tokens=False)
            for turn in range(args.turns):
                started_at = time.perf_counter()
                _, generated_ids, generated_logprobs, _ = trainer.vllm_generation.generate(
                    [current_ids], images=None, num_generations=1
                )
                turn_times.append((time.perf_counter() - started_at) * 1000)
                generated_ids = generated_ids[0]
                token_counts.append(len(generated_ids))
                completion_ids.extend(generated_ids)
                env_mask.extend([1] * len(generated_ids))

                if generated_logprobs is not None:
                    completion_logprobs.extend(token_logprobs[0] for token_logprobs in generated_logprobs[0])

                current_ids = current_ids + generated_ids
                if turn < args.turns - 1:
                    completion_ids.extend(tool_result_ids)
                    env_mask.extend([0] * len(tool_result_ids))
                    if generated_logprobs is not None:
                        completion_logprobs.extend([0.0] * len(tool_result_ids))
                    current_ids.extend(tool_result_ids)

            current_run["turn_ms"] = turn_times
            current_run["turn_tokens"] = token_counts
            # The GRPO batch has two samples; duplicating this single sampled probe keeps
            # the issue benchmark small while allowing the trainer to finish its rollout phase.
            return {
                "prompt_ids": [prompt_ids, prompt_ids.copy()],
                "completion_ids": [completion_ids, completion_ids.copy()],
                "logprobs": [completion_logprobs, completion_logprobs.copy()]
                if generated_logprobs is not None
                else None,
                "env_mask": [env_mask, env_mask.copy()],
            }
        finally:
            rollout_active = False

    with tempfile.TemporaryDirectory(prefix="trl-issue7429-live-") as output_dir:
        trainer = GRPOTrainer(
            model=model_source,
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            args=GRPOConfig(
                output_dir=output_dir,
                use_cpu=False,
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                vllm_max_model_length=1024,
                model_init_kwargs={"torch_dtype": dtype, "revision": args.revision},
                bf16=dtype == torch.bfloat16,
                fp16=dtype == torch.float16,
                per_device_train_batch_size=2,
                num_generations=2,
                max_completion_length=args.max_new_tokens,
                report_to="none",
            ),
            train_dataset=Dataset.from_dict({"prompt": ["Reply with a short phrase.", "Reply with a short phrase."]}),
            rollout_func=rollout_func,
            peft_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
        if trainer.accelerator.device.type != "cuda":
            raise RuntimeError(f"The vLLM benchmark must run on CUDA, got device {trainer.accelerator.device}.")

        generation = trainer.vllm_generation
        original_sync = generation.sync_weights
        original_sleep = generation.llm.sleep

        def timed_sync():
            started_at = time.perf_counter()
            try:
                return original_sync()
            finally:
                sync_times.append((time.perf_counter() - started_at) * 1000)

        def counted_sleep(*sleep_args, **sleep_kwargs):
            if rollout_active:
                sleep_calls_inside_rollout.append(sleep_kwargs.get("level"))
            else:
                sleep_calls_after_rollout.append(sleep_kwargs.get("level"))
            return original_sleep(*sleep_args, **sleep_kwargs)

        def stop_before_scoring(*args, **kwargs):
            raise BenchmarkBoundaryReached

        generation.sync_weights = timed_sync
        generation.llm.sleep = counted_sleep
        trainer._get_per_token_logps_and_entropies = stop_before_scoring
        results = []
        for _ in range(args.iterations):
            if not generation._llm_weights_sleeping:
                original_sleep(level=2)
            generation._llm_weights_sleeping = True
            sync_times.clear()
            sleep_calls_inside_rollout.clear()
            sleep_calls_after_rollout.clear()
            current_run.clear()
            torch.cuda.reset_peak_memory_stats()
            started_at = time.perf_counter()
            try:
                trainer._generate_and_score_completions([{"prompt": "Reply with a short phrase."}] * 2)
            except BenchmarkBoundaryReached:
                pass
            else:
                raise RuntimeError("The rollout did not stop at the intended pre-scoring boundary.")
            torch.cuda.synchronize()
            results.append(
                {
                    "syncs": len(sync_times),
                    "sleeps": len(sleep_calls_inside_rollout),
                    "sleeps_after_rollout": len(sleep_calls_after_rollout),
                    "sync_ms": [round(value, 3) for value in sync_times],
                    "turn_ms": [round(value, 3) for value in current_run["turn_ms"]],
                    "turn_tokens": current_run["turn_tokens"],
                    "rollout_ms": round((time.perf_counter() - started_at) * 1000, 3),
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                }
            )

    counts = {(result["syncs"], result["sleeps"], result["sleeps_after_rollout"]) for result in results}
    behavior = (
        "buggy"
        if counts == {(args.turns, args.turns, 0)}
        else "fixed"
        if counts == {(1, 0, 1)}
        else "unexpected"
    )
    summary = {
        "median_rollout_ms": round(statistics.median(result["rollout_ms"] for result in results), 3),
        "median_turn_ms": [
            round(statistics.median(result["turn_ms"][turn] for result in results), 3)
            for turn in range(args.turns)
        ],
        "max_peak_allocated_gib": max(result["peak_allocated_gib"] for result in results),
        "max_peak_reserved_gib": max(result["peak_reserved_gib"] for result in results),
    }
    report = {
        "issue": "huggingface/trl#7428",
        "fix_pr": "huggingface/trl#7429",
        "label": args.label,
        "behavior": behavior,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "model": model_path.relative_to(REPO_ROOT).as_posix() if model_path.is_relative_to(REPO_ROOT) else args.model,
        "model_revision": args.revision,
        "gpu": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": __import__("vllm").__version__,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "trl": __import__("trl").__version__,
        "turns_per_rollout": args.turns,
        "dtype": args.dtype,
        "summary": summary,
        "iterations": results,
        "timing_note": "Real GPU/vLLM measurements; compare only matched hardware, model, workload, and software revisions.",
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Run label: {args.label or 'not set'} | observed behavior: {behavior}")
        print(f"TRL source: {report['source_commit']} | working tree dirty: {report['source_dirty']}")
        print(
            f"GPU: {report['gpu']} | model: {args.model}@{args.revision} | "
            f"PyTorch: {report['torch']} | vLLM: {report['vllm']}"
        )
        for index, result in enumerate(results, start=1):
            print(
                f"Run {index}: {result['syncs']} sync(s), {result['sleeps']} sleep(s) inside rollout, "
                f"{result['sleeps_after_rollout']} after rollout, "
                f"sync {result['sync_ms']} ms, turns {result['turn_ms']} ms, "
                f"rollout {result['rollout_ms']} ms, peak allocated {result['peak_allocated_gib']} GiB"
            )
        print(f"Median rollout: {summary['median_rollout_ms']} ms | median turn times: {summary['median_turn_ms']} ms")
        print(report["timing_note"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.json:
        try:
            output_label = args.output.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            output_label = str(args.output)
        print(f"Saved JSON results to {output_label}")

    if behavior == "unexpected":
        raise SystemExit(f"Unexpected sync/sleep counts: {sorted(counts)}")


if __name__ == "__main__":
    main()
