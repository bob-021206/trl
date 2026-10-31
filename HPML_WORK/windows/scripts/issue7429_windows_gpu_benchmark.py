import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HPML_WORK = Path(__file__).resolve().parents[2]
REPO_ROOT = HPML_WORK.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-0.6B GPU weight residency on Windows using Transformers."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HPML_MODEL", "models/Qwen3-0.6B"),
    )
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("HPML_WORK/windows/results/windows_gpu_benchmark_results.json"),
        help="JSON output path (defaults inside HPML_WORK/windows/results).",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.turns < 2:
        parser.error("--turns must be at least 2.")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    return args


def _sleep_model(model):
    model.to("cpu")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _run_policy(model, tokenizer, turns, max_new_tokens, sleep_between_turns):
    device = torch.device("cuda")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with a short phrase."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    sync_ms = []
    sleep_ms = []
    generation_ms = []
    generated_tokens = []
    torch.cuda.reset_peak_memory_stats(device)
    rollout_started = time.perf_counter()

    for turn in range(turns):
        if next(model.parameters()).device.type != "cuda":
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            model.to(device)
            torch.cuda.synchronize(device)
            sync_ms.append((time.perf_counter() - started) * 1000)

        input_ids = torch.tensor([prompt_ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize(device)
        generation_ms.append((time.perf_counter() - started) * 1000)
        completion_ids = output_ids[0, input_ids.shape[1] :].tolist()
        generated_tokens.append(len(completion_ids))
        prompt_ids = (
            prompt_ids
            + completion_ids
            + tokenizer.encode("\nCPU calculator result: 1 + 1 = 2.\n", add_special_tokens=False)
        )

        if sleep_between_turns or turn == turns - 1:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            _sleep_model(model)
            sleep_ms.append((time.perf_counter() - started) * 1000)

    rollout_ms = (time.perf_counter() - rollout_started) * 1000
    return {
        "weight_reloads": len(sync_ms),
        "sleep_calls": len(sleep_ms),
        "inter_turn_sleeps": max(0, len(sleep_ms) - 1) if sleep_between_turns else 0,
        "weight_reload_ms": [round(value, 3) for value in sync_ms],
        "sleep_ms": [round(value, 3) for value in sleep_ms],
        "generation_ms": [round(value, 3) for value in generation_ms],
        "generated_tokens": generated_tokens,
        "rollout_ms": round(rollout_ms, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3),
    }


def main():
    args = parse_args()
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output
    if platform.system() != "Windows":
        raise RuntimeError("This benchmark targets native Windows. Use issue7429_live_benchmark.py on Linux with vLLM.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch install and NVIDIA GPU are required.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; use --dtype float16.")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        repo_model_path = REPO_ROOT / model_path
        if repo_model_path.exists():
            model_path = repo_model_path
    model_source = str(model_path)

    config = AutoConfig.from_pretrained(model_source, revision=args.revision)
    if config.model_type != "qwen3" or config.hidden_size != 1024 or config.num_hidden_layers != 28:
        raise ValueError(
            "Expected the proposal's Qwen3-0.6B architecture "
            "(model_type=qwen3, hidden_size=1024, num_hidden_layers=28)."
        )
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(model_source, revision=args.revision, dtype=dtype)
    model.eval()

    policies = {
        "affected_reload_each_turn": True,
        "fixed_resident_during_rollout": False,
    }
    results = {policy: [] for policy in policies}

    for sleep_between_turns in policies.values():
        if next(model.parameters()).device.type != "cpu":
            _sleep_model(model)
        _run_policy(model, tokenizer, args.turns, args.max_new_tokens, sleep_between_turns)

    for _ in range(args.iterations):
        for policy, sleep_between_turns in policies.items():
            if next(model.parameters()).device.type != "cpu":
                _sleep_model(model)
            results[policy].append(
                _run_policy(model, tokenizer, args.turns, args.max_new_tokens, sleep_between_turns)
            )

    report = {
        "benchmark": "Windows GPU Qwen3 weight-residency emulation (Transformers, not vLLM)",
        "issue": "huggingface/trl#7428",
        "fix_pr": "huggingface/trl#7429",
        "model": model_path.relative_to(REPO_ROOT).as_posix() if model_path.is_relative_to(REPO_ROOT) else args.model,
        "model_revision": args.revision,
        "gpu": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "turns": args.turns,
        "dtype": args.dtype,
        "policies": results,
        "summary": {
            policy: {
                "median_rollout_ms": round(statistics.median(run["rollout_ms"] for run in runs), 3),
                "median_generation_ms_per_turn": [
                    round(statistics.median(run["generation_ms"][turn] for run in runs), 3)
                    for turn in range(args.turns)
                ],
                "median_weight_reload_ms": round(
                    statistics.median(sum(run["weight_reload_ms"]) for run in runs), 3
                ),
                "max_peak_allocated_gib": max(run["peak_allocated_gib"] for run in runs),
            }
            for policy, runs in results.items()
        },
        "scope": (
            "Uses real Qwen3 inference on CUDA and measures actual CPU/GPU weight transfers. "
            "Each policy is warmed up before measurements. This emulates the sleep/reload lifecycle; "
            "it does not execute vLLM, GRPOTrainer, or the upstream fix."
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"GPU: {report['gpu']} | model: Qwen3-0.6B@{args.revision} | dtype: {args.dtype}")
        print("Windows GPU transfer/generation benchmark; vLLM itself is not used.")
        for policy, runs in results.items():
            summary = report["summary"][policy]
            print(
                f"{policy}: {args.turns} turns, "
                f"{runs[0]['weight_reloads']} weight reload(s), {runs[0]['inter_turn_sleeps']} inter-turn sleep(s), "
                f"median rollout {summary['median_rollout_ms']} ms, "
                f"median reload time {summary['median_weight_reload_ms']} ms, "
                f"peak allocated {summary['max_peak_allocated_gib']} GiB"
            )
        print(report["scope"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.json:
        try:
            output_label = args.output.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            output_label = str(args.output)
        print(f"Saved JSON results to {output_label}")


if __name__ == "__main__":
    main()
