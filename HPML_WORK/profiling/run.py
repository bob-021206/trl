"""Profile and evaluate Yidan's real W1/W2/W3 workloads with the existing GRPO tool loop."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from .capture import make_ranges, measurement_callback, memory_history, profiler_context
from .instrumentation import instrument, validate_lifecycle
from .memory import MemorySampler, write_csv
from .recorder import Recorder, write_json
from .workloads import load_workload


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"baseline": False, "baseline_cache": True, "awake": False, "awake_cache": True}
DEFAULTS = dict(
    mode="benchmark",
    timing_mode="throughput",
    eval_samples=16,
    eval_batch_size=2,
    split_seed=1729,
    train_samples=0,
    demo_samples=2,
    profile_wait=1,
    profile_warmup=1,
    profile_active=2,
    with_stack=False,
    memory_snapshot=False,
    cuda_events=False,
    memory_sampling_ms=100,
    nvml_uuid=None,
    followed_weight=0.0,
    temperature=1.0,
    save_adapter=False,
)


def command(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def package_versions():
    names = (
        "torch",
        "vllm",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "tokenizers",
        "numpy",
        "bm25s",
        "nvidia-ml-py",
        "matplotlib",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_config(config):
    for key, value in DEFAULTS.items():
        config.setdefault(key, value)
    if config["variant"] not in VARIANTS or config["mode"] not in ("benchmark", "profile", "nsight"):
        raise ValueError("Unknown variant or capture mode")
    if config["timing_mode"] not in ("throughput", "diagnostic"):
        raise ValueError("timing_mode must be throughput or diagnostic")
    if not 0 <= config["warmup_steps"] < config["max_steps"]:
        raise ValueError("Require 0 <= warmup_steps < max_steps")
    for key in (
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "eval_batch_size",
        "profile_warmup",
        "profile_active",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "warmup_steps",
        "lora_r",
        "eval_samples",
        "train_samples",
        "demo_samples",
        "profile_wait",
        "memory_sampling_ms",
    ):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    generations = config["num_generations"]
    batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    if not isinstance(generations, int) or generations < 2 or batch % generations:
        raise ValueError("Generation batch must be divisible by num_generations >= 2")
    if not config.get("revision"):
        raise ValueError("Pin the model revision, including for a local snapshot")
    if not 0 < config["vllm_gpu_memory_utilization"] < 1:
        raise ValueError("Invalid vLLM memory fraction")
    if config["temperature"] <= 0:
        raise ValueError("Training temperature must be positive")
    if Path(config["cell"]).parent.name not in ("w1", "w2", "w3"):
        raise ValueError("Expected a generated W1/W2/W3 cell directory")
    if config["mode"] == "profile":
        required = sum(config[k] for k in ("warmup_steps", "profile_wait", "profile_warmup", "profile_active"))
        if config["max_steps"] < required:
            raise ValueError(f"Profiler schedule needs at least {required} optimizer steps")
    if config["memory_snapshot"] and config["mode"] == "benchmark":
        raise ValueError("Memory history adds overhead; use --mode profile or nsight for snapshots")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "HPML_WORK/profiling/configs/smoke.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate config without importing GPU dependencies")
    parser.add_argument("--plots", action="store_true", help="Generate per-run figures and HTML demo report")
    for name in ("cell", "model", "revision", "nvml-uuid"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--mode", choices=("benchmark", "profile", "nsight"))
    parser.add_argument("--timing-mode", choices=("throughput", "diagnostic"))
    for name in (
        "seed",
        "max-steps",
        "warmup-steps",
        "eval-samples",
        "eval-batch-size",
        "split-seed",
        "train-samples",
        "demo-samples",
        "profile-wait",
        "profile-warmup",
        "profile-active",
        "memory-sampling-ms",
        "lora-r",
        "num-generations",
        "per-device-train-batch-size",
        "gradient-accumulation-steps",
    ):
        parser.add_argument(f"--{name}", type=int)
    for name in ("learning-rate", "vllm-gpu-memory-utilization", "followed-weight", "temperature"):
        parser.add_argument(f"--{name}", type=float)
    for name in ("with-stack", "memory-snapshot", "cuda-events", "save-adapter"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    for key, value in vars(args).items():
        if key not in ("config", "output", "dry_run", "plots") and value is not None:
            config[key] = value
    validate_config(config)
    return args, config


def run_experiment(config, output):
    if platform.system() != "Linux":
        raise RuntimeError("Real runs require Linux and a CUDA GPU; use --dry-run or the CPU tests on this machine")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This harness currently supports one GPU and one training process")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=2,
        kind="gpu_training",
        status="running",
        config=config,
        started_unix_ns=time.time_ns(),
        platform=platform.platform(),
        python=sys.version,
        torch_profile=config["mode"] == "profile",
        nvtx=config["mode"] == "nsight",
        cache_metrics={
            "source": None,
            "prefill_tokens": None,
            "hit_tokens": None,
            "query_tokens": None,
            "evictions": None,
            "preemptions": None,
        },
    )
    recorder, sampler = None, None
    allocator_samples, demos = [], []
    write_json(output / "manifest.json", manifest)
    try:
        import torch
        from peft import LoraConfig
        from transformers import TrainerCallback, set_seed

        import trl.generation.vllm_generation as backend
        from trl import GRPOConfig, GRPOTrainer

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one allocated CUDA GPU with CUDA_VISIBLE_DEVICES")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("These configs use BF16; select a BF16-capable GPU")
        if ROOT not in Path(backend.__file__).resolve().parents:
            raise RuntimeError("Imported TRL is not this checkout; run pip install -e . --no-deps")
        validate_lifecycle(backend.VLLMGeneration, config["variant"])
        recorder = Recorder(
            output / "events.jsonl",
            ranges=make_ranges(torch, config["mode"]),
            synchronize=torch.cuda.synchronize if config["timing_mode"] == "diagnostic" else None,
        )
        recorder.phase = "setup"
        manifest.update(
            git_commit=command("git", "rev-parse", "HEAD"),
            git_status=command("git", "status", "--porcelain"),
            packages=package_versions(),
            gpu=torch.cuda.get_device_name(0),
            cuda_runtime=torch.version.cuda,
            gpu_inventory=command("nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            clock_origin_unix_ns=recorder.origin_unix_ns,
        )
        source = output / "harness-source"
        source.mkdir()
        manifest["harness_hashes"] = {}
        for path in sorted(Path(__file__).parent.glob("*.py")):
            shutil.copyfile(path, source / path.name)
            manifest["harness_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "source.patch").write_text(command("git", "diff", "HEAD", "--", "trl", "HPML_WORK/data_prepare"))
        (output / "pip-freeze.txt").write_text(command(sys.executable, "-m", "pip", "freeze") + "\n")

        cell = (ROOT / config["cell"]).resolve()
        train, evaluation, factory, rewards, meta, split, demos = load_workload(cell, config, recorder)
        manifest.update(
            data_meta=meta, split_sha256=hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
        )
        write_json(output / "split.json", split)
        set_seed(config["seed"])
        profiler_ref = [None]
        callback = measurement_callback(torch, TrainerCallback, recorder, config, profiler_ref, allocator_samples)
        original_llm = backend.LLM

        def configured_llm(*llm_args, **llm_kwargs):
            # Experimental controls only; the checked-out TRL source owns the sleep lifecycle.
            llm_kwargs.update(
                enable_prefix_caching=VARIANTS[config["variant"]],
                seed=config["seed"],
                revision=config["revision"],
                tokenizer_revision=config["revision"],
            )
            return original_llm(*llm_args, **llm_kwargs)

        with patch.object(backend, "LLM", configured_llm):
            trainer = GRPOTrainer(
                model=config["model"],
                train_dataset=train,
                eval_dataset=evaluation if len(evaluation) else None,
                environment_factory=factory,
                reward_funcs=rewards,
                peft_config=LoraConfig(
                    r=config["lora_r"],
                    lora_alpha=2 * config["lora_r"],
                    target_modules="all-linear",
                    task_type="CAUSAL_LM",
                )
                if config["lora_r"]
                else None,
                args=GRPOConfig(
                    output_dir=str(output / "trainer"),
                    max_steps=config["max_steps"],
                    per_device_train_batch_size=config["per_device_train_batch_size"],
                    gradient_accumulation_steps=config["gradient_accumulation_steps"],
                    steps_per_generation=config["gradient_accumulation_steps"],
                    num_iterations=1,
                    num_generations=config["num_generations"],
                    num_generations_eval=1,
                    per_device_eval_batch_size=config["eval_batch_size"],
                    learning_rate=config["learning_rate"],
                    seed=config["seed"],
                    data_seed=config["seed"],
                    temperature=config["temperature"],
                    bf16=True,
                    gradient_checkpointing=True,
                    reward_weights=[1.0, config["followed_weight"]],
                    model_init_kwargs={"revision": config["revision"], "dtype": torch.bfloat16},
                    use_vllm=True,
                    vllm_mode="colocate",
                    vllm_enable_sleep_mode=True,
                    vllm_gpu_memory_utilization=config["vllm_gpu_memory_utilization"],
                    chat_template_kwargs={"enable_thinking": False},
                    remove_unused_columns=False,
                    logging_steps=1,
                    save_strategy="no",
                    report_to="none",
                    **meta["grpo_config"],
                ),
                callbacks=[callback],
            )
        actual_cache = trainer.vllm_generation.llm.llm_engine.vllm_config.cache_config.enable_prefix_caching
        if actual_cache != VARIANTS[config["variant"]]:
            raise RuntimeError("Resolved prefix cache setting differs from the requested variant")
        manifest.update(
            resolved_prefix_caching=actual_cache,
            resolved_grpo_config=json.loads(json.dumps(trainer.args.to_dict(), default=str)),
            tokenizer={
                "name_or_path": trainer._tokenizer.name_or_path,
                "commit": trainer._tokenizer.init_kwargs.get("_commit_hash"),
                "chat_template_sha256": hashlib.sha256(str(trainer._tokenizer.chat_template).encode()).hexdigest(),
            },
        )
        write_json(output / "manifest.json", manifest)
        sampler = MemorySampler(torch, recorder, config["memory_sampling_ms"], config["nvml_uuid"])
        recorder.phase = "train"
        sampler.start()
        with instrument(trainer, recorder):
            with memory_history(torch, config, output), profiler_context(torch, config, output) as prof:
                profiler_ref[0] = prof
                with recorder.span("train_loop"):
                    trainer.train()
                    torch.cuda.synchronize()
            profiler_ref[0] = None
            if len(evaluation):
                recorder.phase = "eval"
                recorder.step = None
                torch.cuda.synchronize()
                with recorder.span("evaluation"):
                    metrics = trainer.evaluate()
                    torch.cuda.synchronize()
                write_json(output / "eval-trainer-metrics.json", metrics)
            recorder.phase = "save"
            if config["save_adapter"]:
                trainer.save_model(str(output / "final_model"))
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if recorder:
            recorder.close_open(exc)
        raise
    finally:
        if sampler:
            sampler.stop()
            manifest["nvml"] = sampler.status
            write_csv(output / "memory.csv", sampler.rows)
        write_csv(output / "allocator_memory.csv", allocator_samples)
        write_json(output / "demo-trajectories.json", demos)
        if recorder:
            recorder.flush()
        manifest["finished_unix_ns"] = time.time_ns()
        write_json(output / "manifest.json", manifest)
        print(f"Artifacts ({manifest['status']}): {output}")
    from .summarize import summarize_run

    summary = summarize_run(output)
    print(json.dumps(summary, indent=2))
    return output


def main():
    args, config = parse_args()
    if args.dry_run:
        print(json.dumps({"config": config, "prefix_caching": VARIANTS[config["variant"]]}, indent=2))
        return
    output = (args.output or ROOT / "HPML_WORK/profiling/runs" / f"{config['variant']}-{time.time_ns()}").resolve()
    run_experiment(config, output)
    if args.plots:
        from .plot import plot_run

        plot_run(output)


if __name__ == "__main__":
    main()
