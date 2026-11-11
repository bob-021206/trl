"""Run GRPO steps with Qwen3-0.6B (vLLM colocate, sleep mode) on each dataset cell, to check TRL runs end to end.

python HPML_WORK/data/smoke_test.py HPML_WORK/data/out/w1/K2_L64_P256
python HPML_WORK/data/smoke_test.py HPML_WORK/data/out/w*/K*        # every cell, one subprocess each

Writes one JSON per cell to out/smoke/ and prints a summary table.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer


OUT_DIR = Path(__file__).resolve().parent / "out" / "smoke"


def workload(cell):
    """Environment factory, reward functions and a per-environment 'followed the whole chain' test for a cell."""
    name = cell.parent.name
    if name == "w1":
        from w1_env import ChainArithmeticEnv, answer_reward, steps_followed_reward

        return (
            ChainArithmeticEnv,
            [answer_reward, steps_followed_reward],
            lambda env: env.steps_followed == len(env.op_ids),
        )
    if name == "w2":
        from w2_env import KnowledgeBase, KnowledgeSearchEnv, answer_reward, hops_followed_reward

        kb = KnowledgeBase(cell.parent / "kb")
        full = lambda env: env.hops_followed == len(env.gold_entities)  # noqa: E731
        return lambda: KnowledgeSearchEnv(kb), [answer_reward, hops_followed_reward], full
    from w3_env import MusiqueSearchEnv, answer_reward, supporting_found_reward

    return MusiqueSearchEnv, [answer_reward, supporting_found_reward], lambda env: env.found == set(env.supporting_idx)


def run_cell(cell, args):
    environment_factory, reward_funcs, followed_all = workload(cell)
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
            output_dir=str(OUT_DIR / "trainer"),
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
            **budget,
        ),
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    aux = reward_funcs[1].__name__
    history = [
        {
            "step": logs["step"],
            "reward": logs["reward"],
            "followed_fraction": logs[f"rewards/{aux}/mean"],
            "tool_calls_per_sample": logs["tools/call_frequency"],
            "step_time": logs["step_time"],
        }
        for logs in trainer.state.log_history
        if "reward" in logs
    ]
    logs = trainer.state.log_history[-2]  # last step (the final entry is the train summary)
    return {
        "cell": f"{cell.parent.name}/{cell.name}",
        "ok": True,
        "step_seconds": round(time.perf_counter() - start, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "num_samples": len(trainer.environments),
        "followed_all_rate": sum(map(followed_all, trainer.environments)) / len(trainer.environments),
        "reward": logs["reward"],
        "followed_fraction": logs[f"rewards/{aux}/mean"],
        "tool_calls_per_sample": logs["tools/call_frequency"],
        "tool_failures_per_sample": logs["tools/failure_frequency"],
        "mean_completion_length": logs["completions/mean_length"],
        "clipped_ratio": logs["completions/clipped_ratio"],
        "max_completion_length": budget["max_completion_length"],
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="+", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lora-r", type=int, default=0, help="LoRA rank; 0 trains all parameters.")
    parser.add_argument(
        "--followed-weight",
        type=float,
        default=0.0,
        help="Reward weight of the chain-following metric; > 0 lets GRPO itself teach the model to run all K turns.",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(args.cells) == 1:
        cell = args.cells[0].resolve()
        result_path = OUT_DIR / f"{cell.parent.name}_{cell.name}.json"
        result_path.write_text(json.dumps({"cell": f"{cell.parent.name}/{cell.name}", "ok": False}) + "\n")
        result_path.write_text(json.dumps(run_cell(cell, args), indent=2) + "\n")
        return

    results = []
    for cell in args.cells:
        cell = cell.resolve()
        if not (cell / "meta.json").exists() or cell.name == "kb":
            continue
        options = [f"--{key.replace('_', '-')}={value}" for key, value in vars(args).items() if key != "cells"]
        log = OUT_DIR / f"{cell.parent.name}_{cell.name}.log"
        with open(log, "w") as f:
            subprocess.run([sys.executable, __file__, str(cell), *options], stdout=f, stderr=subprocess.STDOUT)
        result = json.loads((OUT_DIR / f"{cell.parent.name}_{cell.name}.json").read_text())
        results.append(result)
        print(json.dumps(result), flush=True)
    print(f"\n{sum(r['ok'] for r in results)}/{len(results)} cells ran one GRPO step")


if __name__ == "__main__":
    main()
