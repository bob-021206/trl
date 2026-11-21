import os
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer


HPML_WORK = Path(__file__).resolve().parents[2]
REPO_ROOT = HPML_WORK.parent


def exact_answer_reward(completions, solution, **kwargs):
    rewards = []
    for completion, expected in zip(completions, solution, strict=True):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        match = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text)
        rewards.append(float(match is not None and match.group(1) == str(expected)))
    return rewards


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("The smoke test requires a CUDA-enabled PyTorch installation.")

    model = os.environ.get("HPML_MODEL", str(REPO_ROOT / "models" / "Qwen3-0.6B"))
    dtype_name = os.environ.get("HPML_DTYPE", "float16")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    use_vllm = os.environ.get("HPML_USE_VLLM", "0") == "1"
    output_dir = os.environ.get("HPML_OUTPUT_DIR", str(HPML_WORK / "windows" / "results" / "role1-smoke"))
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU: {torch.cuda.get_device_name()} | dtype: {dtype_name} | vLLM: {use_vllm}")

    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": f"Compute {a} + {b}. Reply only as <answer>NUMBER</answer>."}],
                "solution": str(a + b),
            }
            for a, b in [(1, 1), (1, 2), (2, 3), (3, 5)]
        ]
    )
    args = GRPOConfig(
        output_dir=output_dir,
        max_steps=1,
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
        model_init_kwargs={"torch_dtype": dtype},
        use_vllm=use_vllm,
        vllm_mode="colocate",
        logging_steps=1,
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=exact_answer_reward,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    torch.cuda.synchronize()
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Completed one GRPO optimizer step. Peak PyTorch GPU memory: {peak_memory_gib:.2f} GiB")


if __name__ == "__main__":
    main()
