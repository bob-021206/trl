"""Generate the W1 chain-arithmetic datasets, one directory per (K, L_obs, P) cell.

python HPML_WORK/data/w1_generate.py                       # full 4 x 3 x 2 grid
python HPML_WORK/data/w1_generate.py --K 3 --L-obs 64 --P 256 --num-samples 2 --preview
"""

import argparse
import itertools
import json
import random
import string
from pathlib import Path

from common import (
    WIKITEXT_REVISION,
    Lengths,
    fit_from_corpus,
    load_corpus_ids,
    round_up,
    rows_sha256,
    trace_completion_len,
)
from datasets import Dataset
from w1_env import ChainArithmeticEnv, reference_trace


OUT_DIR = Path(__file__).resolve().parent / "out" / "w1"
SYSTEM_HEAD = "You are a calculator agent that follows an operation chain.\n\nReference notes:\n"
USER_TEMPLATE = (
    "The starting value is {x0}. The chain has {K} steps. The op_id of step 1 is {op_id}.\n"
    "Call get_op with one op_id at a time and use the next_id it returns for the following call. After each "
    "result, first write the new running value (for example: 45 - 2 = 43), then make the next call. Text after the "
    "--- line in a tool result is unrelated; ignore it. After the last step, do not call get_op again: reply with "
    "the final value inside answer tags, for example <answer>43</answer>."
)
ID_CHARS = string.ascii_lowercase + string.digits


def sample_chain(rng, K):
    while True:
        x0 = rng.randint(10, 99)
        ops = [rng.choice((-1, 1)) * rng.randint(1, 9) for _ in range(K)]
        if all(0 <= value <= 199 for value in itertools.accumulate([x0] + ops)):
            return x0, ops


def sample_op_ids(rng, K):
    op_ids = []
    while len(op_ids) < K:
        op_id = "".join(rng.choice(ID_CHARS) for _ in range(4))
        # One letter and one digit at least, so an id never shows up as a word in the filler text
        if op_id not in op_ids and any(c.isdigit() for c in op_id) and any(c.isalpha() for c in op_id):
            op_ids.append(op_id)
    return op_ids


def generate_cell(lengths, corpus_ids, K, L_obs, P, seed, num_samples):
    rng = random.Random(f"w1-{seed}-{K}-{L_obs}-{P}")
    # One system prompt per cell, shared by every sample like a real agent's system prompt
    system = fit_from_corpus(lengths, lengths.system, SYSTEM_HEAD, corpus_ids, rng, P)
    rows = []
    for _ in range(num_samples):
        x0, ops = sample_chain(rng, K)
        op_ids = sample_op_ids(rng, K)
        observations = []
        for i in range(K):
            if i + 1 < K:
                head = f"[step {i + 1}/{K}] op: {ops[i]:+d}\nnext_id: {op_ids[i + 1]}\n---\n"
            else:
                head = f"[step {K}/{K}] op: {ops[i]:+d}\nThis was the last step: reply with the final value.\n---\n"
            observations.append(fit_from_corpus(lengths, lengths.tool_suffix, head, corpus_ids, rng, L_obs))
        user = USER_TEMPLATE.format(x0=x0, K=K, op_id=op_ids[0])
        rows.append(
            {
                "prompt": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "x0": x0,
                "ops": ops,
                "op_ids": op_ids,
                "observations": observations,
                "answer": x0 + sum(ops),
                "K": K,
                "L_obs": L_obs,
                "P": P,
                "seed": seed,
            }
        )
    return rows


def cell_meta(lengths, rows, tokenizer_name):
    K = rows[0]["K"]
    max_prompt = max(lengths.prompt(row["prompt"]) for row in rows)
    max_completion = max(trace_completion_len(lengths, reference_trace(**row)) for row in rows)
    # Margin of 32 tokens per generated turn, for outputs more verbose than the reference trace
    max_completion_length = round_up(max_completion + 32 * (K + 1))
    return {
        "K": K,
        "L_obs": rows[0]["L_obs"],
        "P": rows[0]["P"],
        "seed": rows[0]["seed"],
        "num_samples": len(rows),
        "max_prompt_len": max_prompt,
        "max_reference_completion_len": max_completion,
        "grpo_config": {
            "max_completion_length": max_completion_length,
            "vllm_max_model_length": round_up(max_prompt + max_completion_length + 1),
            "max_tool_calling_iterations": K + 2,
        },
        "tokenizer": tokenizer_name,
        "wikitext_revision": WIKITEXT_REVISION,
        "sha256": rows_sha256(rows),
    }


def cell_dir(out_dir, K, L_obs, P):
    return Path(out_dir) / f"K{K}_L{L_obs}_P{P}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--K", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--L-obs", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--P", type=int, nargs="+", default=[256, 2048])
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--preview", action="store_true", help="Print the first sample of each cell, do not save.")
    args = parser.parse_args()

    lengths = Lengths(args.tokenizer, [ChainArithmeticEnv.get_op])
    corpus_ids = load_corpus_ids(lengths.tokenizer)
    for K, L_obs, P in itertools.product(args.K, args.L_obs, args.P):
        rows = generate_cell(lengths, corpus_ids, K, L_obs, P, args.seed, args.num_samples)
        meta = cell_meta(lengths, rows, args.tokenizer)
        if args.preview:
            print(json.dumps(rows[0], indent=2, ensure_ascii=False))
            print(json.dumps(meta, indent=2))
            continue
        path = cell_dir(args.out_dir, K, L_obs, P)
        Dataset.from_list(rows).save_to_disk(path)
        (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{path.name}: {meta['grpo_config']}")


if __name__ == "__main__":
    main()
