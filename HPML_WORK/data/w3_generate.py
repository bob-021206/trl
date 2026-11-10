"""Generate the W3 MuSiQue datasets, one directory per annotated hop count K.

python HPML_WORK/data/w3_generate.py                       # K = 2, 3, 4 from the validation split
python HPML_WORK/data/w3_generate.py --K 2 --num-samples 1 --preview
"""

import argparse
import json
import random
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
from datasets import Dataset, load_dataset
from w3_env import MusiqueSearchEnv, reference_trace


MUSIQUE_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
OUT_DIR = Path(__file__).resolve().parent / "out" / "w3"
SYSTEM_HEAD = "You are a research agent that answers questions with a search tool.\n\nReference notes:\n"
USER_TEMPLATE = (
    "Question: {question}\n"
    "Search for the facts you need, one search per message; do not answer from memory. When you know the answer, "
    "reply with the short answer inside answer tags, for example <answer>Paris</answer>."
)


def generate_cell(lengths, corpus_ids, musique, K, top_k, P, seed, num_samples):
    rng = random.Random(f"w3-{seed}-{K}-{top_k}-{P}")
    system = fit_from_corpus(lengths, lengths.system, SYSTEM_HEAD, corpus_ids, rng, P)
    examples = [x for x in musique if len(x["question_decomposition"]) == K]
    rng.shuffle(examples)
    rows = []
    for x in examples:
        row = {
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": USER_TEMPLATE.format(question=x["question"])},
            ],
            "id": x["id"],
            "docs": [f"{p['title']}: {p['paragraph_text']}" for p in x["paragraphs"]],
            "supporting_idx": [i for i, p in enumerate(x["paragraphs"]) if p["is_supporting"]],
            "decomposition": [d["question"] for d in x["question_decomposition"]],
            "decomposition_answers": [d["answer"] for d in x["question_decomposition"]],
            "answer": x["answer"],
            "answer_aliases": x["answer_aliases"],
            "K": K,
            "top_k": top_k,
            "P": P,
            "seed": seed,
        }
        # Keep only questions whose annotated sub-questions retrieve every supporting paragraph
        _, env = reference_trace(**row)
        if env.found == set(row["supporting_idx"]):
            rows.append(row)
        if len(rows) == num_samples:
            break
    return rows, len(examples)


def cell_meta(lengths, rows, num_candidates, tokenizer_name):
    K, top_k = rows[0]["K"], rows[0]["top_k"]
    traces = [reference_trace(**row)[0] for row in rows]
    L_obs = [lengths.tool_suffix(m["content"]) for trace in traces for m in trace if m["role"] == "tool"]
    # The model chooses its own queries: budget K + 2 searches that each return the k longest paragraphs
    worst = []
    for row in rows:
        longest = sorted(row["docs"], key=len, reverse=True)[:top_k]
        content = "\n\n".join(f"[{rank + 1}] {doc}" for rank, doc in enumerate(longest))
        worst.append((K + 2) * lengths.tool_suffix(content) + (K + 3) * 64)
    max_prompt = max(lengths.prompt(row["prompt"]) for row in rows)
    max_completion_length = round_up(max(worst))
    return {
        "K": K,
        "top_k": top_k,
        "P": rows[0]["P"],
        "seed": rows[0]["seed"],
        "num_samples": len(rows),
        "num_candidates": num_candidates,
        "L_obs": {"min": min(L_obs), "mean": round(sum(L_obs) / len(L_obs), 1), "max": max(L_obs)},
        "max_prompt_len": max_prompt,
        "max_reference_completion_len": max(trace_completion_len(lengths, trace) for trace in traces),
        "grpo_config": {
            "max_completion_length": max_completion_length,
            "vllm_max_model_length": round_up(max_prompt + max_completion_length + 1),
            "max_tool_calling_iterations": K + 2,
        },
        "tokenizer": tokenizer_name,
        "wikitext_revision": WIKITEXT_REVISION,
        "musique_revision": MUSIQUE_REVISION,
        "sha256": rows_sha256(rows),
    }


def load_musique():
    return load_dataset("dgslibisey/MuSiQue", split="validation", revision=MUSIQUE_REVISION).to_list()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--K", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--preview", action="store_true", help="Print the first sample of each cell, do not save.")
    args = parser.parse_args()

    lengths = Lengths(args.tokenizer, [MusiqueSearchEnv.search])
    corpus_ids = load_corpus_ids(lengths.tokenizer)
    musique = load_musique()
    for K in args.K:
        rows, num_candidates = generate_cell(
            lengths, corpus_ids, musique, K, args.top_k, args.P, args.seed, args.num_samples
        )
        meta = cell_meta(lengths, rows, num_candidates, args.tokenizer)
        if args.preview:
            print(json.dumps(rows[0], indent=2, ensure_ascii=False))
            print(json.dumps(meta, indent=2))
            continue
        path = args.out_dir / f"K{K}_top{args.top_k}"
        Dataset.from_list(rows).save_to_disk(path)
        (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{path.name}: {len(rows)}/{num_candidates} kept, L_obs={meta['L_obs']} {meta['grpo_config']}")


if __name__ == "__main__":
    main()
