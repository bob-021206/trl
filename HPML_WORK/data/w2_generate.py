"""Generate the W2 multi-hop retrieval datasets, one directory per (K, top_k) cell. Run w2_kb.py first.

python HPML_WORK/data/w2_generate.py                       # full 4 x 3 grid
python HPML_WORK/data/w2_generate.py --K 3 --top-k 3 --num-samples 2 --preview
"""

import argparse
import itertools
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
from datasets import Dataset
from w2_env import KnowledgeBase, KnowledgeSearchEnv, reference_trace
from w2_kb import KB_DIR


OUT_DIR = Path(__file__).resolve().parent / "out" / "w2"
SYSTEM_HEAD = "You are a research agent that answers questions about a private knowledge base.\n\nReference notes:\n"
USER_TEMPLATE = (
    "Question: {chain}\n"
    "Follow the chain from left to right. Call search with one entity name at a time, starting with {start}, and "
    "read the document about that entity to find the next one. Reply with the final value inside answer tags, for example <answer>1950</answer>."
)


def sample_question(rng, kb, K, top_k):
    """Random walk of K - 1 relations then one attribute; K searches in total. Resampled until every hop is in top-k."""
    while True:
        chain, relations = [rng.choice(kb.entities)], []
        while len(chain) < K:
            entity = chain[-1]
            visited = {e["name"] for e in chain}
            options = [r for r, target in entity["relations"].items() if target not in visited]
            if not options:
                break
            relation = rng.choice(options)
            relations.append(relation)
            chain.append(kb.entities[kb.index[entity["relations"][relation]]])
        if len(chain) < K:
            continue
        if all(kb.index[e["name"]] in kb.search_ids(e["name"], top_k) for e in chain):
            attribute = rng.choice(sorted(chain[-1]["attributes"]))
            return chain, relations, attribute


def generate_cell(lengths, corpus_ids, kb, K, top_k, P, seed, num_samples):
    rng = random.Random(f"w2-{seed}-{K}-{top_k}-{P}")
    system = fit_from_corpus(lengths, lengths.system, SYSTEM_HEAD, corpus_ids, rng, P)
    rows = []
    for _ in range(num_samples):
        chain, relations, attribute = sample_question(rng, kb, K, top_k)
        steps = [chain[0]["name"]] + [r.replace("_", " ") for r in relations] + [attribute.replace("_", " ")]
        user = USER_TEMPLATE.format(chain=" → ".join(steps), start=chain[0]["name"])
        rows.append(
            {
                "prompt": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "start_entity": chain[0]["name"],
                "relations": relations,
                "attribute": attribute,
                "gold_entities": [e["name"] for e in chain],
                "answer": str(chain[-1]["attributes"][attribute]),
                "K": K,
                "top_k": top_k,
                "P": P,
                "seed": seed,
            }
        )
    return rows


def cell_meta(lengths, kb, kb_meta, rows, tokenizer_name):
    K = rows[0]["K"]
    traces = [reference_trace(kb, **row) for row in rows]
    max_prompt = max(lengths.prompt(row["prompt"]) for row in rows)
    max_completion = max(trace_completion_len(lengths, trace) for trace in traces)
    L_obs = [lengths.tool_suffix(m["content"]) for trace in traces for m in trace if m["role"] == "tool"]
    max_completion_length = round_up(max_completion + 32 * (K + 1))
    return {
        "K": K,
        "top_k": rows[0]["top_k"],
        "L_doc": kb_meta["L_doc"],
        "P": rows[0]["P"],
        "seed": rows[0]["seed"],
        "num_samples": len(rows),
        "L_obs": {"min": min(L_obs), "mean": round(sum(L_obs) / len(L_obs), 1), "max": max(L_obs)},
        "max_prompt_len": max_prompt,
        "max_reference_completion_len": max_completion,
        "grpo_config": {
            "max_completion_length": max_completion_length,
            "vllm_max_model_length": round_up(max_prompt + max_completion_length + 1),
            "max_tool_calling_iterations": K + 2,
        },
        "tokenizer": tokenizer_name,
        "wikitext_revision": WIKITEXT_REVISION,
        "kb_sha256": kb_meta["sha256"],
        "sha256": rows_sha256(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--K", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--P", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kb-dir", type=Path, default=KB_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--preview", action="store_true", help="Print the first sample of each cell, do not save.")
    args = parser.parse_args()

    kb = KnowledgeBase(args.kb_dir)
    kb_meta = json.loads((args.kb_dir / "meta.json").read_text())
    lengths = Lengths(kb_meta["tokenizer"], [KnowledgeSearchEnv.search])
    corpus_ids = load_corpus_ids(lengths.tokenizer)
    for K, top_k in itertools.product(args.K, args.top_k):
        rows = generate_cell(lengths, corpus_ids, kb, K, top_k, args.P, args.seed, args.num_samples)
        meta = cell_meta(lengths, kb, kb_meta, rows, kb_meta["tokenizer"])
        if args.preview:
            print(json.dumps(rows[0], indent=2, ensure_ascii=False))
            print(json.dumps(meta, indent=2))
            continue
        path = args.out_dir / f"K{K}_top{top_k}"
        Dataset.from_list(rows).save_to_disk(path)
        (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{path.name}: L_obs={meta['L_obs']} {meta['grpo_config']}")


if __name__ == "__main__":
    main()
