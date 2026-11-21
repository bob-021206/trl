"""Static checks and scripted-policy replay for the W2 knowledge base and datasets (CPU only).

python HPML_WORK/data/w2_check.py                 # the knowledge base and every cell under out/w2
python HPML_WORK/data/w2_check.py out/w2/K3_top3
"""

import argparse
import json
import re
from pathlib import Path

from common import Lengths, load_corpus_ids, rows_sha256
from datasets import load_from_disk
from w2_env import KnowledgeBase, KnowledgeSearchEnv, answer_reward
from w2_generate import OUT_DIR, cell_meta, generate_cell
from w2_kb import ATTRIBUTES, KB_DIR, RELATIONS


def read_fact(result, name, template):
    """Find `template` filled with `name` in a search result; the placeholder for the value is captured."""
    pattern = re.escape(template.format(s=name, t="\0", v="\0")).replace("\0", "([^.]+?)")
    match = re.search(pattern, result)
    return match.group(1) if match else None


def replay(kb, row):
    """Scripted policy: read each next entity and the final value from the search results only."""
    env = KnowledgeSearchEnv(kb)
    env.reset(**row)
    entity, calls = row["start_entity"], 0
    for relation in row["relations"]:
        result = env.search(entity)
        calls += 1
        entity = read_fact(result, entity, RELATIONS[relation][2])
    value = read_fact(env.search(entity), entity, ATTRIBUTES[row["attribute"]][3])
    completion = [{"role": "assistant", "content": f"<answer>{value}</answer>"}]
    return calls + 1, env, answer_reward([completion], [row["answer"]])[0]


def check_kb(kb, kb_meta, lengths):
    errors = []
    if len(kb.index) != len(kb.entities):
        errors.append("entity names are not unique")
    if rows_sha256(kb.entities) != kb_meta["sha256"]:
        errors.append("entities do not match meta sha256")
    bad = [e["name"] for e in kb.entities if lengths.text(e["doc"]) != kb_meta["L_doc"]]
    if bad:
        errors.append(f"{len(bad)} documents are not exactly {kb_meta['L_doc']} tokens, e.g. {bad[0]}")
    return errors


def check_cell(path, kb, kb_meta, lengths, corpus_ids):
    rows = load_from_disk(str(path)).to_list()
    meta = json.loads((path / "meta.json").read_text())
    K, top_k, P = meta["K"], meta["top_k"], meta["P"]
    errors = []
    for n, row in enumerate(rows):
        entity = kb.entities[kb.index[row["start_entity"]]]
        chain = [entity["name"]]
        for relation in row["relations"]:
            entity = kb.entities[kb.index[entity["relations"][relation]]]
            chain.append(entity["name"])
        if chain != row["gold_entities"] or len(chain) != K or len(set(chain)) != K:
            errors.append(f"row {n}: gold entities do not follow the relations")
        if str(entity["attributes"][row["attribute"]]) != row["answer"]:
            errors.append(f"row {n}: answer wrong")
        user = row["prompt"][1]["content"]
        if any(name in user for name in chain[1:]):
            errors.append(f"row {n}: a later hop is named in the question")
        if any(kb.index[name] not in kb.search_ids(name, top_k) for name in chain):
            errors.append(f"row {n}: a gold document is not in the top-{top_k}")
        if lengths.system(row["prompt"][0]["content"]) != P:
            errors.append(f"row {n}: system prefix length != {P}")
        calls, env, reward = replay(kb, row)
        if calls != K or env.hops_followed != K or reward != 1.0:
            errors.append(f"row {n}: replay calls={calls} hops_followed={env.hops_followed} reward={reward}")
    if len({row["prompt"][0]["content"] for row in rows}) != 1:
        errors.append("system prompt differs across samples")
    if meta["kb_sha256"] != kb_meta["sha256"]:
        errors.append("dataset was generated from a different knowledge base")
    if rows_sha256(rows) != meta["sha256"]:
        errors.append("saved rows do not match meta sha256")
    regenerated = generate_cell(lengths, corpus_ids, kb, K, top_k, P, meta["seed"], meta["num_samples"])
    if cell_meta(lengths, kb, kb_meta, regenerated, meta["tokenizer"]) != meta:
        errors.append("regenerating with the same seed gives a different dataset or budget")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="*", type=Path)
    parser.add_argument("--kb-dir", type=Path, default=KB_DIR)
    args = parser.parse_args()
    cells = args.cells or sorted(path for path in OUT_DIR.iterdir() if path.is_dir() and path != KB_DIR)

    kb = KnowledgeBase(args.kb_dir)
    kb_meta = json.loads((args.kb_dir / "meta.json").read_text())
    lengths = Lengths(kb_meta["tokenizer"], [KnowledgeSearchEnv.search])
    corpus_ids = load_corpus_ids(lengths.tokenizer)
    results = {"kb": check_kb(kb, kb_meta, lengths)}
    for path in cells:
        results[path.name] = check_cell(path, kb, kb_meta, lengths, corpus_ids)
    for name, errors in results.items():
        print(f"{name}: {'OK' if not errors else 'FAIL'}")
        for error in errors[:10]:
            print(f"  {error}")
    raise SystemExit(1 if any(results.values()) else 0)


if __name__ == "__main__":
    main()
