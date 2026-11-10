"""Static checks and reference replay for the W3 MuSiQue datasets (CPU only).

python HPML_WORK/data/w3_check.py                 # every cell under out/w3
"""

import argparse
import json
from pathlib import Path

from common import Lengths, load_corpus_ids, rows_sha256
from datasets import load_from_disk
from w3_env import MusiqueSearchEnv, answer_reward, reference_trace
from w3_generate import OUT_DIR, cell_meta, generate_cell, load_musique


def check_cell(path, lengths, corpus_ids, musique):
    rows = load_from_disk(str(path)).to_list()
    meta = json.loads((path / "meta.json").read_text())
    errors = []
    for n, row in enumerate(rows):
        if lengths.system(row["prompt"][0]["content"]) != meta["P"]:
            errors.append(f"row {n}: system prefix length != {meta['P']}")
        messages, env = reference_trace(**row)
        calls = sum(m["role"] == "tool" for m in messages)
        reward = answer_reward([messages], [row["answer"]], [row["answer_aliases"]])[0]
        if calls != row["K"] or env.found != set(row["supporting_idx"]) or reward != 1.0:
            errors.append(f"row {n}: replay calls={calls} found={sorted(env.found)} reward={reward}")
    if len({row["prompt"][0]["content"] for row in rows}) != 1:
        errors.append("system prompt differs across samples")
    if rows_sha256(rows) != meta["sha256"]:
        errors.append("saved rows do not match meta sha256")
    regenerated, num_candidates = generate_cell(
        lengths, corpus_ids, musique, meta["K"], meta["top_k"], meta["P"], meta["seed"], meta["num_samples"]
    )
    if cell_meta(lengths, regenerated, num_candidates, meta["tokenizer"]) != meta:
        errors.append("regenerating with the same seed gives a different dataset or budget")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="*", type=Path)
    args = parser.parse_args()
    cells = args.cells or sorted(path for path in OUT_DIR.iterdir() if path.is_dir())

    tokenizer = json.loads((cells[0] / "meta.json").read_text())["tokenizer"]
    lengths = Lengths(tokenizer, [MusiqueSearchEnv.search])
    corpus_ids = load_corpus_ids(lengths.tokenizer)
    musique = load_musique()
    failed = False
    for path in cells:
        errors = check_cell(path, lengths, corpus_ids, musique)
        failed |= bool(errors)
        print(f"{path.name}: {'OK' if not errors else 'FAIL'}")
        for error in errors[:10]:
            print(f"  {error}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
