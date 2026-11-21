"""Static checks and scripted-policy replay for the W1 datasets (CPU only).

python HPML_WORK/data/w1_check.py                 # every cell under out/w1
python HPML_WORK/data/w1_check.py out/w1/K2_L64_P256
"""

import argparse
import json
import re
from pathlib import Path

from common import Lengths, load_corpus_ids, rows_sha256
from datasets import load_from_disk
from w1_env import ChainArithmeticEnv, answer_reward
from w1_generate import OUT_DIR, cell_meta, generate_cell


def replay(row):
    """Scripted policy: follow the ids it reads from the tool results, then answer. Returns (calls, env, reward)."""
    env = ChainArithmeticEnv()
    env.reset(**row)
    value, op_id, calls = row["x0"], row["op_ids"][0], 0
    while op_id is not None:
        result = env.get_op(op_id)
        calls += 1
        value += int(re.match(r"\[step \d+/\d+\] op: ([+-]\d+)\n", result).group(1))
        next_id = re.match(r".*\nnext_id: (\w+)\n---\n", result)
        op_id = next_id.group(1) if next_id else None
    completion = [{"role": "assistant", "content": f"<answer>{value}</answer>"}]
    return calls, env, answer_reward([completion], [row["answer"]])[0]


def check_cell(path, lengths, corpus_ids):
    rows = load_from_disk(str(path)).to_list()
    meta = json.loads((path / "meta.json").read_text())
    K, L_obs, P = meta["K"], meta["L_obs"], meta["P"]
    errors = []
    for n, row in enumerate(rows):
        values = [row["x0"]]
        for op in row["ops"]:
            values.append(values[-1] + op)
        if values[-1] != row["answer"] or not all(0 <= v <= 199 for v in values):
            errors.append(f"row {n}: answer or intermediate values wrong")
        if len(set(row["op_ids"])) != K or len(row["ops"]) != K or len(row["observations"]) != K:
            errors.append(f"row {n}: wrong chain length or duplicate op_ids")
        # Each id must be reachable only from its predecessor, otherwise the model can skip or parallelize calls
        user = row["prompt"][1]["content"]
        for i, op_id in enumerate(row["op_ids"]):
            sources = ([user] if i == 0 else []) + [obs for j, obs in enumerate(row["observations"]) if j == i - 1]
            others = [user] + row["observations"]
            if sum(op_id in text for text in others) != 1 or not any(op_id in text for text in sources):
                errors.append(f"row {n}: op_id {op_id} leaks outside its predecessor")
        lens = {lengths.tool_suffix(obs) for obs in row["observations"]}
        if lens != {L_obs}:
            errors.append(f"row {n}: tool suffix lengths {lens} != {L_obs}")
        if lengths.system(row["prompt"][0]["content"]) != P:
            errors.append(f"row {n}: system prefix length != {P}")
        calls, env, reward = replay(row)
        if calls != K or env.steps_followed != K or reward != 1.0:
            errors.append(f"row {n}: replay calls={calls} steps_followed={env.steps_followed} reward={reward}")
    if len({row["prompt"][0]["content"] for row in rows}) != 1:
        errors.append("system prompt differs across samples")
    if rows_sha256(rows) != meta["sha256"]:
        errors.append("saved rows do not match meta sha256")
    # Regenerate from the seed: data and derived budgets must be identical
    regenerated = generate_cell(lengths, corpus_ids, K, L_obs, P, meta["seed"], meta["num_samples"])
    if cell_meta(lengths, regenerated, meta["tokenizer"]) != meta:
        errors.append("regenerating with the same seed gives a different dataset or budget")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="*", type=Path)
    args = parser.parse_args()
    cells = args.cells or sorted(path for path in OUT_DIR.iterdir() if path.is_dir())

    tokenizer = json.loads((cells[0] / "meta.json").read_text())["tokenizer"]
    lengths = Lengths(tokenizer, [ChainArithmeticEnv.get_op])
    corpus_ids = load_corpus_ids(lengths.tokenizer)
    failed = False
    for path in cells:
        errors = check_cell(path, lengths, corpus_ids)
        failed |= bool(errors)
        print(f"{path.name}: {'OK' if not errors else 'FAIL'}")
        for error in errors[:10]:
            print(f"  {error}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
