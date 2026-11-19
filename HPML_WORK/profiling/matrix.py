"""Materialize an experiment matrix. This generates configs; it does not launch GPU jobs."""

import argparse
import itertools
import json
from pathlib import Path

from .recorder import write_json
from .run import validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("HPML_WORK/profiling/configs/smoke.json"))
    parser.add_argument("--matrix", type=Path, default=Path("HPML_WORK/profiling/configs/matrix.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, matrix = json.loads(args.base.read_text()), json.loads(args.matrix.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    for variant, seed, k, obs, prefix in itertools.product(
        *(matrix[key] for key in ("variants", "seeds", "K", "L_obs", "P"))
    ):
        config = {
            **base,
            "variant": variant,
            "seed": seed,
            "cell": f"HPML_WORK/data_prepare/out/w1/K{k}_L{obs}_P{prefix}",
            "max_steps": matrix["max_steps"],
            "warmup_steps": matrix["warmup_steps"],
        }
        validate_config(config)
        name = f"{variant}-s{seed}-K{k}-L{obs}-P{prefix}.json"
        write_json(args.output / name, config)
    print(f"Configs written to {args.output}; select the matching source lifecycle before running each variant.")


if __name__ == "__main__":
    main()
