"""Aggregate the issue #7428 benchmark outputs into plot-ready CSVs under results/issue7428/tables/.

- runs.csv        one row per training run: medians over steps 1.. (step 0 is warmup), plus totals
- steps.csv       one row per (run, step)
- turns.csv       one row per (run, step, turn): llm.generate time and prompt / cached / completion tokens
- live.csv        one row per (label, turns, iteration) from the live benchmark
- live_summary.csv  medians per (label, turns)
- comparison.csv  baseline vs fixed side by side per configuration, with speedups

Run from the repository root: python HPML_WORK/benchmark_7428/aggregate.py [--results-dir results/issue7428]
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path("results/issue7428")  # overridden by --results-dir
OUT = ROOT / "tables"
WARMUP = 1


def write(name, rows):
    if not rows:
        return
    keys = list(rows[0])
    for row in rows[1:]:
        keys += [k for k in row if k not in keys]
    with open(OUT / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def main():
    OUT.mkdir(exist_ok=True)
    runs, steps_rows, turns_rows = [], [], []
    for summary_path in sorted((ROOT / "train").glob("*/*/summary.json")):
        run_dir = summary_path.parent
        if run_dir.parent.name == "smoke":
            continue
        summary = json.loads(summary_path.read_text())
        steps = json.loads((run_dir / "steps.json").read_text())
        # wake_up(tags=["weights"]) runs inside sync_weights(), so its time is already in sync_ms: split the wake-ups
        # by tag so that the breakdown does not count it twice
        wake = {}
        for line in (run_dir / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["event"] == "llm.wake_up":
                key = (e["step"], "weights" if e["tags"] == ["weights"] else "kv_cache")
                wake[key] = wake.get(key, 0.0) + e["ms"]
        for s in steps:
            s["wake_weights_ms"] = round(wake.get((s["step"], "weights"), 0.0), 3)
            s["wake_kv_cache_ms"] = round(wake.get((s["step"], "kv_cache"), 0.0), 3)
        config = run_dir.name  # e.g. w1_K4_L256_P256_1.7B_prefix-on
        workload, cell = summary["cell"].split("/")
        prefix = "off" if summary["args"]["disable_prefix_caching"] else "on"
        base = {
            "config": config,
            "label": summary["label"],
            "workload": workload,
            "cell": cell,
            "prefix_caching": prefix,
            "model": summary["args"]["model"],
        }
        for s in steps:
            cached_after_first = sum(c or 0 for c in s["turn_cached_tokens"][1:])
            steps_rows.append(
                {
                    **base,
                    **{k: v for k, v in s.items() if not k.startswith("turn_")},
                    # Disjoint engine components: sync_ms includes the weight wake-up, so only the KV wake-up is added
                    "engine_ms": round(
                        s["sync_ms"] + s["wake_kv_cache_ms"] + s["llm_sleep_ms"] + s["llm_generate_ms"], 3
                    ),
                    "tools_scoring_other_ms": round(
                        s["generate_and_score_ms"]
                        - s["sync_ms"]
                        - s["wake_kv_cache_ms"]
                        - s["llm_sleep_ms"]
                        - s["llm_generate_ms"],
                        3,
                    ),
                    "train_compute_ms": round(s["step_ms"] - s["generate_and_score_ms"], 3) if s["step_ms"] else None,
                    "prompt_tokens": sum(s["turn_prompt_tokens"]),
                    "cached_tokens": sum(c or 0 for c in s["turn_cached_tokens"]),
                    "cached_tokens_after_turn1": cached_after_first,
                    "completion_tokens": sum(s["turn_completion_tokens"]),
                }
            )
            for t, ms in enumerate(s["turn_ms"]):
                turns_rows.append(
                    {
                        **base,
                        "step": s["step"],
                        "turn": t,
                        "llm_generate_ms": ms,
                        "requests": s["turn_requests"][t],
                        "prompt_tokens": s["turn_prompt_tokens"][t],
                        "cached_tokens": s["turn_cached_tokens"][t],
                        "completion_tokens": s["turn_completion_tokens"][t],
                    }
                )
        measured = [r for r in steps_rows if r["config"] == config and r["label"] == summary["label"]][WARMUP:]
        metrics = [
            "step_ms",
            "rollout_ms",
            "generate_and_score_ms",
            "train_compute_ms",
            "num_syncs",
            "sync_ms",
            "num_llm_sleeps",
            "llm_sleep_ms",
            "num_llm_wake_ups",
            "llm_wake_up_ms",
            "num_turns",
            "wake_weights_ms",
            "wake_kv_cache_ms",
            "llm_generate_ms",
            "tools_scoring_other_ms",
            "prompt_tokens",
            "cached_tokens",
            "cached_tokens_after_turn1",
            "tool_calls_per_sample",
            "followed_fraction",
            "reward",
            "mean_completion_length",
        ]
        runs.append(
            {
                **base,
                "git_commit": summary["git_commit"][:8],
                "steps": summary["completed_steps"],
                "train_wall_s": summary["train_wall_s"],
                "peak_gpu_used_mib": summary["peak_gpu_used_mib"],
                **{f"median_{m}": median(r[m] for r in measured) for m in metrics},
                "total_sync_ms": round(sum(r["sync_ms"] for r in measured), 3),
                "total_step_ms": round(sum(r["step_ms"] or 0 for r in measured), 3),
            }
        )
    write("runs.csv", runs)
    write("steps.csv", steps_rows)
    write("turns.csv", turns_rows)

    comparison = []
    for config in sorted({r["config"] for r in runs}):
        pair = {r["label"]: r for r in runs if r["config"] == config}
        if not {"baseline", "fixed"} <= pair.keys():
            continue
        b, f = pair["baseline"], pair["fixed"]
        row = {"config": config, "workload": b["workload"], "cell": b["cell"], "prefix_caching": b["prefix_caching"]}
        for m in [
            "median_step_ms",
            "median_rollout_ms",
            "median_num_syncs",
            "median_sync_ms",
            "median_llm_generate_ms",
            "median_num_turns",
            "median_tool_calls_per_sample",
            "median_followed_fraction",
            "peak_gpu_used_mib",
            "train_wall_s",
        ]:
            row[f"baseline_{m}"], row[f"fixed_{m}"] = b[m], f[m]
        for m in ["median_step_ms", "median_rollout_ms"]:
            row[f"speedup_{m.removeprefix('median_')}"] = round(b[m] / f[m], 3)
            row[f"saved_pct_{m.removeprefix('median_')}"] = round(100 * (1 - f[m] / b[m]), 1)
        comparison.append(row)
    write("comparison.csv", comparison)

    live_rows, live_summary = [], []
    for path in sorted((ROOT / "live").glob("*/turns*.json")):
        report = json.loads(path.read_text())
        for i, it in enumerate(report["iterations"]):
            live_rows.append(
                {
                    "label": report["label"],
                    "turns": report["turns_per_rollout"],
                    "iteration": i,
                    "behavior": report["behavior"],
                    "syncs": it["syncs"],
                    "sleeps_inside_rollout": it["sleeps"],
                    "sleeps_after_rollout": it["sleeps_after_rollout"],
                    "sync_ms_total": round(sum(it["sync_ms"]), 3),
                    "rollout_ms": it["rollout_ms"],
                    **{f"turn{t}_ms": ms for t, ms in enumerate(it["turn_ms"])},
                }
            )
        live_summary.append(
            {
                "label": report["label"],
                "turns": report["turns_per_rollout"],
                "behavior": report["behavior"],
                # run_live.sh adds the commit of the TRL actually imported; source_commit is the commit of the script's repo
                "git_commit": report.get("trl_git_commit", report["source_commit"])[:8],
                "gpu": report["gpu"],
                "median_rollout_ms": report["summary"]["median_rollout_ms"],
                "median_sync_ms_total": median(sum(it["sync_ms"]) for it in report["iterations"]),
                "median_syncs": median(it["syncs"] for it in report["iterations"]),
                **{f"median_turn{t}_ms": ms for t, ms in enumerate(report["summary"]["median_turn_ms"])},
            }
        )
    write("live.csv", live_rows)
    write("live_summary.csv", live_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT)
    ROOT = parser.parse_args().results_dir
    OUT = ROOT / "tables"
    main()
