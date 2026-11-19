"""Reduce raw runs to per-run CSV. Warmup is excluded by optimizer-step ID, never by event count."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .recorder import write_json


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def reduce_events(events, warmup_steps, variant=None):
    steps = [
        e
        for e in events
        if e["name"] == "optimizer_step"
        and e["step"] > warmup_steps
        and e.get("phase", "train") == "train"
        and e["status"] == "ok"
    ]
    selected = {event["step"] for event in steps}
    measured = [
        e for e in events if e["step"] in selected and e["status"] == "ok" and e.get("phase", "train") == "train"
    ]
    if not steps:
        raise ValueError("No completed measured optimizer steps remain after warmup")
    by_name = defaultdict(list)
    for event in measured:
        by_name[event["name"]].append(event)
    parents = {e["event_id"]: e for e in events}

    def inside_rollout(event):
        parent = event["parent_id"]
        while parent is not None:
            ancestor = parents[parent]
            if ancestor["name"] == "rollout":
                return True
            parent = ancestor["parent_id"]
        return False

    rollout_rows = []
    for event in by_name["rollout"]:
        related = [e for e in measured if e["rollout"] == event["rollout"]]
        sleeps = [e for e in related if e["name"] == "engine_sleep"]
        cleanup_ms = sum(e["duration_ms"] for e in sleeps if not inside_rollout(e))
        rollout_rows.append(
            {
                "step": event["step"],
                "rollout": event["rollout"],
                "rollout_raw_ms": event["duration_ms"],
                "cleanup_ms": cleanup_ms,
                "rollout_lifecycle_ms": event["duration_ms"] + cleanup_ms,
                "generation_calls": sum(e["name"] == "generation_turn" for e in related),
                "sync_calls": sum(e["name"] == "sync_weights" for e in related),
                "wake_calls": sum(e["name"] == "engine_wake" for e in related),
                "cache_resets": sum(e["name"] == "cache_reset" for e in related),
                "trajectories": event["fields"]["trajectories"],
                "generated_tokens": sum(
                    e["fields"]["generated_tokens"] for e in related if e["name"] == "generation_turn"
                ),
                "prompt_tokens": sum(
                    e["fields"].get("prompt_tokens", 0) for e in related if e["name"] == "generation_turn"
                ),
                "sleep_inside": sum(inside_rollout(e) for e in sleeps),
                "sleep_after": sum(not inside_rollout(e) for e in sleeps),
            }
        )
    total_ms = sum(e["duration_ms"] for e in steps)
    total_tokens = sum(e["fields"]["generated_tokens"] for e in by_name["generation_turn"])
    trajectories = sum(e["fields"]["trajectories"] for e in by_name["rollout"])
    quality = [e["fields"] for e in by_name["quality_sample"]]
    exclusive = defaultdict(float)
    for event in measured:
        exclusive[event["name"]] += event["exclusive_ms"]
    eval_quality = [
        e["fields"]
        for e in events
        if e["name"] == "quality_sample" and e.get("phase") == "eval" and e["status"] == "ok"
    ]
    mismatches = 0
    for row in rollout_rows:
        expected = None
        if variant and row["generation_calls"]:
            expected = (
                (row["generation_calls"], row["generation_calls"], 0) if variant.startswith("baseline") else (1, 0, 1)
            )
        row["lifecycle_matches"] = (
            None if expected is None else (row["sync_calls"], row["sleep_inside"], row["sleep_after"]) == expected
        )
        mismatches += row["lifecycle_matches"] is False
    # Sum exclusive nested spans once. Inclusive sync/generation/rollout times overlap.
    summary = {
        "measured_steps": len(steps),
        "measured_rollouts": len(rollout_rows),
        "step_median_ms": statistics.median(e["duration_ms"] for e in steps),
        "step_p95_ms": percentile([e["duration_ms"] for e in steps], 0.95),
        "rollout_median_ms": percentile([e["rollout_lifecycle_ms"] for e in rollout_rows], 0.5),
        "rollout_p95_ms": percentile([e["rollout_lifecycle_ms"] for e in rollout_rows], 0.95),
        "generated_tokens": total_tokens,
        "trajectories": trajectories,
        "generated_tokens_per_s": total_tokens / (total_ms / 1000),
        "trajectories_per_s": trajectories / (total_ms / 1000),
        "sync_calls": len(by_name["sync_weights"]),
        "syncs_per_rollout": len(by_name["sync_weights"]) / len(rollout_rows) if rollout_rows else None,
        "generation_calls": len(by_name["generation_turn"]),
        "accuracy": statistics.mean(q["correct"] for q in quality) if quality else None,
        "fully_followed_rate": statistics.mean(q["fully_followed"] for q in quality) if quality else None,
        "actual_tool_calls_mean": statistics.mean(q["tool_calls"] for q in quality) if quality else None,
        "actual_tool_calls_p95": percentile([q["tool_calls"] for q in quality], 0.95),
        "tool_errors": sum(q.get("tool_errors", 0) for q in quality),
        "quality_samples": len(quality),
        "torch_peak_allocated_bytes": max(
            (e["fields"].get("torch_peak_allocated_bytes", 0) for e in steps), default=None
        )
        or None,
        "torch_peak_reserved_bytes": max(
            (e["fields"].get("torch_peak_reserved_bytes", 0) for e in steps), default=None
        )
        or None,
        "device_peak_used_bytes": None,
        "prefill_tokens": None,
        "cache_hit_rate": None,
        "heldout_accuracy": statistics.mean(q["correct"] for q in eval_quality) if eval_quality else None,
        "heldout_samples": len(eval_quality),
        "heldout_fully_followed_rate": statistics.mean(q["fully_followed"] for q in eval_quality)
        if eval_quality
        else None,
        "heldout_tool_calls_mean": statistics.mean(q["tool_calls"] for q in eval_quality) if eval_quality else None,
        "evaluation_wall_ms": sum(e["duration_ms"] for e in events if e["name"] == "evaluation") or None,
        "train_loop_wall_ms": sum(e["duration_ms"] for e in events if e["name"] == "train_loop") or None,
        "lifecycle_mismatches": mismatches,
    }
    step_rows = []
    for step in steps:
        children = [e for e in measured if e["step"] == step["step"]]
        q = [e["fields"] for e in children if e["name"] == "quality_sample"]
        tokens = sum(e["fields"]["generated_tokens"] for e in children if e["name"] == "generation_turn")
        step_rows.append(
            dict(
                step=step["step"],
                step_ms=step["duration_ms"],
                generated_tokens=tokens,
                generated_tokens_per_s=tokens / (step["duration_ms"] / 1000),
                accuracy=statistics.mean(r["correct"] for r in q) if q else None,
                tool_calls_mean=statistics.mean(r["tool_calls"] for r in q) if q else None,
                **step["fields"],
            )
        )
    log_rows = [
        {**e["fields"]["logs"], "step": e["fields"]["global_step"]}
        for e in events
        if e["name"] == "trainer_log" and e.get("phase", "train") == "train"
    ]
    details = {
        "rollouts": rollout_rows,
        "steps": step_rows,
        "trainer_logs": log_rows,
        "exclusive_ms_per_step": {k: v / len(steps) for k, v in exclusive.items()},
        "tool_calls_histogram": dict(Counter(q["tool_calls"] for q in quality)),
        "note": "Exclusive host wall times partition measured steps; they are not GPU kernel times. Prefill/decode remain combined.",
    }
    return summary, details


def summarize_run(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Refusing incomplete/failed run: {path}")
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    config = manifest["config"]
    summary, details = reduce_events(events, config["warmup_steps"], config["variant"])
    memory_path = path / "memory.csv"
    if memory_path.exists():
        with memory_path.open() as stream:
            samples = list(csv.DictReader(stream))
        selected = {str(s["step"]) for s in details["steps"]}
        memory = [int(s["used_bytes"]) for s in samples if s["step"] in selected and s["phase"] == "train"]
        summary["device_peak_used_bytes"] = max(memory) if memory else None
    # Source commit and variant intentionally differ across A/B/C; all other settings must match.
    matched = {
        "config": {k: v for k, v in config.items() if k not in ("variant", "seed", "cell")},
        "data": manifest["data_meta"]["sha256"],
        "gpu": manifest["gpu"],
        "gpu_inventory": manifest.get("gpu_inventory"),
        "python": manifest.get("python"),
        "cuda_runtime": manifest.get("cuda_runtime"),
        "packages": manifest["packages"],
        "tokenizer": manifest.get("tokenizer"),
        "kind": manifest["kind"],
        "torch_profile": manifest["torch_profile"],
        "nvtx": manifest["nvtx"],
        "harness": manifest.get("harness_hashes"),
        "split": manifest.get("split_sha256"),
        "budgets": manifest["data_meta"].get("grpo_config"),
    }
    summary.update(
        run=str(path),
        variant=config["variant"],
        seed=config["seed"],
        model=config["model"],
        cell=config["cell"],
        kind=manifest["kind"],
        timing_mode=config["timing_mode"],
        profiled=manifest["torch_profile"] or manifest["nvtx"] or config.get("memory_snapshot", False),
        mode=config.get("mode", "benchmark"),
        comparison_key=hashlib.sha256(json.dumps(matched, sort_keys=True).encode()).hexdigest()[:16],
    )
    write_json(path / "summary.json", {"summary": summary, **details})
    write_table(path / "steps.csv", details["steps"])
    write_table(path / "rollouts.csv", details["rollouts"])
    write_table(path / "trainer-logs.csv", details["trainer_logs"])
    quality = [
        {"phase": e.get("phase", "train"), "step": e["step"], "rollout": e["rollout"], **e["fields"]}
        for e in events
        if e["name"] == "quality_sample"
    ]
    write_table(path / "quality.csv", quality)
    return summary


def write_table(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [summarize_run(path) for path in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
