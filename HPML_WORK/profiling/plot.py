"""Generate per-run figures, an HTML demo report, and matched benchmark comparisons."""

import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .summarize import summarize_run, write_table


ORDER = ("baseline", "baseline_cache", "awake", "awake_cache")
PHASES = {
    "sync_weights": "Weight sync (exclusive)",
    "engine_generate": "Engine generation",
    "engine_sleep": "Sleep / wake / reset",
    "engine_wake": "Sleep / wake / reset",
    "cache_reset": "Sleep / wake / reset",
    "cpu_tool": "CPU tools",
    "environment_reset": "Environment reset",
    "reward": "Rewards",
    "quality_sample": "Rewards",
    "policy_logprob": "Policy logprob",
    "loss_forward": "Forward / loss (exclusive)",
    "backward": "Backward",
    "optimizer_update": "Optimizer update",
}


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def load_run(path):
    summarize_run(path)
    return json.loads((path / "manifest.json").read_text()), json.loads((path / "summary.json").read_text())


def breakdown(details):
    groups = defaultdict(float)
    for name, duration in details["exclusive_ms_per_step"].items():
        groups[PHASES.get(name, "Other host / bookkeeping")] += duration
    return dict(groups)


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def plot_run(path):
    path = Path(path)
    manifest, details = load_run(path)
    summary = details["summary"]
    plt = pyplot()
    figures = path / "figures"
    figures.mkdir(exist_ok=True)
    fixture = manifest["kind"] != "gpu_training"
    tag = "CPU FIXTURE - NOT GPU RESULTS" if fixture else f"{summary['variant']} / {summary['mode']}"
    title = f"{tag} | {Path(summary['cell']).name}"

    def save(fig, name):
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(figures / name, dpi=160)
        plt.close(fig)

    steps = details["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([s["step"] for s in steps], [s["step_ms"] for s in steps], "o-", label="Optimizer step")
    axes[0].plot(
        [r["step"] for r in details["rollouts"]],
        [r["rollout_lifecycle_ms"] for r in details["rollouts"]],
        ".--",
        label="Rollout + cleanup",
    )
    axes[0].set(xlabel="Optimizer step (warmup excluded)", ylabel="Wall time (ms)")
    axes[0].legend()
    axes[1].plot([s["step"] for s in steps], [s["generated_tokens_per_s"] for s in steps], "o-")
    axes[1].set(xlabel="Optimizer step", ylabel="Generated tokens / step-second")
    save(fig, "performance.png")

    groups = breakdown(details)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(groups)
    values = list(groups.values())
    bars = ax.barh(labels, values, color="#3979a8")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.invert_yaxis()
    ax.set_xlabel("Mean exclusive host wall time per measured step (ms); not GPU kernel time")
    ax.margins(x=0.2)
    save(fig, "breakdown.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([s["step"] for s in steps], [s["accuracy"] for s in steps], "o-", label="Train answer score")
    logs = [
        r
        for r in details["trainer_logs"]
        if r.get("reward") is not None and r["step"] > manifest["config"]["warmup_steps"]
    ]
    if logs:
        axes[0].plot([r["step"] for r in logs], [r["reward"] for r in logs], ".--", label="Weighted training reward")
    if summary["heldout_accuracy"] is not None:
        axes[0].axhline(
            summary["heldout_accuracy"],
            color="#b45c37",
            linestyle=":",
            label=f"Held-out answer score (n={summary['heldout_samples']})",
        )
    axes[0].set(xlabel="Optimizer step", ylabel="Reward / answer score")
    axes[0].legend(fontsize=8)
    axes[1].plot([s["step"] for s in steps], [s["tool_calls_mean"] for s in steps], "o-")
    axes[1].set(xlabel="Optimizer step", ylabel="Actual tool calls / trajectory")
    save(fig, "quality.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    device = read_csv(path / "memory.csv")
    allocator = read_csv(path / "allocator_memory.csv")
    for rows, key, label in (
        (device, "used_bytes", "NVML device used (sampled)"),
        (allocator, "allocated_bytes", "PyTorch allocated (step boundaries)"),
        (allocator, "reserved_bytes", "PyTorch reserved (step boundaries)"),
    ):
        if rows:
            ax.plot([int(r["time_ns"]) / 1e9 for r in rows], [int(r[key]) / 2**30 for r in rows], ".-", label=label)
    ax.set(xlabel="Seconds since recorder creation", ylabel="GiB")
    if device or allocator:
        ax.legend(fontsize=8)
    if not device:
        ax.text(0.01, 0.95, "NVML unavailable/disabled; no device peak inferred", transform=ax.transAxes, va="top")
    save(fig, "memory.png")

    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    selected_step = steps[0]["step"]
    selected = [e for e in events if e["step"] == selected_step and e.get("phase", "train") == "train"]
    origin = min(e["start_ns"] for e in selected)
    lanes = [
        "sync_weights",
        "engine_wake",
        "engine_generate",
        "cpu_tool",
        "engine_sleep",
        "policy_logprob",
        "reward",
        "loss_forward",
        "backward",
        "optimizer_update",
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    for index, lane in enumerate(lanes):
        intervals = [((e["start_ns"] - origin) / 1e6, e["duration_ms"]) for e in selected if e["name"] == lane]
        ax.broken_barh(intervals, (index - 0.35, 0.7), facecolors=f"C{index % 10}")
    ax.set_yticks(range(len(lanes)), lanes)
    ax.invert_yaxis()
    ax.set(xlabel=f"Host timeline of measured step {selected_step} (ms); nested spans can overlap")
    save(fig, "timeline.png")
    write_report(path, manifest, details)
    print(f"Demo report: {path / 'report.html'}")
    return path / "report.html"


def write_report(path, manifest, details):
    summary = details["summary"]
    escape = html.escape

    def fmt(value):
        return "unavailable" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    keys = [
        "step_median_ms",
        "step_p95_ms",
        "rollout_median_ms",
        "generated_tokens_per_s",
        "trajectories_per_s",
        "device_peak_used_bytes",
        "accuracy",
        "heldout_accuracy",
        "heldout_samples",
        "syncs_per_rollout",
        "actual_tool_calls_mean",
        "lifecycle_mismatches",
    ]
    table = "".join(f"<tr><th>{escape(key)}</th><td>{escape(fmt(summary[key]))}</td></tr>" for key in keys)
    demos_path = path / "demo-trajectories.json"
    demos = json.loads(demos_path.read_text()) if demos_path.exists() else []
    interactions = []
    for demo in demos:
        messages = [*(demo.get("prompt") or []), *demo["completion"]]
        blocks = []
        for message in messages:
            text = str(message.get("content", ""))
            if message.get("tool_calls"):
                text += "\n" + json.dumps(message["tool_calls"], ensure_ascii=False, indent=2)
            blocks.append(f"<b>{escape(message['role'])}</b><pre>{escape(text)}</pre>")
        interactions.append(
            f"<details><summary>{escape(demo['phase'])} / sample {escape(str(demo['sample_id']))} "
            f"/ answer score {demo['correct']} / tools {demo['tool_calls']}</summary>{''.join(blocks)}</details>"
        )
    images = "".join(
        f'<figure><img src="figures/{name}.png" alt="{name}"></figure>'
        for name in ("performance", "breakdown", "memory", "quality", "timeline")
    )
    traces = "".join(
        f'<li><a href="{p.relative_to(path).as_posix()}">{escape(p.name)}</a></li>'
        for p in sorted((path / "traces").glob("*"))
    )
    label = (
        "CPU FIXTURE - NOT GPU PERFORMANCE"
        if manifest["kind"] != "gpu_training"
        else f"{summary['variant']} / {summary['mode']}"
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>HPML profiling report</title>
<style>body{{font:16px system-ui;max-width:1160px;margin:40px auto;padding:0 24px;color:#182b3a}}
table{{border-collapse:collapse}}td,th{{text-align:left;padding:8px 16px;border-bottom:1px solid #ddd}}
img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:12px;max-height:420px;overflow:auto}}
details{{margin:16px 0}}.note{{background:#fff3d9;padding:16px}}figure{{margin:24px 0}}</style>
<h1>HPML profiling: {escape(label)}</h1>
<p>{escape(summary["model"])} · {escape(summary["cell"])} · {escape(manifest["gpu"])}</p>
<p>Measured steps: {summary["measured_steps"]}; warmup excluded: {manifest["config"]["warmup_steps"]}.
Commit: {escape(manifest.get("git_commit", "fixture"))}</p>
<p class="note">Use unprofiled benchmark runs for speedup claims. Breakdown bars are exclusive host wall time.
Generation combines prefill and decode. NVML is device-wide and sampled; PyTorch memory covers its allocator.
Held-out scores use Yidan's workload-specific reward, which is lenient for W3. Missing values are not zero.</p>
<table>{table}</table>{images}<h2>Real interaction examples</h2>{"".join(interactions) or "<p>No saved interactions.</p>"}
<h2>Profiler artifacts</h2><ul>{traces}</ul>
<p>Open trace JSON in <a href="https://ui.perfetto.dev">Perfetto</a>.
Open memory-snapshot.pickle in <a href="https://pytorch.org/memory_viz">PyTorch memory_viz</a>, if collected.</p>
<p><a href="manifest.json">Manifest</a> · <a href="summary.json">Summary</a> ·
<a href="steps.csv">Steps</a> · <a href="rollouts.csv">Rollouts</a> · <a href="quality.csv">Quality</a></p></html>"""
    (path / "report.html").write_text(document, encoding="utf-8")


def validate_comparison(rows):
    if len({r["comparison_key"] for r in rows}) != 1:
        raise ValueError(
            "Runs differ in workload, split, model, hardware, dependencies, harness or measurement settings"
        )
    if any(
        r["kind"] != "gpu_training" or r["mode"] != "benchmark" or r["timing_mode"] != "throughput" or r["profiled"]
        for r in rows
    ):
        raise ValueError(
            "Comparison requires unprofiled GPU benchmark runs; fixtures/profile/nsight runs are excluded"
        )
    pairs = [(r["variant"], r["seed"]) for r in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate variant/seed: use distinct seeds for independent comparison points")


def compare_runs(paths, output):
    loaded = [load_run(Path(path)) for path in paths]
    rows = [details["summary"] for _, details in loaded]
    validate_comparison(rows)
    output.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    order = [variant for variant in ORDER if any(r["variant"] == variant for r in rows)]
    metrics = [
        ("rollout_median_ms", "Rollout + cleanup (ms)"),
        ("step_median_ms", "Optimizer step (ms)"),
        ("generated_tokens_per_s", "Generated tokens / step-second"),
        ("heldout_accuracy", "Held-out answer score"),
        ("syncs_per_rollout", "Syncs / rollout"),
        ("device_peak_used_bytes", "Sampled device peak (GiB)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (metric, label) in zip(axes.flat, metrics, strict=True):
        for index, variant in enumerate(order):
            values = [r[metric] for r in rows if r["variant"] == variant and r[metric] is not None]
            if metric == "device_peak_used_bytes":
                values = [v / 2**30 for v in values]
            if values:
                ax.scatter([index] * len(values), values, color="#3979a8")
                ax.plot([index - 0.18, index + 0.18], [statistics.mean(values)] * 2, color="black")
        ax.set_xticks(range(len(order)), order, rotation=20)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"{rows[0]['model']} | {Path(rows[0]['cell']).name}\nDots = independent runs; line = mean; no confidence interval"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)
    write_table(output / "comparison.csv", rows)

    baseline = {r["seed"]: r for r in rows if r["variant"] == "baseline"}
    speedups = []
    for row in rows:
        if row["variant"] == "baseline" or row["seed"] not in baseline:
            continue
        base = baseline[row["seed"]]
        speedups.append(
            dict(
                variant=row["variant"],
                seed=row["seed"],
                step_speedup=base["step_median_ms"] / row["step_median_ms"],
                rollout_speedup=base["rollout_median_ms"] / row["rollout_median_ms"],
                throughput_ratio=row["generated_tokens_per_s"] / base["generated_tokens_per_s"]
                if base["generated_tokens_per_s"]
                else None,
            )
        )
    write_table(output / "paired-speedups.csv", speedups)
    print(f"Comparison: {output / 'comparison.png'}; paired speedups use matching seeds only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Required for comparisons of multiple benchmark runs")
    args = parser.parse_args()
    for path in args.runs:
        plot_run(path)
    if len(args.runs) > 1:
        if args.output_dir is None:
            parser.error("--output-dir is required for a multi-run comparison")
        compare_runs(args.runs, args.output_dir)


if __name__ == "__main__":
    main()
