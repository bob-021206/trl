"""Generate styled run figures, an HTML report, and matched benchmark comparisons."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .summarize import summarize_run, write_table


# All data series use the four project colors; white is only a background/marker fill.
INK, OLIVE, GOLD, CORAL = PALETTE = ("#4C4B16", "#898121", "#E6C767", "#F87A53")
ORDER = ("baseline", "baseline_cache", "awake", "awake_cache")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "baseline_cache": "Baseline + APC",
    "awake": "Awake",
    "awake_cache": "Awake + APC",
}
VARIANT_COLORS = {"baseline": INK, "baseline_cache": GOLD, "awake": OLIVE, "awake_cache": CORAL}
VARIANT_MARKERS = {"baseline": "o", "baseline_cache": "D", "awake": "s", "awake_cache": "^"}
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
FAMILIES = {
    "Lifecycle": (INK, "o"),
    "Generation": (CORAL, "D"),
    "Model / gradient": (OLIVE, "s"),
    "CPU / bookkeeping": (GOLD, "^"),
}


def phase_family(label):
    if label in ("Weight sync (exclusive)", "Sleep / wake / reset"):
        return "Lifecycle"
    if label == "Engine generation":
        return "Generation"
    if label in ("Policy logprob", "Forward / loss (exclusive)", "Backward", "Optimizer update"):
        return "Model / gradient"
    return "CPU / bookkeeping"


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.labelsize": 11,
            "axes.labelpad": 9,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 2.4,
            "lines.markersize": 7,
            "lines.markeredgewidth": 1.2,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 200,
            "svg.fonttype": "none",
        }
    )
    return plt


def style_axis(ax, *, grid="both", steps=False):
    from matplotlib.ticker import MaxNLocator

    ax.grid(True, which="major", axis=grid, color=INK, linewidth=0.7, alpha=0.16)
    for spine in ax.spines.values():
        spine.set_alpha(0.28)
    ax.tick_params(axis="both", which="major", pad=6)
    if steps:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, min_n_ticks=3))
        ax.margins(x=0.06)


def finite_values(values):
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def use_log_scale(values):
    """Two orders of magnitude justify log; never hide zeros or negatives to force it."""
    values = finite_values(values)
    return bool(values) and min(values) > 0 and max(values) / min(values) >= 100


def value_axis(ax, values, *, axis="y", allow_log=True, zero=True):
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator, NullFormatter

    values = finite_values(values)
    logarithmic = allow_log and use_log_scale(values)
    target = ax.xaxis if axis == "x" else ax.yaxis
    if logarithmic:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
        target.set_major_locator(LogLocator(base=10, numticks=8))
        target.set_major_formatter(LogFormatterSciNotation(base=10))
        target.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=30))
        target.set_minor_formatter(NullFormatter())
        ax.grid(True, which="minor", axis=axis, color=INK, linewidth=0.5, alpha=0.08, linestyle=":")
    else:
        target.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
        if zero and values and min(values) >= 0:
            (ax.set_xlim if axis == "x" else ax.set_ylim)(**({"left": 0} if axis == "x" else {"bottom": 0}))
    return logarithmic


def family_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=INK,
            label=label,
        )
        for label, (color, marker) in FAMILIES.items()
    ]


def finish_figure(plt, fig, destination, title, context, note, *, handles=None):
    if handles is None:
        unique = {}
        for ax in fig.axes:
            lines, labels = ax.get_legend_handles_labels()
            unique.update(zip(labels, lines, strict=True))
        handles = list(unique.values())
    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.055, 0.925, context, ha="left", va="top", fontsize=10.5, color=INK, alpha=0.85)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.055, 0.065),
            ncol=min(4, len(handles)),
            handlelength=2.4,
            columnspacing=1.8,
            borderaxespad=0,
        )
    fig.text(0.055, 0.022, note, fontsize=9, color=INK, alpha=0.82, va="bottom")
    fig.tight_layout(rect=(0.025, 0.145 if handles else 0.105, 0.985, 0.96), w_pad=3.5)
    for suffix in (".png", ".svg"):
        fig.savefig(Path(destination).with_suffix(suffix))
    plt.close(fig)


def load_run(path):
    # Restyling existing figures should not rewrite recorded metrics or their provenance.
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Refusing incomplete/failed run: {path}")
    summary_path = path / "summary.json"
    if not summary_path.exists():
        summarize_run(path)
    return manifest, json.loads(summary_path.read_text())


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


def run_context(manifest, summary):
    cell = Path(summary["cell"])
    prefix = f"{cell.parent.name.upper()} · {cell.name}" if cell.parent.name in ("w1", "w2", "w3") else cell.name
    model = Path(summary["model"]).name
    label = VARIANT_LABELS.get(summary["variant"], summary["variant"])
    context = f"{prefix}  |  {model}  |  {label} · {summary['mode']} · seed {summary['seed']}"
    if manifest["kind"] != "gpu_training":
        context = f"CPU FIXTURE — NOT GPU RESULTS  |  {context}"
    return context


def plot_run(path):
    path = Path(path)
    manifest, details = load_run(path)
    summary, steps = details["summary"], details["steps"]
    plt = pyplot()
    figures = path / "figures"
    figures.mkdir(exist_ok=True)
    context = run_context(manifest, summary)
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    warmup = manifest["config"]["warmup_steps"]
    counts = f"{len(steps)} measured steps · {warmup} warmup steps excluded."
    if summary["mode"] != "benchmark" or summary["timing_mode"] != "throughput":
        counts += " Diagnostic capture adds overhead."

    def save(fig, name, title, note, handles=None):
        finish_figure(plt, fig, figures / name, title, context, note, handles=handles)

    # Keep the original samples: no smoothing, interpolation or outlier removal.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    step_ms = [s["step_ms"] for s in steps]
    rollout_ms = [r["rollout_lifecycle_ms"] for r in details["rollouts"]]
    axes[0].plot(
        [s["step"] for s in steps], [v / 1000 for v in step_ms], color=INK, marker="o", label="Optimizer step"
    )
    axes[0].plot(
        [r["step"] for r in details["rollouts"]],
        [v / 1000 for v in rollout_ms],
        color=CORAL,
        marker="D",
        markeredgecolor=INK,
        linestyle="--",
        label="Rollout + cleanup",
    )
    axes[0].set(title="Step and rollout latency", xlabel="Optimizer step")
    throughput = [s["generated_tokens_per_s"] for s in steps]
    axes[1].plot(
        [s["step"] for s in steps],
        throughput,
        color=OLIVE,
        marker="s",
        markeredgecolor=INK,
        label="Generated throughput",
    )
    axes[1].set(title="Generation throughput", xlabel="Optimizer step")
    for ax in axes:
        style_axis(ax, steps=True)
    log_latency = value_axis(axes[0], [v / 1000 for v in step_ms + rollout_ms])
    log_throughput = value_axis(axes[1], throughput)
    axes[0].set_ylabel("Wall time (s)" + (" · log scale" if log_latency else ""))
    axes[1].set_ylabel("Generated tokens / s" + (" · log scale" if log_throughput else ""))
    save(fig, "performance", "Latency & throughput", counts + " Tokens/s use full optimizer-step time.")

    # A point plot avoids implying that lengths measured from zero make sense on a log axis.
    groups = dict(sorted(breakdown(details).items(), key=lambda item: -item[1]))
    labels, values = list(groups), list(groups.values())
    total = sum(values)
    fig, ax = plt.subplots(figsize=(12.8, 7.6))
    style_axis(ax, grid="x")
    logarithmic = value_axis(ax, values, axis="x", zero=False)
    positive = [v for v in values if v > 0]
    left = min(positive) / 2.2 if logarithmic else 0
    right = max(values, default=1) * (2.5 if logarithmic else 1.12)
    ax.set_xlim(left, right if right else 1)
    for index, (label, value) in enumerate(groups.items()):
        color, marker = FAMILIES[phase_family(label)]
        ax.hlines(index, left, right, color=INK, linewidth=0.6, alpha=0.08)
        ax.plot(
            value,
            index,
            marker=marker,
            markersize=9,
            linestyle="none",
            color=color,
            markeredgecolor=INK,
            markeredgewidth=0.9,
            zorder=3,
        )
        share = value / total * 100 if total else 0
        share_label = "<0.1%" if 0 < share < 0.1 else f"{share:.1f}%"
        duration_label = f"{value:,.1f}" if value >= 100 else f"{value:.3g}"
        ax.text(
            1.025,
            index,
            f"{duration_label} ms   ·   {share_label}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=10,
            color=INK,
        )
    ax.text(1.025, 1.025, "Time / step   ·   Share", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlabel("Mean exclusive host time per step (ms)" + (" · log scale" if logarithmic else ""))
    save(
        fig,
        "breakdown",
        "Where the step time goes",
        counts + " Exclusive host wall time; not GPU kernel time.",
        handles=family_handles(),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    accuracy = [s["accuracy"] for s in steps]
    axes[0].plot([s["step"] for s in steps], accuracy, color=INK, marker="o", markersize=7, label="Train answer score")
    logs = [r for r in details["trainer_logs"] if r.get("reward") is not None and r["step"] > warmup]
    scores = finite_values(accuracy)
    if logs:
        reward = [r["reward"] for r in logs]
        axes[0].plot(
            [r["step"] for r in logs],
            reward,
            color=CORAL,
            marker="D",
            markersize=9,
            markerfacecolor="none",
            linestyle="--",
            label="Weighted train reward",
        )
        scores.extend(finite_values(reward))
    if summary["heldout_accuracy"] is not None:
        score = summary["heldout_accuracy"]
        axes[0].axhline(
            score,
            color=OLIVE,
            linestyle=(0, (2, 2)),
            linewidth=2.1,
            label=f"Holdout score (n={summary['heldout_samples']})",
            zorder=1,
        )
        scores.append(score)
    axes[0].set(title="Answer quality", xlabel="Optimizer step", ylabel="Answer score / reward")
    axes[0].set_ylim(min([0, *scores]) - 0.04, max([1, *scores]) + 0.04)
    calls = [s["tool_calls_mean"] for s in steps]
    axes[1].plot(
        [s["step"] for s in steps], calls, color=CORAL, marker="s", markeredgecolor=INK, label="Actual tool calls"
    )
    axes[1].set(title="Work performed", xlabel="Optimizer step", ylabel="Mean tool calls / trajectory")
    call_max = max([1, *finite_values(calls)])
    axes[1].set_ylim(-0.04 * call_max, call_max * 1.1)
    for ax in axes:
        style_axis(ax, steps=True)
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    axes[0].yaxis.set_major_locator(MultipleLocator(0.25) if max([1, *scores]) <= 1 else MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    save(
        fig,
        "quality",
        "Quality & actual tool use",
        counts + " Original workload rewards; W3 uses lenient answer matching.",
    )

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    device, allocator = read_csv(path / "memory.csv"), read_csv(path / "allocator_memory.csv")
    samples = device + allocator
    memory_origin = min((int(r["time_ns"]) for r in samples), default=0)
    memory_values = []
    for rows, key, label, color, marker, line in (
        (device, "used_bytes", "Device used · NVML", INK, "o", "-"),
        (allocator, "allocated_bytes", "PyTorch allocated", CORAL, "s", "--"),
        (allocator, "reserved_bytes", "PyTorch reserved", OLIVE, "D", "-."),
    ):
        if rows:
            values_gib = [int(r[key]) / 2**30 for r in rows]
            memory_values.extend(values_gib)
            ax.plot(
                [(int(r["time_ns"]) - memory_origin) / 1e9 for r in rows],
                values_gib,
                color=color,
                marker=marker,
                markersize=3.5 if rows is device else 5.5,
                markevery=max(1, len(rows) // 32),
                linewidth=1.6 if rows is device else 2.2,
                linestyle=line,
                label=label,
                markeredgewidth=0.7,
                markeredgecolor=color,
            )
    style_axis(ax)
    logarithmic_memory = value_axis(ax, memory_values)
    ax.set(
        xlabel="Time since first memory sample (s)",
        ylabel="GPU memory (GiB)" + (" · log scale" if logarithmic_memory else ""),
    )
    if memory_values and not logarithmic_memory:
        ax.set_ylim(0, max(memory_values) * 1.18)
    if samples:
        end = max((int(r["time_ns"]) - memory_origin) / 1e9 for r in samples)
        measured_start = [
            e["start_ns"] for e in events if e["name"] == "optimizer_step" and e["step"] == steps[0]["step"]
        ]
        evaluation_start = [e["start_ns"] for e in events if e["name"] == "evaluation"]
        if measured_start and warmup:
            boundary = (min(measured_start) - memory_origin) / 1e9
            if 0 < boundary < end:
                ax.axvspan(0, boundary, color=GOLD, alpha=0.16, linewidth=0, zorder=0)
                ax.text(
                    boundary / 2,
                    0.96,
                    "Warmup",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=10,
                )
        if evaluation_start:
            boundary = (min(evaluation_start) - memory_origin) / 1e9
            if 0 < boundary < end:
                ax.axvspan(boundary, end, color=CORAL, alpha=0.08, linewidth=0, zorder=0)
                ax.axvline(boundary, color=CORAL, linestyle=":", linewidth=1.3)
                ax.text(
                    (boundary + end) / 2,
                    0.96,
                    "Evaluation",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=10,
                )
        ax.set_xlim(0, end * 1.015 if end else 1)
    if not device:
        ax.text(
            0.02,
            0.82,
            "NVML unavailable / disabled\nNo device peak inferred",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
    save(
        fig,
        "memory",
        "GPU memory over time",
        "NVML is sampled device-wide usage; PyTorch curves are allocator step-boundary samples. All samples retained.",
    )

    selected_step = steps[0]["step"]
    selected = [e for e in events if e["step"] == selected_step and e.get("phase", "train") == "train"]
    origin = min(e["start_ns"] for e in selected)
    lanes = [
        "environment_reset",
        "sync_weights",
        "engine_wake",
        "cache_reset",
        "engine_generate",
        "cpu_tool",
        "engine_sleep",
        "policy_logprob",
        "reward",
        "loss_forward",
        "backward",
        "optimizer_update",
    ]
    lane_labels = [
        "Environment reset",
        "Weight sync",
        "Wake",
        "Cache reset",
        "Engine generation",
        "CPU tools",
        "Sleep",
        "Policy logprob",
        "Rewards",
        "Forward / loss",
        "Backward",
        "Optimizer update",
    ]
    fig, ax = plt.subplots(figsize=(12.8, 7.6))
    style_axis(ax, grid="x")
    for index, lane in enumerate(lanes):
        intervals = [((e["start_ns"] - origin) / 1e9, e["duration_ms"] / 1000) for e in selected if e["name"] == lane]
        color, _ = FAMILIES[phase_family(PHASES.get(lane, "Other host / bookkeeping"))]
        if index % 2 == 0:
            ax.axhspan(index - 0.48, index + 0.48, color=INK, alpha=0.025, linewidth=0, zorder=0)
        ax.broken_barh(intervals, (index - 0.29, 0.58), facecolors=color, edgecolors=INK, linewidth=0.55, zorder=3)
        if not intervals:
            ax.text(
                0.99,
                index,
                "no events",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                alpha=0.6,
            )
    ax.set_yticks(range(len(lanes)), lane_labels)
    ax.set_ylim(len(lanes) - 0.45, -0.6)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=3))
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlabel(f"Elapsed time within optimizer step {selected_step} (s)")
    save(
        fig,
        "timeline",
        f"Inside optimizer step {selected_step}",
        "Host timeline on a linear time axis. Nested spans can overlap; short events are drawn at their actual duration.",
        handles=family_handles(),
    )
    write_report(path, manifest, details)
    print(
        f"Five figures (PNG + SVG); breakdown axis: {'log' if logarithmic else 'linear'}. Demo report: {path / 'report.html'}"
    )
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
<style>body{{font:16px system-ui;max-width:1160px;margin:40px auto;padding:0 24px;color:{INK}}}
table{{border-collapse:collapse}}td,th{{text-align:left;padding:8px 16px;border-bottom:1px solid {GOLD}}}
a{{color:{OLIVE}}}img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:rgba(230,199,103,.12);padding:12px;max-height:420px;overflow:auto}}
details{{margin:16px 0}}.note{{background:rgba(230,199,103,.22);padding:16px}}figure{{margin:24px 0}}</style>
<h1>HPML profiling: {escape(label)}</h1>
<p>{escape(summary["model"])} · {escape(summary["cell"])} · {escape(manifest["gpu"])}</p>
<p>Measured steps: {summary["measured_steps"]}; warmup excluded: {manifest["config"]["warmup_steps"]}.
Commit: {escape(manifest.get("git_commit", "fixture"))}</p>
<p class="note">Use unprofiled benchmark runs for speedup claims. Breakdown points show exclusive host wall time; axes explicitly identify logarithmic scales.
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
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    order = [variant for variant in ORDER if any(r["variant"] == variant for r in rows)]
    metrics = [
        ("rollout_median_ms", "Rollout + cleanup", "Wall time (s)", 1000),
        ("step_median_ms", "Optimizer step", "Wall time (s)", 1000),
        ("generated_tokens_per_s", "Generation throughput", "Generated tokens / s", 1),
        ("heldout_accuracy", "Held-out answer score", "Answer score", 1),
        ("syncs_per_rollout", "Weight synchronization", "Syncs / rollout", 1),
        ("device_peak_used_bytes", "Sampled device peak", "GPU memory (GiB)", 2**30),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2))
    for ax, (metric, title, label, divisor) in zip(axes.flat, metrics, strict=True):
        collected = []
        for index, variant in enumerate(order):
            matching = sorted((r for r in rows if r["variant"] == variant), key=lambda row: row["seed"])
            values = [r[metric] / divisor for r in matching if r[metric] is not None]
            collected.extend(values)
            if values:
                offsets = (
                    [0] if len(values) == 1 else [-0.12 + 0.24 * i / (len(values) - 1) for i in range(len(values))]
                )
                ax.scatter(
                    [index + offset for offset in offsets],
                    values,
                    s=65,
                    marker=VARIANT_MARKERS[variant],
                    color=VARIANT_COLORS[variant],
                    edgecolors=INK,
                    linewidths=0.9,
                    zorder=4,
                )
                ax.plot(
                    [index - 0.23, index + 0.23],
                    [statistics.mean(values)] * 2,
                    color=INK,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=3,
                )
        labels = [f"{VARIANT_LABELS[v]}\nn={sum(r['variant'] == v for r in rows)}" for v in order]
        ax.set_xticks(range(len(order)), labels, rotation=12, ha="right" if len(order) > 1 else "center")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(title, loc="left")
        style_axis(ax, grid="y")
        logarithmic = value_axis(ax, collected, allow_log=metric != "heldout_accuracy")
        ax.set_ylabel(label + (" · log scale" if logarithmic else ""))
        if metric == "heldout_accuracy":
            ax.set_ylim(-0.04, 1.04)
        elif collected and not logarithmic:
            extent = max(collected) or 1
            ax.set_ylim(-0.04 * extent, 1.14 * extent)
        if not collected:
            ax.text(
                0.5, 0.5, "Not collected", transform=ax.transAxes, ha="center", va="center", fontsize=11, alpha=0.6
            )
    cell = Path(rows[0]["cell"])
    context = f"{cell.parent.name.upper()} · {cell.name}  |  {Path(rows[0]['model']).name}  |  Matched benchmark runs"
    finish_figure(
        plt,
        fig,
        output / "comparison",
        "Performance across implementations",
        context,
        "Points = independent runs; horizontal marks = means. No confidence intervals. All panels exclude warmup.",
    )
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
    print(f"Comparison: {output / 'comparison.png'} (+ SVG); paired speedups use matching seeds only")


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
