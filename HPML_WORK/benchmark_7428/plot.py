"""Figures for the issue #7428 study, from results/issue7428/tables/*.csv and the raw run folders.

Run from the repository root (after aggregate.py): python HPML_WORK/benchmark_7428/plot.py [--results-dir results/issue7428]
Writes PNG (200 dpi) + PDF into results/issue7428/figures/.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("results/issue7428")  # overridden by --results-dir
TABLES = ROOT / "tables"
FIG = ROOT / "figures"
WARMUP = 1

# Reference palette, light mode, categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABEL_COLOR = {"baseline": SERIES[0], "fixed": SERIES[1]}
LABEL_NAME = {"baseline": "baseline (main)", "fixed": "fixed (this branch)"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2,
    }
)


def read(name):
    with open(TABLES / name) as f:
        return list(csv.DictReader(f))


def num(x):
    return None if x in ("", "None", None) else float(x)


def save(fig, name, note=None):
    if note:
        # Place the footnote below everything already drawn (tick labels, x labels), never on top of them
        fig.canvas.draw()
        bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
        fig.text(0.01, bottom - 0.02, note, fontsize=7, color=MUTED, ha="left", va="top")
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figures/{name}.png")


def grouped_bars(ax, groups, values, fmt="{:.2f}", unit="", width=0.36, annotate_speedup=True):
    """values: {label: [v per group]}; labels baseline/fixed. Direct value labels on the bar ends."""
    xs = range(len(groups))
    for i, label in enumerate(["baseline", "fixed"]):
        offs = [x + (i - 0.5) * (width + 0.02) for x in xs]
        bars = ax.bar(
            offs,
            values[label],
            width,
            color=LABEL_COLOR[label],
            label=LABEL_NAME[label],
            edgecolor=SURFACE,
            linewidth=1,
        )
        for b, v in zip(bars, values[label], strict=False):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                fmt.format(v) + unit,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK2,
            )
    if annotate_speedup:
        top = max(max(v) for v in values.values())
        for x, b, f in zip(xs, values["baseline"], values["fixed"], strict=False):
            ax.text(
                x,
                max(b, f) + top * 0.09,
                f"{b / f:.2f}×",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK,
                fontweight="bold",
            )
        ax.set_ylim(0, top * 1.25)
    ax.set_xticks(list(xs), groups)
    ax.tick_params(axis="x", length=0)


def runs_by(workload=None, prefix=None, model_tag="1.7B"):
    out = {}
    for r in read("runs.csv"):
        if workload and r["workload"] != workload:
            continue
        if prefix and r["prefix_caching"] != prefix:
            continue
        if model_tag not in r["config"] or "fullft" in r["config"]:
            continue
        out[(r["cell"], r["label"])] = r
    return out


def k_of(cell):
    return int(cell.split("_")[0][1:])


# ---------------------------------------------------------------- 1. live benchmark
def fig_live():
    rows = read("live_summary.csv")
    turns = sorted({int(r["turns"]) for r in rows})
    val = {
        lab: [
            num(next(r for r in rows if r["label"] == lab and int(r["turns"]) == t)["median_rollout_ms"]) / 1000
            for t in turns
        ]
        for lab in ("baseline", "fixed")
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    grouped_bars(ax, [f"{t} turns" for t in turns], val, unit=" s")
    ax.set_ylabel("median rollout time (s)")
    ax.set_title("Rollout time vs. number of turns")
    ax.legend(loc="upper left")

    ax = axes[1]
    r8 = {lab: next(r for r in rows if r["label"] == lab and r["turns"] == "8") for lab in ("baseline", "fixed")}
    idx = list(range(1, 9))
    for lab in ("baseline", "fixed"):
        ys = [num(r8[lab][f"median_turn{t - 1}_ms"]) for t in idx]
        ax.plot(
            idx,
            ys,
            color=LABEL_COLOR[lab],
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=LABEL_NAME[lab],
        )
        ax.text(8.15, ys[-1], f"{ys[-1]:.0f} ms", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(idx)
    ax.set_xlim(0.7, 8.9)
    ax.set_ylim(0, None)
    ax.set_xlabel("turn index (8-turn rollout)")
    ax.set_ylabel("median time per turn (ms)")
    ax.set_title("Per-turn time: only the first turn pays for the weight sync")
    ax.legend(loc="center right")
    fig.tight_layout()
    save(
        fig,
        "fig1_live_benchmark",
        "Live benchmark (issue7429_live_benchmark.py): Qwen3-0.6B + LoRA, colocated vLLM 0.32.0 with sleep mode, "
        "1 RTX 4090, 10 iterations each, rollout only.",
    )


# ---------------------------------------------------------------- 2. W1 step time
def fig_w1_step_time():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    tops = []
    for ax, prefix in zip(axes, ("off", "on"), strict=False):
        runs = runs_by("w1", prefix)
        cells = sorted({c for c, _ in runs}, key=k_of)
        val = {lab: [num(runs[(c, lab)]["median_step_ms"]) / 1000 for c in cells] for lab in ("baseline", "fixed")}
        grouped_bars(ax, [f"K={k_of(c)}" for c in cells], val, fmt="{:.1f}", unit=" s")
        tops.append(max(max(v) for v in val.values()))
        ax.set_title(f"Prefix caching {prefix}" + (" (weight reload only)" if prefix == "off" else ""))
    axes[0].set_ylim(0, max(tops) * 1.25)  # shared y: headroom for the tallest panel's speedup labels
    axes[0].set_ylabel("median training-step time (s)")
    axes[0].legend(loc="upper left")
    fig.supxlabel("tool calls per sample (W1 chain arithmetic, L_obs=256, P=256)", fontsize=9, color=INK2)
    fig.tight_layout()
    save(
        fig,
        "fig2_w1_step_time",
        "Qwen3-1.7B + LoRA r16, GRPO with tools, 16 rollouts/step, 10 steps (step 0 excluded), 1 RTX 4090. "
        "Speedup = baseline / fixed.",
    )


# ---------------------------------------------------------------- 3. W1 step-time breakdown (prefix off)
def fig_w1_breakdown(prefix="off"):
    steps = [
        r
        for r in read("steps.csv")
        if r["workload"] == "w1"
        and r["prefix_caching"] == prefix
        and "1.7B" in r["config"]
        and int(r["step"]) >= WARMUP
    ]
    comp = [
        ("training (forward/backward)", lambda r: num(r["train_compute_ms"])),
        ("tools, scoring, other", lambda r: num(r["tools_scoring_other_ms"])),
        ("vLLM generate", lambda r: num(r["llm_generate_ms"])),
        ("sleep / wake-up", lambda r: num(r["llm_sleep_ms"]) + num(r["wake_weights_ms"]) + num(r["wake_kv_cache_ms"])),
        ("weight load", lambda r: num(r["sync_ms"]) - num(r["wake_weights_ms"])),
    ]
    groups = defaultdict(list)
    for r in steps:
        groups[(k_of(r["cell"]), r["label"])].append(r)
    keys = sorted(groups, key=lambda k: (k[0], k[1] != "baseline"))
    med = {k: [statistics.median(f(r) for r in groups[k]) / 1000 for _, f in comp] for k in keys}
    syncs = {k: statistics.median(num(r["num_syncs"]) for r in groups[k]) for k in keys}

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    xs, labels = [], []
    x = 0
    for i, k in enumerate(keys):
        if i and k[0] != keys[i - 1][0]:
            x += 0.6
        xs.append(x)
        labels.append(f"{'base' if k[1] == 'baseline' else 'fixed'}\nK={k[0]}")
        x += 1
    order = list(range(len(comp)))
    color_for = {j: SERIES[len(comp) - 1 - j] for j in order}  # weight sync gets slot 1 (the job of the chart)
    bottoms = [0.0] * len(keys)
    for j, (name, _) in enumerate(comp):
        vals = [med[k][j] for k in keys]
        ax.bar(xs, vals, 0.72, bottom=bottoms, color=color_for[j], label=name, edgecolor=SURFACE, linewidth=1.5)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=False)]
    for xx, k, top in zip(xs, keys, bottoms, strict=False):
        ax.text(
            xx,
            top,
            f"{top:.1f} s\n{int(syncs[k])} sync{'s' if syncs[k] > 1 else ''}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK2,
        )
    ax.set_xticks(xs, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(bottoms) * 1.18)
    ax.set_ylabel("median time per step (s)")
    ax.set_title(f"Where the step time goes (W1, prefix caching {prefix})")
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], names[::-1], loc="upper left", fontsize=8)
    save(
        fig,
        f"fig3_w1_breakdown_prefix-{prefix}",
        "Stacked medians over steps 1-9. Qwen3-1.7B + LoRA r16, 16 rollouts/step, 1 RTX 4090. "
        "Baseline syncs once per generation turn (K tool turns + final answer).",
    )


# ---------------------------------------------------------------- 4. prefix-cache reuse across turns
def fig_prefix_reuse():
    turns = [
        r
        for r in read("turns.csv")
        if r["workload"] == "w1"
        and r["prefix_caching"] == "on"
        and "1.7B" in r["config"]
        and int(r["step"]) >= WARMUP
        and int(r["turn"]) >= 1
    ]
    agg = defaultdict(lambda: [0, 0])
    for r in turns:
        a = agg[(k_of(r["cell"]), r["label"])]
        a[0] += num(r["cached_tokens"])
        a[1] += num(r["prompt_tokens"])
    ks = sorted({k for k, _ in agg})
    val = {lab: [100 * agg[(k, lab)][0] / agg[(k, lab)][1] for k in ks] for lab in ("baseline", "fixed")}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    grouped_bars(ax, [f"K={k}" for k in ks], val, fmt="{:.0f}", unit="%", annotate_speedup=False)
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("prompt tokens served from prefix cache (%)")
    ax.set_title("Prefix-cache hits on turns ≥ 2")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    save(
        fig,
        "fig4_prefix_cache_hits",
        "vLLM num_cached_tokens / prompt tokens, turns after the first, W1, prefix caching on.\n"
        "Baseline hits come only from rollouts of the same prompt within one call (the KV is dropped every turn).",
    )


# ---------------------------------------------------------------- 5. GPU memory timeline
def fig_memory(config="w1_K4_L256_P256_1.7B_prefix-on", first=2, last=3):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True, sharey=True)
    for ax, lab in zip(axes, ("baseline", "fixed"), strict=False):
        run = ROOT / "train" / lab / config
        events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
        steps = {e["step"]: e for e in events if e["event"] == "train_step"}
        t0 = steps[first]["t_start"]
        t1 = steps[last]["t_start"] + steps[last]["ms"] / 1000
        mem = [
            (float(r["t_s"]) - t0, float(r["used_mib"]) / 1024)
            for r in csv.DictReader(open(run / "gpu_mem.csv"))
            if t0 <= float(r["t_s"]) <= t1
        ]
        for e in events:
            if e["event"] == "rollout" and first <= e["step"] <= last:
                ax.axvspan(e["t_start"] - t0, e["t_start"] - t0 + e["ms"] / 1000, color=GRID, alpha=0.6, lw=0)
        ax.plot([m[0] for m in mem], [m[1] for m in mem], color=LABEL_COLOR[lab], lw=1.5)
        syncs = [e["t_start"] - t0 for e in events if e["event"] == "sync_weights" and first <= e["step"] <= last]
        ax.plot(
            syncs, [0.8] * len(syncs), linestyle="none", marker="v", markersize=7, color=INK2, markeredgecolor=SURFACE
        )
        ax.set_title(f"{LABEL_NAME[lab]}: {len(syncs)} weight syncs in {last - first + 1} steps", fontsize=9)
        ax.set_ylabel("GPU memory used (GiB)")
        ax.set_ylim(0, 24)
    axes[1].set_xlabel("time (s)  —  shaded: rollout (generation + tool calls); ▼ weight sync")
    fig.suptitle(
        "GPU memory over two training steps (W1 K=4, prefix caching on)",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(
        fig,
        "fig5_gpu_memory_timeline",
        "NVML whole-device memory, sampled every 50 ms. Baseline drops and reloads the vLLM weights and KV cache "
        "on every turn; fixed keeps them for the whole rollout and releases them before training.",
    )


# ---------------------------------------------------------------- 6. correctness with real weight updates
def fig_correctness(config="w1_K4_L256_P256_0.6B-fullft_followed1_prefix-on"):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    metrics = [
        ("reward", "reward (answer + chain followed)"),
        ("tools/call_frequency", "tool calls per sample"),
        ("sampling/sampling_logp_difference/mean", "|vLLM − trainer| log-prob, mean"),
    ]
    for ax, (key, title) in zip(axes, metrics, strict=False):
        for lab in ("baseline", "fixed"):
            hist = [
                h
                for h in json.loads((ROOT / "train" / lab / config / "log_history.json").read_text())
                if "reward" in h
            ]
            ax.plot(
                [h["step"] for h in hist],
                [h[key] for h in hist],
                color=LABEL_COLOR[lab],
                label=LABEL_NAME[lab],
                marker="o",
                markersize=3.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("training step")
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.set_ylim(0, None)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Training is unchanged when weights actually update",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(
        fig,
        "fig6_correctness",
        "Qwen3-0.6B full fine-tuning, lr 3e-6, reward = answer + share of chain followed, W1 K=4, 15 steps. "
        "A stale-weight bug would show as a jump in the log-prob gap after each update.",
    )


# ---------------------------------------------------------------- 7. W2
def fig_w2():
    runs = runs_by("w2", "on")
    cells = sorted({c for c, _ in runs})
    val = {lab: [num(runs[(c, lab)]["median_step_ms"]) / 1000 for c in cells] for lab in ("baseline", "fixed")}
    names = {"K3_top1": "K=3, 1 doc/turn\n(~266 tok)", "K3_top3": "K=3, 3 docs/turn\n(~773 tok)"}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    grouped_bars(ax, [names.get(c, c) for c in cells], val, fmt="{:.1f}", unit=" s")
    ax.set_ylabel("median training-step time (s)")
    ax.set_title("W2 multi-hop retrieval")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig7_w2_step_time", "Qwen3-1.7B + LoRA r16, prefix caching on, 10 steps (step 0 excluded).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT)
    ROOT = parser.parse_args().results_dir
    TABLES, FIG = ROOT / "tables", ROOT / "figures"
    fig_live()
    fig_w1_step_time()
    fig_w1_breakdown("off")
    fig_w1_breakdown("on")
    fig_prefix_reuse()
    fig_memory()
    fig_correctness()
    fig_w2()
