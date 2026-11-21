import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HPML_WORK = Path(__file__).resolve().parents[2]
REPO_ROOT = HPML_WORK.parent
DEFAULT_INPUT = Path("HPML_WORK/windows/results/windows_gpu_benchmark_results.json")


def _read_report(path):
    with path.open(encoding="utf-8-sig") as file:
        report = json.load(file)

    if "training" in report:
        training = report["training"]
        return report, [
            {
                "label": report.get("label") or Path(report.get("output_dir", path.stem)).name,
                "duration_ms": training["duration_ms"],
                "median_step_ms": training["median_step_ms"],
                "step_times_ms": training["step_times_ms"],
                "peak_allocated_gib": training["peak_allocated_gib"],
                "peak_reserved_gib": training["peak_reserved_gib"],
                "train_loss": training["train_loss"],
            }
        ]

    policies = []
    if "policies" in report:
        for policy, runs in report["policies"].items():
            summary = report["summary"][policy]
            policies.append(
                {
                    "label": policy.replace("_", " "),
                    "rollout_ms": summary["median_rollout_ms"],
                    "sync_ms": summary["median_weight_reload_ms"],
                    "generation_ms": summary["median_generation_ms_per_turn"],
                    "peak_gib": summary["max_peak_allocated_gib"],
                }
            )
    else:
        runs = report["iterations"]
        sync_samples = [sum(run["sync_ms"]) for run in runs]
        summary = report["summary"]
        policies.append(
            {
                "label": report.get("label") or path.stem.replace("_", " "),
                "rollout_ms": summary["median_rollout_ms"],
                "sync_ms": float(np.median(sync_samples)),
                "generation_ms": summary["median_turn_ms"],
                "peak_gib": summary["max_peak_allocated_gib"],
            }
        )
    return report, policies


def _plot_training_reports(reports, runs, output):
    labels = [run["label"] for run in runs]
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(runs)))
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    figure.suptitle(
        f"GRPO training benchmark\n{reports[0].get('gpu', 'GPU not recorded')}",
        fontsize=15,
        fontweight="bold",
    )

    metrics = [
        ("duration_ms", "Total training time", "minutes", 1 / 60000),
        ("median_step_ms", "Median training step time", "seconds", 1 / 1000),
        ("train_loss", "Training loss", "loss", 1),
    ]
    for axis, (key, title, unit, scale) in zip((axes[0, 0], axes[0, 1], axes[1, 1]), metrics, strict=True):
        values = [run[key] * scale if run[key] is not None else np.nan for run in runs]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", labelrotation=25)
        axis.bar_label(bars, fmt="%.3f")
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.15)

    x_positions = np.arange(len(runs))
    width = 0.8 / len(runs)
    for offset, (key, label) in enumerate(
        (("peak_allocated_gib", "Allocated"), ("peak_reserved_gib", "Reserved"))
    ):
        values = [run[key] for run in runs]
        axes[0, 2].bar(
            x_positions + (offset - 0.5) * width,
            values,
            width,
            label=label,
            color=plt.get_cmap("Set2")(offset),
        )
    axes[0, 2].set_title("Peak GPU memory")
    axes[0, 2].set_ylabel("GiB")
    axes[0, 2].set_xticks(x_positions, labels, rotation=25)
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(axis="y", alpha=0.25)

    for index, run in enumerate(runs):
        axes[1, 0].plot(
            np.arange(1, len(run["step_times_ms"]) + 1),
            np.asarray(run["step_times_ms"]) / 1000,
            marker="o",
            label=run["label"],
            color=colors[index],
        )
    axes[1, 0].set_title("Training time by step")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 2].set_visible(False)

    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Plot rollout or GRPO training benchmark JSON results.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT],
        help="One or more rollout reports or GRPO training reports of the same type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path (default: next to the first input file).",
    )
    args = parser.parse_args()

    args.input = [path if path.is_absolute() else REPO_ROOT / path for path in args.input]
    if args.output is not None and not args.output.is_absolute():
        args.output = REPO_ROOT / args.output
    loaded = [_read_report(path) for path in args.input]
    reports = [item[0] for item in loaded]
    policies = [policy for _, entries in loaded for policy in entries]
    if not policies:
        raise ValueError("No benchmark policies found in input JSON.")

    is_training = ["training" in report for report in reports]
    if any(is_training) and not all(is_training):
        raise ValueError("Use separate visualization commands for rollout reports and GRPO training reports.")

    output = args.output or args.input[0].with_name(f"{args.input[0].stem}_visualization.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    if all(is_training):
        _plot_training_reports(reports, policies, output)
        try:
            output_label = output.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            output_label = str(output)
        print(f"Saved benchmark visualization: {output_label}")
        return

    for policy in policies:
        if policy["label"] == "affected reload each turn":
            policy["label"] = "Affected\n(reload per turn)"
        elif policy["label"] == "fixed resident during rollout":
            policy["label"] = "Resident\n(during rollout)"

    labels = [policy["label"] for policy in policies]
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(policies)))

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.suptitle(
        f"Qwen3-0.6B weight residency benchmark\n{reports[0].get('gpu', 'GPU not recorded')}",
        fontsize=15,
        fontweight="bold",
    )

    charts = [
        ("rollout_ms", "Median rollout time", "milliseconds"),
        ("sync_ms", "Median weight reload time", "milliseconds"),
        ("peak_gib", "Peak allocated GPU memory", "GiB"),
    ]
    for axis, (key, title, unit) in zip((axes[0, 0], axes[0, 1], axes[1, 1]), charts, strict=True):
        values = [policy[key] for policy in policies]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", labelrotation=0)
        axis.bar_label(bars, fmt="%.1f")
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.15)

    turn_count = max(len(policy["generation_ms"]) for policy in policies)
    positions = np.arange(turn_count)
    width = 0.8 / len(policies)
    for index, policy in enumerate(policies):
        values = policy["generation_ms"] + [np.nan] * (turn_count - len(policy["generation_ms"]))
        axes[1, 0].bar(
            positions + (index - (len(policies) - 1) / 2) * width,
            values,
            width,
            label=policy["label"],
            color=colors[index],
        )
    axes[1, 0].set_title("Median generation time by turn")
    axes[1, 0].set_xlabel("Generation turn")
    axes[1, 0].set_ylabel("milliseconds")
    axes[1, 0].set_xticks(positions, [str(index + 1) for index in positions])
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    figure.text(
        0.5,
        0.015,
        "Windows results emulate the residency policy with Transformers; Linux results use real colocated vLLM.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    figure.savefig(output, dpi=160)
    try:
        output_label = output.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        output_label = str(output)
    print(f"Saved benchmark visualization: {output_label}")


if __name__ == "__main__":
    main()
