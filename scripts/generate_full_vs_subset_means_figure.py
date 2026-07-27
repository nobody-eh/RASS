#!/usr/bin/env python3
"""Generate a modern full-vs-subset metric means figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full-vs-subset means figure.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/strong_accept/distributional_fidelity_summary.csv"),
        help="CSV with metric, full_mean, subset_mean, abs_mean_gap columns.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/fig_full_vs_subset_means.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/fig_full_vs_subset_means.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def _metric_order(metric: str) -> int:
    return {"PSNR": 0, "SSIM": 1, "LPIPS": 2}.get(str(metric).upper(), 99)


def _fmt_value(metric: str, value: float) -> str:
    if metric == "PSNR":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _fmt_gap(metric: str, value: float) -> str:
    if metric == "PSNR":
        return f"{value:.3f}"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def _axis_limits(metric: str, values: list[float], abs_gap: float) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if metric == "PSNR":
        pad = max(abs_gap * 2.0, 0.22)
    elif metric == "SSIM":
        pad = max(abs_gap * 2.0, 0.012)
    else:
        pad = max(abs_gap * 20.0, 0.012)
    return max(0.0, lo - pad), hi + pad


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv).copy()
    required = {"metric", "full_mean", "subset_mean", "abs_mean_gap"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.input_csv}: {sorted(missing)}")

    df["metric"] = df["metric"].astype(str).str.upper()
    df["_order"] = df["metric"].map(_metric_order)
    df = df.sort_values(["_order", "metric"], kind="mergesort").reset_index(drop=True)

    full_n = int(pd.to_numeric(df["full_count"], errors="coerce").dropna().iloc[0]) if "full_count" in df else None
    subset_n = int(pd.to_numeric(df["subset_count"], errors="coerce").dropna().iloc[0]) if "subset_count" in df else None

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": False,
        }
    )

    fig, axes = plt.subplots(1, len(df), figsize=(9.3, 4.4), constrained_layout=False)
    if len(df) == 1:
        axes = [axes]

    full_color = "#64748b"
    subset_color = "#0f766e"
    bg_color = "#fbfbfd"
    bar_colors = [full_color, subset_color]

    for ax, (_, row) in zip(axes, df.iterrows()):
        metric = str(row["metric"])
        full_mean = float(row["full_mean"])
        subset_mean = float(row["subset_mean"])
        abs_gap = float(row["abs_mean_gap"])
        values = [full_mean, subset_mean]

        ax.set_facecolor(bg_color)
        y_min, y_max = _axis_limits(metric, values, abs_gap)
        bars = ax.bar(
            [0, 1],
            values,
            color=bar_colors,
            width=0.58,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.65, 1.65)
        ax.set_xticks([0, 1], ["Full", "RASS-48"])
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.9, zorder=0)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", colors="#64748b")
        ax.spines["left"].set_color("#cbd5e1")

        title = metric
        if metric == "LPIPS":
            title = "LPIPS\n(lower is better)"
        ax.set_title(title, fontweight="semibold", pad=11)

        label_pad = (y_max - y_min) * 0.035
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + label_pad,
                _fmt_value(metric, value),
                ha="center",
                va="bottom",
                color="#0f172a",
                fontsize=9.5,
                fontweight="semibold",
            )

        gap_text = _fmt_gap(metric, abs_gap)
        gap_label = rf"$|\Delta|$ {gap_text}" if gap_text.startswith("<") else rf"$|\Delta|$ = {gap_text}"
        ax.text(
            0.5,
            0.95,
            gap_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="#334155",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#e2e8f0", "alpha": 0.95},
        )

        if ax is axes[0]:
            ax.set_ylabel("Mean value")

    title = "Full Benchmark vs RASS-48 Mean Metrics"
    fig.suptitle(title, fontsize=14.5, fontweight="semibold", y=0.975)
    if full_n is not None and subset_n is not None:
        fig.text(
            0.5,
            0.91,
            f"Exact mean values are printed above bars; each panel uses its own zoomed y-axis. Full n={full_n:,}, subset n={subset_n}.",
            ha="center",
            va="top",
            color="#475569",
            fontsize=9.2,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=full_color, label="Full benchmark"),
        plt.Rectangle((0, 0), 1, 1, color=subset_color, label="RASS-48 subset"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.72, bottom=0.14, wspace=0.35)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(args.output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output_pdf}")
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
