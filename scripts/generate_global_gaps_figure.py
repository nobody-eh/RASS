#!/usr/bin/env python3
"""Generate a readable global gap-vs-threshold figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


SOURCE_LABELS = {
    "fi": "Instant-NGP (full)",
    "oc": "Instant-NGP (object)",
    "zipnerf": "ZipNeRF",
}
SOURCE_ORDER = {"fi": 0, "oc": 1, "zipnerf": 2}
METRIC_ORDER = {"PSNR": 0, "SSIM": 1, "LPIPS": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate global gap threshold figure.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("sweep_cluster_k/share_bundle_prism_20260304/fig2_global_gaps_vs_threshold.csv"),
        help="CSV with global gap rows.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/fig_global_gaps.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/fig_global_gaps.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def _metric_label(metric: str) -> str:
    metric = metric.upper()
    if metric == "PSNR":
        return "PSNR gap (dB)"
    if metric == "SSIM":
        return "SSIM gap"
    if metric == "LPIPS":
        return "LPIPS gap"
    return f"{metric} gap"


def _fmt_signed(metric: str, value: float) -> str:
    sign = "+" if value >= 0 else "-"
    value = abs(value)
    if metric == "PSNR":
        return f"{sign}{value:.3f} dB"
    return f"{sign}{value:.4f}"


def _fmt_bar_label(metric: str, signed_gap: float, threshold: float) -> str:
    pct = abs(signed_gap) / threshold * 100.0 if threshold > 0 else 0.0
    return f"{_fmt_signed(metric, signed_gap)} ({pct:.0f}%)"


def _fmt_threshold(metric: str, value: float) -> str:
    if metric == "PSNR":
        return f"{value:.2f} dB"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv).copy()
    required = {"csv", "metric", "diff_subset_minus_full", "abs_diff", "threshold", "pass"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.input_csv}: {sorted(missing)}")

    df["csv"] = df["csv"].astype(str).str.strip().str.lower()
    df["metric"] = df["metric"].astype(str).str.strip().str.upper()
    for col in ["diff_subset_minus_full", "abs_diff", "threshold"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pass"] = df["pass"].astype(str).str.lower().isin(["true", "1", "yes"])
    df = df.dropna(subset=["abs_diff", "threshold", "diff_subset_minus_full"]).copy()
    df["_metric_order"] = df["metric"].map(METRIC_ORDER).fillna(99)
    df["_source_order"] = df["csv"].map(SOURCE_ORDER).fillna(99)
    df = df.sort_values(["_metric_order", "_source_order", "csv"], kind="mergesort")

    metrics = df["metric"].drop_duplicates().tolist()
    if not metrics:
        raise ValueError(f"No plottable rows in {args.input_csv}")

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )

    width = 9.4 if len(metrics) <= 2 else 12.0
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, 4.55), constrained_layout=False, sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    method_colors = {
        "fi": "#2563eb",
        "oc": "#0f766e",
        "zipnerf": "#b45309",
    }
    pass_bg = "#ecfdf5"
    fail_bg = "#fef2f2"
    grid = "#e2e8f0"
    text = "#0f172a"
    muted = "#64748b"

    all_pass = bool(df["pass"].all())

    for panel_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        sub = df[df["metric"] == metric].copy()
        labels = [SOURCE_LABELS.get(src, src.upper()) for src in sub["csv"].tolist()]
        y_positions = list(range(len(sub)))
        threshold = float(sub["threshold"].max())
        x_max = max(float(sub["abs_diff"].max()) * 1.28, threshold * 1.12, 1e-9)

        ax.set_facecolor("white")
        ax.axvspan(0, threshold, color=pass_bg, zorder=0)
        ax.axvspan(threshold, x_max, color=fail_bg, zorder=0)
        ax.axvline(threshold, color="#059669", linestyle=(0, (4, 4)), linewidth=1.4, zorder=2)
        ax.text(
            threshold,
            1.035,
            f"threshold {_fmt_threshold(metric, threshold)}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="bottom",
            color="#047857",
            fontsize=8.7,
            fontweight="semibold",
        )

        colors = [method_colors.get(src, "#64748b") for src in sub["csv"].tolist()]
        bars = ax.barh(
            y_positions,
            sub["abs_diff"].tolist(),
            color=colors,
            height=0.54,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.set_yticks(y_positions)
        if panel_idx == 0:
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", color=grid, linewidth=0.9, zorder=1)
        ax.tick_params(axis="x", length=0, colors=muted)
        ax.tick_params(axis="y", length=0, colors=text)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel(_metric_label(metric))
        ax.set_title(metric, fontweight="semibold", color=text, pad=18)

        label_pad = x_max * 0.022
        for bar, (_, row) in zip(bars, sub.iterrows()):
            signed = _fmt_bar_label(metric, float(row["diff_subset_minus_full"]), threshold)
            x = float(row["abs_diff"]) + label_pad
            ha = "left"
            if x > x_max * 0.96:
                x = float(row["abs_diff"]) - label_pad
                ha = "right"
            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2.0,
                signed,
                ha=ha,
                va="center",
                fontsize=9.4,
                color=text,
                fontweight="semibold",
                zorder=4,
            )
    title = "RASS-48 Global Gaps vs Paper Thresholds"
    fig.suptitle(title, fontsize=15.5, fontweight="semibold", color=text, y=0.97)
    status = "All shown method-metric pairs pass." if all_pass else "Some method-metric pairs exceed the threshold."
    fig.text(
        0.5,
        0.91,
        f"Bars show absolute subset-vs-full gaps; labels show signed direction and percent of threshold. "
        f"Green span is accepted. {status}",
        ha="center",
        va="top",
        color="#475569",
        fontsize=9.4,
    )

    fig.subplots_adjust(left=0.18, right=0.985, top=0.76, bottom=0.15, wspace=0.18)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(args.output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output_pdf}")
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
