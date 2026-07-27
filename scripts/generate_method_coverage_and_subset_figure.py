#!/usr/bin/env python3
"""Generate a readable method-coverage and subset-overlap figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


METHOD_LABELS = {
    "instant-ngp-fi": "Instant-NGP (full)",
    "feature-splatting": "Feature Splatting",
    "zipnerf": "ZipNeRF",
}
METHOD_ORDER = {
    "instant-ngp-fi": 0,
    "feature-splatting": 1,
    "zipnerf": 2,
}
METRIC_ORDER = {
    "psnr": 0,
    "ssim": 1,
    "lpips": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate method coverage and subset figure.")
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path("results/cross_method_ranking_fidelity/coverage_summary.csv"),
        help="Coverage summary CSV from analyze_cross_method_ranking_fidelity.py.",
    )
    parser.add_argument(
        "--metric-coverage-csv",
        type=Path,
        default=Path("results/cross_method_ranking_fidelity/metric_coverage_summary.csv"),
        help="Metric-level common intersection coverage CSV.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/fig_method_coverage_and_subset.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/fig_method_coverage_and_subset.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def _fmt_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def _metric_label(metric: str) -> str:
    return str(metric).upper()


def _rounded_panel(ax: plt.Axes, face: str = "#ffffff", edge: str = "#e2e8f0") -> None:
    panel = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(panel)


def main() -> None:
    args = parse_args()
    coverage_df = pd.read_csv(args.coverage_csv).copy()
    metric_df = pd.read_csv(args.metric_coverage_csv).copy()

    coverage_required = {
        "ranking_metric",
        "method",
        "full_scene_count",
        "nominal_subset_overlap",
        "shared_full_overlap",
        "shared_subset_overlap",
    }
    metric_required = {"ranking_metric", "shared_full_size", "shared_subset_size", "num_methods"}
    missing_coverage = coverage_required.difference(coverage_df.columns)
    missing_metric = metric_required.difference(metric_df.columns)
    if missing_coverage:
        raise ValueError(f"Missing columns in {args.coverage_csv}: {sorted(missing_coverage)}")
    if missing_metric:
        raise ValueError(f"Missing columns in {args.metric_coverage_csv}: {sorted(missing_metric)}")

    coverage_df["ranking_metric"] = coverage_df["ranking_metric"].astype(str).str.lower()
    coverage_df["method"] = coverage_df["method"].astype(str)
    metric_df["ranking_metric"] = metric_df["ranking_metric"].astype(str).str.lower()

    for col in [
        "full_scene_count",
        "nominal_subset_overlap",
        "shared_full_overlap",
        "shared_subset_overlap",
    ]:
        coverage_df[col] = pd.to_numeric(coverage_df[col], errors="coerce")
    for col in ["shared_full_size", "shared_subset_size", "num_methods"]:
        metric_df[col] = pd.to_numeric(metric_df[col], errors="coerce")

    method_df = (
        coverage_df.groupby("method", as_index=False)
        .agg(
            full_scene_count=("full_scene_count", "max"),
            shared_full_overlap=("shared_full_overlap", "min"),
            nominal_subset_overlap=("nominal_subset_overlap", "min"),
            shared_subset_overlap=("shared_subset_overlap", "min"),
        )
        .dropna()
    )
    method_df["_order"] = method_df["method"].map(METHOD_ORDER).fillna(99)
    method_df = method_df.sort_values(["_order", "method"], kind="mergesort").reset_index(drop=True)

    metric_df["_order"] = metric_df["ranking_metric"].map(METRIC_ORDER).fillna(99)
    metric_df = metric_df.sort_values(["_order", "ranking_metric"], kind="mergesort").reset_index(drop=True)

    common_full = int(metric_df["shared_full_size"].min()) if not metric_df.empty else 0
    shared_subset = int(metric_df["shared_subset_size"].min()) if not metric_df.empty else 0
    nominal_subset = int(coverage_df["nominal_subset_overlap"].max()) if not coverage_df.empty else shared_subset
    num_methods = int(metric_df["num_methods"].max()) if not metric_df.empty else len(method_df)

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

    fig = plt.figure(figsize=(10.4, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.24)
    ax_methods = fig.add_subplot(gs[0, 0])
    ax_subset = fig.add_subplot(gs[0, 1])

    text = "#0f172a"
    muted = "#64748b"
    grid = "#e2e8f0"
    total_color = "#dbeafe"
    common_color = "#2563eb"
    subset_track = "#e2e8f0"
    subset_color = "#0f766e"

    for ax in (ax_methods, ax_subset):
        _rounded_panel(ax)
        ax.tick_params(axis="both", length=0, colors=muted)

    y = np.arange(len(method_df))
    max_total = max(float(method_df["full_scene_count"].max()) * 1.16, float(common_full) * 1.18)
    ax_methods.barh(
        y,
        method_df["full_scene_count"],
        color=total_color,
        height=0.62,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
        label="available method logs",
    )
    ax_methods.barh(
        y,
        method_df["shared_full_overlap"],
        color=common_color,
        height=0.36,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="common cross-method pool",
    )

    ax_methods.set_yticks(y, [METHOD_LABELS.get(m, m) for m in method_df["method"]])
    ax_methods.invert_yaxis()
    ax_methods.set_xlim(0, max_total)
    ax_methods.grid(axis="x", color=grid, linewidth=0.9, zorder=0)
    ax_methods.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _fmt_int(value)))
    ax_methods.set_xlabel("Scenes with valid metric logs")
    ax_methods.set_title("Method Log Coverage", fontweight="semibold", color=text, pad=14)

    for yy, row in zip(y, method_df.to_dict("records")):
        full = int(row["full_scene_count"])
        shared = int(row["shared_full_overlap"])
        ax_methods.text(
            shared - max_total * 0.018,
            yy,
            f"{_fmt_int(shared)} shared",
            ha="right",
            va="center",
            fontsize=9.0,
            color="white",
            fontweight="semibold",
        )
        ax_methods.text(
            full + max_total * 0.012,
            yy,
            f"total {_fmt_int(full)}",
            ha="left",
            va="center",
            fontsize=8.8,
            color=text,
            fontweight="semibold",
        )

    ax_methods.text(
        0.01,
        0.98,
        "blue = common pool, pale = method logs",
        transform=ax_methods.transAxes,
        ha="left",
        va="top",
        color=muted,
        fontsize=8.6,
    )

    yy = np.arange(len(method_df))
    max_subset = max(nominal_subset, int(method_df["nominal_subset_overlap"].max())) if not method_df.empty else 48
    ax_subset.barh(
        yy,
        [max_subset] * len(yy),
        color=subset_track,
        height=0.54,
        edgecolor="white",
        linewidth=1.0,
        zorder=1,
    )
    ax_subset.barh(
        yy,
        method_df["shared_subset_overlap"],
        color=subset_color,
        height=0.54,
        edgecolor="white",
        linewidth=1.0,
        zorder=2,
    )

    ax_subset.set_yticks(yy)
    ax_subset.tick_params(axis="y", labelleft=False)
    ax_subset.invert_yaxis()
    ax_subset.set_xlim(0, max_subset * 1.18)
    ax_subset.grid(axis="x", color=grid, linewidth=0.9, zorder=0)
    ax_subset.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _fmt_int(value)))
    ax_subset.set_xlabel("Selected scenes available to all methods")
    ax_subset.set_title("RASS-48 Subset Coverage by Method", fontweight="semibold", color=text, pad=14)

    for idx, row in method_df.iterrows():
        value = int(row["shared_subset_overlap"])
        ax_subset.text(
            value + max_subset * 0.025,
            idx,
            f"{value}/{max_subset}",
            ha="left",
            va="center",
            fontsize=10.0,
            color=text,
            fontweight="semibold",
        )

    metric_note = ", ".join(_metric_label(m) for m in metric_df["ranking_metric"].tolist())
    ax_subset.text(
        0.02,
        0.98,
        f"coverage checked for {metric_note}",
        transform=ax_subset.transAxes,
        ha="left",
        va="top",
        color=muted,
        fontsize=8.6,
    )

    fig.suptitle("Cross-Method Coverage and RASS-48 Subset Overlap", fontsize=15.0, fontweight="semibold", color=text, y=0.975)
    fig.text(
        0.5,
        0.915,
        f"Common full pool: {_fmt_int(common_full)} scenes across {num_methods} methods; selected subset overlap: {shared_subset}/{nominal_subset} scenes for each method.",
        ha="center",
        va="top",
        color="#475569",
        fontsize=9.4,
    )
    fig.subplots_adjust(left=0.125, right=0.985, top=0.78, bottom=0.16)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(args.output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output_pdf}")
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
