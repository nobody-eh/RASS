#!/usr/bin/env python3
"""Generate the fixed-k pass-rate/LCB figure as a vector PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures/fig_budget_pass_lcb.pdf from hardcoded sweep data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fig_budget_pass_lcb.pdf"),
        help="Output PDF path (default: figures/fig_budget_pass_lcb.pdf).",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/fig_budget_pass_lcb.png"),
        help="Output PNG path (default: figures/fig_budget_pass_lcb.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    b = [2, 4, 6, 8]
    subset_size = [12, 24, 36, 48]
    pass_rate = [0.0000, 0.0075, 0.0375, 0.1525]
    wilson_lcb = [0.0000, 0.0030, 0.0247, 0.1253]
    p_min = 0.08

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    y_max = 0.17
    ax.axhspan(p_min, y_max, color="#e8f4ef", alpha=0.8, zorder=0)
    ax.axhline(
        p_min,
        linestyle=(0, (5, 4)),
        color="#238b6c",
        linewidth=1.6,
        label=r"Reliability target $p_{\min}$",
        zorder=1,
    )

    ax.plot(
        subset_size,
        pass_rate,
        marker="o",
        markersize=6.5,
        linewidth=2.6,
        color="#1d4ed8",
        label="Empirical pass rate",
        zorder=3,
    )
    ax.plot(
        subset_size,
        wilson_lcb,
        marker="s",
        markersize=6.2,
        linewidth=2.6,
        color="#f97316",
        label="Wilson LCB",
        zorder=3,
    )

    selected_x = 48
    selected_y = 0.1253
    ax.axvline(selected_x, color="#9ca3af", linewidth=1.0, linestyle=":", zorder=1)
    ax.scatter([selected_x], [selected_y], s=74, color="#f97316", edgecolor="white", linewidth=1.2, zorder=4)
    ax.annotate(
        "Selected\n|S|=48",
        xy=(selected_x, selected_y),
        xytext=(41.3, 0.146),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#374151"},
        color="#111827",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.95},
    )

    ax.set_xlabel("Subset size |S|")
    ax.set_ylabel("Pass probability")
    ax.set_xticks(subset_size)
    ax.set_xticklabels([f"{s}\n(b={bb})" for s, bb in zip(subset_size, b)])
    ax.set_ylim(-0.004, y_max)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Fixed-k Reliability Sweep")
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", length=0)

    ax.text(
        subset_size[-1] + 0.7,
        pass_rate[-1],
        "pass rate",
        color="#1d4ed8",
        va="center",
        fontsize=9.5,
    )
    ax.text(
        subset_size[-1] + 0.7,
        wilson_lcb[-1],
        "LCB",
        color="#f97316",
        va="center",
        fontsize=9.5,
    )
    ax.text(
        subset_size[0] + 0.7,
        p_min + 0.003,
        r"$p_{\min}=0.08$",
        color="#16664f",
        va="bottom",
        fontsize=9.5,
    )
    ax.set_xlim(10, 52.8)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#d1d5db",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    fig.savefig(args.output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output}")
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
