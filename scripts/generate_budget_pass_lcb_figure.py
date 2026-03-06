#!/usr/bin/env python3
"""Generate the fixed-k pass-rate/LCB figure as a vector PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    b = [2, 4, 6, 8]
    subset_size = [12, 24, 36, 48]
    pass_rate = [0.0000, 0.0075, 0.0375, 0.1525]
    wilson_lcb = [0.0000, 0.0030, 0.0247, 0.1253]
    p_min = 0.08

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    ax.plot(subset_size, pass_rate, marker="o", label="pass_rate")
    ax.plot(subset_size, wilson_lcb, marker="o", label="wilson_lcb")
    ax.axhline(p_min, linestyle="--", label="p_min")

    selected_x = 48
    selected_y = 0.1253
    ax.scatter([selected_x], [selected_y], zorder=4)
    ax.annotate(
        "selected",
        xy=(selected_x, selected_y),
        xytext=(selected_x - 10, selected_y + 0.03),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    ax.set_xlabel("Subset size |S|")
    ax.set_ylabel("Rate")
    ax.set_xticks(subset_size)
    ax.set_xticklabels([f"{s}\n(b={bb})" for s, bb in zip(subset_size, b)])
    ax.set_title("Fixed-k Sweep: Pass Rate and Wilson LCB")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, format="pdf")
    plt.close(fig)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
