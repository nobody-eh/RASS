#!/usr/bin/env python3
"""Generate KS-threshold knee figure from a tau-vs-min-size CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read /mnt/data/ks_tau_vs_min_subset_size.csv and generate "
            "figures/fig_ks_tau_knee.pdf"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/data/ks_tau_vs_min_subset_size.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/fig_ks_tau_knee.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--chosen-tau",
        type=float,
        default=0.14,
        help="Chosen tau_KS for annotation.",
    )
    parser.add_argument(
        "--chosen-size",
        type=int,
        default=36,
        help="Chosen minimum feasible subset size for annotation.",
    )
    return parser.parse_args()


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_to_orig = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col(cand)
        if key in norm_to_orig:
            return norm_to_orig[key]
    return None


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    tau_col = _pick_col(df, ["tau_ks", "tau", "tauks"])
    size_col = _pick_col(
        df,
        [
            "min_feasible_subset_size",
            "minimal_subset_size",
            "minimum_subset_size",
            "min_subset_size",
            "subset_size",
        ],
    )
    if tau_col is None or size_col is None:
        raise ValueError(
            "Could not detect required columns. Expected tau_ks and minimum-size columns."
        )

    plot_df = df[[tau_col, size_col]].copy()
    plot_df.columns = ["tau_ks", "min_size"]
    plot_df["tau_ks"] = pd.to_numeric(plot_df["tau_ks"], errors="coerce")
    plot_df["min_size"] = pd.to_numeric(plot_df["min_size"], errors="coerce")
    plot_df = plot_df.dropna(subset=["tau_ks"]).sort_values("tau_ks", kind="mergesort")

    feasible = plot_df[plot_df["min_size"].notna()].copy()
    infeasible = plot_df[plot_df["min_size"].isna()].copy()

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    if not feasible.empty:
        ax.plot(
            feasible["tau_ks"].to_numpy(),
            feasible["min_size"].to_numpy(),
            marker="o",
            label="Min feasible subset size",
        )

    if not infeasible.empty:
        y_top = (
            float(feasible["min_size"].max()) + 8.0
            if not feasible.empty
            else float(args.chosen_size) + 8.0
        )
        ax.plot(
            infeasible["tau_ks"].to_numpy(),
            np.full(len(infeasible), y_top),
            linestyle="none",
            marker="x",
            label="Infeasible",
        )
        for x in infeasible["tau_ks"].to_numpy():
            ax.text(x, y_top + 0.8, "infeasible", ha="center", va="bottom", fontsize=9)
    else:
        y_top = float(feasible["min_size"].max()) + 8.0 if not feasible.empty else float(args.chosen_size) + 8.0

    ax.scatter([args.chosen_tau], [args.chosen_size], zorder=4)
    ax.annotate(
        f"chosen: tau={args.chosen_tau:.2f}, size={args.chosen_size}",
        xy=(args.chosen_tau, args.chosen_size),
        xytext=(args.chosen_tau + 0.015, args.chosen_size + 6),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    ax.set_xlabel(r"$\tau_{KS}$")
    ax.set_ylabel("Minimum feasible subset size")
    ax.set_title("KS Threshold Knee")
    ax.grid(axis="y", alpha=0.3)

    y_min = float(feasible["min_size"].min()) - 6.0 if not feasible.empty else float(args.chosen_size) - 6.0
    ax.set_ylim(y_min, y_top + 3.0)
    ax.set_xticks(plot_df["tau_ks"].to_numpy())
    ax.legend()

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_pdf, format="pdf")
    plt.close(fig)

    print(f"Saved: {args.output_pdf}")


if __name__ == "__main__":
    main()
