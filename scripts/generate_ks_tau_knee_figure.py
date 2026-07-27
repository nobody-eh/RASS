#!/usr/bin/env python3
"""Generate KS-threshold knee figure from a tau-vs-min-size CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


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
        default=Path("sweep_cluster_k/ks_threshold_pilot/ks_tau_vs_min_subset_size.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/fig_ks_tau_knee.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("figures/fig_ks_tau_knee.png"),
        help="Output PNG path.",
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

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    if not feasible.empty:
        ax.plot(
            feasible["tau_ks"].to_numpy(),
            feasible["min_size"].to_numpy(),
            marker="o",
            markersize=7.0,
            linewidth=2.8,
            color="#0f766e",
            markeredgecolor="white",
            markeredgewidth=1.1,
            zorder=3,
        )
        for _, row in feasible.iterrows():
            tau = float(row["tau_ks"])
            size = float(row["min_size"])
            ax.text(
                tau,
                size + 1.25,
                f"{int(size)} scenes",
                ha="center",
                va="bottom",
                color="#0f766e",
                fontsize=9.0,
                fontweight="semibold",
            )

    if not infeasible.empty:
        y_top = (
            float(feasible["min_size"].max()) + 7.0
            if not feasible.empty
            else float(args.chosen_size) + 8.0
        )
        ax.plot(
            infeasible["tau_ks"].to_numpy(),
            np.full(len(infeasible), y_top),
            linestyle="none",
            marker="x",
            markersize=9.0,
            markeredgewidth=2.0,
            color="#f97316",
            zorder=4,
        )
        for x in infeasible["tau_ks"].to_numpy():
            ax.text(
                float(x) + 0.003,
                y_top,
                "No feasible\nsubset",
                ha="left",
                va="center",
                color="#c2410c",
                fontsize=9.0,
                fontweight="semibold",
            )
    else:
        y_top = float(feasible["min_size"].max()) + 8.0 if not feasible.empty else float(args.chosen_size) + 8.0

    ax.axvspan(args.chosen_tau - 0.004, args.chosen_tau + 0.004, color="#ccfbf1", alpha=0.7, zorder=0)
    ax.axvline(args.chosen_tau, color="#0f766e", linestyle=(0, (4, 4)), linewidth=1.4, zorder=1)
    ax.scatter(
        [args.chosen_tau],
        [args.chosen_size],
        s=120,
        color="#0f766e",
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
    )
    ax.annotate(
        f"Selected tolerance\n$\\tau_{{KS}}$={args.chosen_tau:.2f}, |S|={args.chosen_size}",
        xy=(args.chosen_tau, args.chosen_size),
        xytext=(args.chosen_tau + 0.017, args.chosen_size + 5.2),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#334155"},
        ha="left",
        va="center",
        color="#0f172a",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": "#dbe3ee", "alpha": 0.96},
    )

    ax.text(
        0.5,
        1.03,
        "Looser KS tolerance reduces the minimum feasible subset size.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#475569",
        fontsize=9.4,
    )
    y_min = float(feasible["min_size"].min()) - 5.0 if not feasible.empty else float(args.chosen_size) - 5.0

    ax.set_xlabel(r"KS tolerance $\tau_{KS}$")
    ax.set_ylabel("Minimum feasible subset size |S|")
    ax.set_title("KS Tolerance Sensitivity", fontweight="semibold", pad=25)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.9)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    ax.set_ylim(y_min, y_top + 3.0)
    ax.set_xlim(float(plot_df["tau_ks"].min()) - 0.004, float(plot_df["tau_ks"].max()) + 0.004)
    ax.set_xticks(plot_df["tau_ks"].to_numpy())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(args.output_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output_pdf}")
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
