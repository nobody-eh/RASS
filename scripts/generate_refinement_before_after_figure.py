#!/usr/bin/env python3
"""Generate before/after refinement figure (abs gap + KS) as vector PDF."""

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
            "Read /mnt/data/recommended_subset_refined_r0_before_after_stats.csv "
            "and generate figures/fig_refinement_before_after.pdf"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/data/recommended_subset_refined_r0_before_after_stats.csv"),
        help="Input CSV with before/after gap and KS columns.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/fig_refinement_before_after.pdf"),
        help="Output figure path.",
    )
    return parser.parse_args()


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_to_orig = {_norm(c): c for c in df.columns}
    for c in candidates:
        k = _norm(c)
        if k in norm_to_orig:
            return norm_to_orig[k]
    return None


def _find_by_pattern(df: pd.DataFrame, must_contain: list[str]) -> str | None:
    for c in df.columns:
        n = _norm(c)
        if all(token in n for token in must_contain):
            return c
    return None


def _pick_metric_col(df: pd.DataFrame) -> str:
    metric_col = _find_col(df, ["metric", "metric_name", "name"])
    if metric_col is not None:
        return metric_col
    csv_col = _find_col(df, ["csv", "source"])
    met_col = _find_col(df, ["metric_type", "metricid"])
    if csv_col is not None and met_col is not None:
        merged_name = "__merged_metric__"
        df[merged_name] = (
            df[csv_col].astype(str).str.strip() + "_" + df[met_col].astype(str).str.strip()
        )
        return merged_name
    raise ValueError("Could not infer metric labels column.")


def _pick_abs_gap_cols(df: pd.DataFrame) -> tuple[str, str]:
    before = _find_col(df, ["abs_mean_gap_before", "before_abs_gap", "abs_gap_before"])
    after = _find_col(df, ["abs_mean_gap_after", "after_abs_gap", "abs_gap_after"])
    if before and after:
        return before, after

    # Pattern fallback (robust against naming variants).
    if before is None:
        before = _find_by_pattern(df, ["before", "abs"])
    if after is None:
        after = _find_by_pattern(df, ["after", "abs"])
    if before and after:
        return before, after

    # Last-resort fallback from signed gaps.
    before = before or _find_col(df, ["before_gap", "mean_gap_before"])
    after = after or _find_col(df, ["after_gap", "mean_gap_after"])
    if before and after:
        return before, after

    raise ValueError("Could not infer abs-mean-gap before/after columns.")


def _pick_ks_cols(df: pd.DataFrame) -> tuple[str, str]:
    before = _find_col(df, ["ks_before", "before_ks"])
    after = _find_col(df, ["ks_after", "after_ks"])
    if before and after:
        return before, after

    if before is None:
        before = _find_by_pattern(df, ["before", "ks"])
    if after is None:
        after = _find_by_pattern(df, ["after", "ks"])
    if before and after:
        return before, after

    raise ValueError("Could not infer KS before/after columns.")


def _fmt_metric_label(s: str) -> str:
    tokens = str(s).strip().replace("-", "_").split("_")
    if len(tokens) == 2:
        return f"{tokens[0].upper()}-{tokens[1].upper()}"
    return str(s).replace("_", "-")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    metric_col = _pick_metric_col(df)
    abs_before_col, abs_after_col = _pick_abs_gap_cols(df)
    ks_before_col, ks_after_col = _pick_ks_cols(df)

    plot_df = df.copy()
    plot_df["metric_label"] = plot_df[metric_col].astype(str).map(_fmt_metric_label)
    plot_df["abs_before"] = pd.to_numeric(plot_df[abs_before_col], errors="coerce")
    plot_df["abs_after"] = pd.to_numeric(plot_df[abs_after_col], errors="coerce")
    plot_df["ks_before"] = pd.to_numeric(plot_df[ks_before_col], errors="coerce")
    plot_df["ks_after"] = pd.to_numeric(plot_df[ks_after_col], errors="coerce")
    plot_df = plot_df.dropna(
        subset=["metric_label", "abs_before", "abs_after", "ks_before", "ks_after"]
    ).reset_index(drop=True)

    x = np.arange(plot_df.shape[0], dtype=float)
    width = 0.38

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=False
    )

    # Top: absolute mean gap before/after.
    ax_top.bar(x - width / 2.0, plot_df["abs_before"].to_numpy(), width=width, label="Before")
    ax_top.bar(x + width / 2.0, plot_df["abs_after"].to_numpy(), width=width, label="After")
    ax_top.set_ylabel("Abs. mean gap")
    ax_top.set_title("Refinement: Before vs After")
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.legend()

    # Bottom: KS before/after.
    ax_bottom.bar(
        x - width / 2.0, plot_df["ks_before"].to_numpy(), width=width, label="Before"
    )
    ax_bottom.bar(
        x + width / 2.0, plot_df["ks_after"].to_numpy(), width=width, label="After"
    )
    ax_bottom.set_ylabel("KS")
    ax_bottom.set_xlabel("Metric")
    ax_bottom.grid(axis="y", alpha=0.3)
    ax_bottom.legend()

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(plot_df["metric_label"].tolist(), rotation=30, ha="right")

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_pdf, format="pdf")
    plt.close(fig)
    print(f"Saved: {args.output_pdf}")


if __name__ == "__main__":
    main()
