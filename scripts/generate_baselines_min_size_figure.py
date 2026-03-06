#!/usr/bin/env python3
"""Generate a horizontal bar chart of minimum passing subset size by method."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read baseline min-size summary CSV and generate "
            "figures/fig_baselines_min_size.pdf"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/mnt/data/baseline_min_size_summary.csv"),
        help="Path to baseline summary CSV.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/fig_baselines_min_size.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--bass-size",
        type=int,
        default=48,
        help="Reference BASS subset size for vertical line.",
    )
    parser.add_argument(
        "--failed-limit-default",
        type=int,
        default=96,
        help="Fallback limit to render failed methods (e.g., >96).",
    )
    return parser.parse_args()


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _short_method_name(raw: str) -> str:
    key = str(raw).strip().lower()
    mapping = {
        "random_uniform": "Random",
        "random": "Random",
        "kcenter_farthest_first": "k-center",
        "k-center": "k-center",
        "kcenter": "k-center",
        "facility_location": "Facility location",
        "facility": "Facility location",
        "bass": "BASS",
        "ours": "BASS",
        "method": "BASS",
    }
    return mapping.get(key, str(raw))


def _to_bool_failed(status_val: object) -> bool:
    txt = str(status_val).strip().lower()
    return "fail" in txt


def build_plot_table(df: pd.DataFrame, failed_limit_default: int) -> pd.DataFrame:
    method_col = _pick_column(df, ["baseline", "method", "name"])
    if method_col is None:
        raise ValueError("Could not find method column (expected baseline/method/name).")

    min_col = _pick_column(
        df,
        [
            "smallest_size_meeting_lcb",
            "min_subset_size",
            "minimum_subset_size",
            "smallest_size",
            "size",
        ],
    )
    status_col = _pick_column(df, ["status"])
    largest_col = _pick_column(df, ["largest_tested_size", "max_tested_size", "tested_limit"])

    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        method = _short_method_name(str(r[method_col]))
        size_val = np.nan if min_col is None else pd.to_numeric(r[min_col], errors="coerce")
        failed = False

        if status_col is not None and _to_bool_failed(r[status_col]):
            failed = True
        if pd.isna(size_val):
            failed = True

        if failed:
            tested_lim = failed_limit_default
            if largest_col is not None:
                v = pd.to_numeric(r[largest_col], errors="coerce")
                if not pd.isna(v):
                    tested_lim = int(v)
            plot_size = int(tested_lim)
            label = f">{tested_lim}"
        else:
            plot_size = int(size_val)
            label = str(int(size_val))

        rows.append(
            {
                "method": method,
                "plot_size": plot_size,
                "failed": failed,
                "label": label,
            }
        )

    out = pd.DataFrame(rows)

    # Stable, paper-friendly order when available.
    preferred = ["Random", "k-center", "Facility location", "BASS"]
    out["order"] = out["method"].apply(lambda x: preferred.index(x) if x in preferred else len(preferred))
    out = out.sort_values(["order", "method"], kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    plot_df = build_plot_table(df, failed_limit_default=args.failed_limit_default)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    n = len(plot_df)
    fig_h = max(2.8, 0.7 * n + 1.2)
    fig, ax = plt.subplots(figsize=(6.2, fig_h))

    y = np.arange(n)
    bars = ax.barh(y, plot_df["plot_size"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["method"].tolist())
    ax.invert_yaxis()  # top-to-bottom order
    ax.set_xlabel("Minimum subset size")
    ax.set_ylabel("Method")
    ax.set_title("Minimum Passing Subset Size by Method")
    ax.grid(axis="x", alpha=0.3)

    for i, (bar, failed, label_txt) in enumerate(
        zip(bars, plot_df["failed"].tolist(), plot_df["label"].tolist())
    ):
        if failed:
            bar.set_hatch("//")
        x = float(bar.get_width())
        y_mid = float(bar.get_y() + bar.get_height() / 2.0)
        ax.text(x + 1.0, y_mid, label_txt, va="center", ha="left")

    ax.axvline(args.bass_size, linestyle="--")
    ax.text(
        args.bass_size + 0.8,
        -0.45,
        "BASS subset size",
        ha="left",
        va="center",
    )

    xmax = max(int(plot_df["plot_size"].max()), int(args.bass_size)) + 12
    ax.set_xlim(0, xmax)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_pdf, format="pdf")
    plt.close(fig)
    print(f"Saved: {args.output_pdf}")


if __name__ == "__main__":
    main()
