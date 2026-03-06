#!/usr/bin/env python3
"""Export publication-quality before/after normalization histograms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "volume",
    "extent_x",
    "bbox_min_x",
    "point_density",
    "extent_y",
    "bbox_max_y",
]

FEATURE_SYMBOLS = {
    "volume": r"$V$",
    "extent_x": r"$E_x$",
    "bbox_min_x": r"$x_{\min}^{\mathrm{bbox}}$",
    "point_density": r"$\rho_p$",
    "extent_y": r"$E_y$",
    "bbox_max_y": r"$y_{\max}^{\mathrm{bbox}}$",
}


def _parse_features(value: str | None) -> List[str]:
    if not value:
        return DEFAULT_FEATURES
    return [item.strip() for item in value.split(",") if item.strip()]


def _winsor_bounds(series: pd.Series, q_low: float, q_high: float) -> tuple[float, float]:
    vals = series.dropna().to_numpy()
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [q_low, q_high])
    if np.isclose(lo, hi):
        span = max(1.0, abs(lo) * 0.1)
        lo, hi = lo - span, hi + span
    return float(lo), float(hi)


def _signed_log10(values: np.ndarray) -> np.ndarray:
    """Display-only transform to compress heavy tails and keep sign."""
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def build_figure(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    features: List[str],
    out_pdf: Path,
    out_png: Path,
    q_low: float,
    q_high: float,
    raw_display: str,
) -> None:
    rows = len(features)
    fig_h = 1.25 * rows + 0.8
    fig, axes = plt.subplots(rows, 2, figsize=(7.1, fig_h), dpi=300, squeeze=False)

    for i, feat in enumerate(features):
        if feat not in before_df.columns or feat not in after_df.columns:
            raise KeyError(f"Feature '{feat}' is missing in input CSV files.")

        raw = before_df[feat]
        norm = after_df[feat]

        raw_ax = axes[i, 0]
        norm_ax = axes[i, 1]

        raw_vals = raw.dropna().to_numpy()
        if raw_display == "signed-log":
            raw_vals = _signed_log10(raw_vals)
            lo, hi = _winsor_bounds(pd.Series(raw_vals), q_low=q_low, q_high=q_high)
            raw_vals = np.clip(raw_vals, lo, hi)
            raw_ax.hist(raw_vals, bins=35, alpha=0.75, linewidth=0.25)
            raw_ax.set_xlim(lo, hi)
            raw_ax.axvline(0.0, linewidth=0.8, alpha=0.35)
        else:
            lo, hi = _winsor_bounds(raw, q_low=q_low, q_high=q_high)
            raw_vals = np.clip(raw_vals, lo, hi)
            raw_ax.hist(raw_vals, bins=35, alpha=0.75, linewidth=0.25)
            raw_ax.set_xlim(lo, hi)
        raw_ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        raw_ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        raw_ax.set_ylabel("Count", fontsize=8.5)

        norm_vals = norm.dropna().to_numpy()
        nlo, nhi = _winsor_bounds(norm, q_low=q_low, q_high=q_high)
        norm_ax.hist(norm_vals, bins=35, alpha=0.75, linewidth=0.25)
        norm_ax.set_xlim(nlo, nhi)
        norm_ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        norm_ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))

        feature_label = FEATURE_SYMBOLS.get(feat, feat)
        raw_ax.text(
            0.02,
            0.88,
            feature_label,
            transform=raw_ax.transAxes,
            fontsize=8.3,
            fontweight="bold",
            ha="left",
            va="top",
        )

        raw_ax.tick_params(labelsize=8.0)
        norm_ax.tick_params(labelsize=8.0)

        if i == 0:
            raw_title = "Before normalization (display-winsorized)"
            if raw_display == "signed-log":
                raw_title = "Before normalization (signed-log display)"
            raw_ax.set_title(raw_title, fontsize=9.5)
            norm_ax.set_title("After normalization", fontsize=9.5)

    left_xlabel = "Raw value"
    display_note = f"Display note: left-column histograms are clipped to [{q_low:.1f}, {q_high:.1f}] percentiles for readability."
    if raw_display == "signed-log":
        left_xlabel = r"Raw value (signed $\log_{10}(1+|x|)$)"
        display_note = (
            "Display note: left-column histograms use signed "
            r"$\log_{10}(1+|x|)$ plus percentile clipping for readability."
        )

    axes[-1, 0].set_xlabel(left_xlabel, fontsize=8.8)
    axes[-1, 1].set_xlabel("Normalized value", fontsize=8.8)

    fig.text(
        0.5,
        0.005,
        display_note,
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="0.35",
    )

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0], pad=0.45)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export before/after normalization histograms.")
    parser.add_argument("--before-csv", default="feats2.csv", help="Raw features CSV path.")
    parser.add_argument("--after-csv", default="feats_normalized.csv", help="Normalized features CSV path.")
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature list to plot.",
    )
    parser.add_argument("--q-low", type=float, default=0.5, help="Lower percentile for display winsorization.")
    parser.add_argument("--q-high", type=float, default=99.5, help="Upper percentile for display winsorization.")
    parser.add_argument(
        "--raw-display",
        choices=["winsor", "signed-log"],
        default="signed-log",
        help="How to display raw-feature histograms (data unchanged).",
    )
    parser.add_argument("--out-pdf", default="plots/before_after_hist_cvpr.pdf", help="Output PDF path.")
    parser.add_argument("--out-png", default="plots/before_after_hist_cvpr.png", help="Output PNG path.")
    args = parser.parse_args()

    feats = _parse_features(args.features)
    before_df = pd.read_csv(args.before_csv)
    after_df = pd.read_csv(args.after_csv)
    build_figure(
        before_df=before_df,
        after_df=after_df,
        features=feats,
        out_pdf=Path(args.out_pdf),
        out_png=Path(args.out_png),
        q_low=args.q_low,
        q_high=args.q_high,
        raw_display=args.raw_display,
    )
    print(f"Saved: {args.out_pdf}")
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()
