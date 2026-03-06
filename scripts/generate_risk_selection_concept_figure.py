#!/usr/bin/env python3
"""Generate conceptual risk-aware selection figure for paper inclusion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Conceptual sweep points.
    x = np.array([12, 24, 36, 48], dtype=float)
    pass_rate = np.array([0.00, 0.01, 0.04, 0.15], dtype=float)
    lcb = np.array([0.00, 0.003, 0.025, 0.125], dtype=float)
    p_min = 0.08
    x_selected = 48.0

    # Two-column friendly figure size (inches).
    fig, ax = plt.subplots(figsize=(3.55, 2.55), dpi=300)

    # Subtle region shading.
    x_left, x_right = 10.0, 52.0
    ax.axvspan(x_left, x_selected, color="#D9D9D9", alpha=0.14, zorder=0)
    ax.axvspan(x_selected, x_right, color="#E3F2E5", alpha=0.18, zorder=0)

    pass_color = "#4C78A8"
    lcb_color = "#F58518"
    target_color = "#5F6368"

    # Curves.
    ax.plot(
        x,
        pass_rate,
        marker="o",
        color=pass_color,
        linewidth=1.8,
        markersize=4.0,
        label="Empirical pass rate",
        zorder=3,
    )
    ax.plot(
        x,
        lcb,
        marker="o",
        color=lcb_color,
        linewidth=1.8,
        markersize=4.0,
        label="Wilson LCB",
        zorder=3,
    )
    ax.axhline(
        p_min,
        linestyle="--",
        color=target_color,
        linewidth=1.4,
        label="Reliability target",
        zorder=2,
    )

    # Selected budget marker and line.
    y_sel = float(lcb[x.tolist().index(x_selected)])
    ax.axvline(x_selected, linestyle="--", color=target_color, linewidth=1.2, zorder=2)
    ax.scatter(
        [x_selected],
        [y_sel],
        s=30,
        facecolor="white",
        edgecolor=lcb_color,
        linewidth=1.1,
        zorder=4,
    )

    # Annotations.
    ax.annotate(
        "selected budget",
        xy=(x_selected, y_sel),
        xytext=(39.5, 0.140),
        fontsize=7.8,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": target_color},
    )
    ax.annotate(
        r"$p_{\min}$",
        xy=(11.8, p_min),
        xytext=(12.6, p_min + 0.010),
        fontsize=7.8,
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": target_color},
    )
    ax.text(22.0, 0.156, "too risky", fontsize=7.7, color="#444444", ha="center", va="bottom")
    ax.text(49.6, 0.156, "acceptable reliability", fontsize=7.7, color="#444444", ha="center", va="bottom")

    # Axes formatting.
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0.0, 0.17)
    ax.set_xticks(x)
    ax.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16])
    ax.set_xlabel(r"Subset size $|\mathcal{S}|$", fontsize=9.5)
    ax.set_ylabel("Joint pass probability", fontsize=9.5)
    ax.tick_params(labelsize=8.6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=7.3, framealpha=0.95, borderpad=0.25, handlelength=1.8)

    fig.tight_layout(pad=0.35)

    out_pdf = Path("fig_risk_selection_concept.pdf")
    out_png = Path("fig_risk_selection_concept.png")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
