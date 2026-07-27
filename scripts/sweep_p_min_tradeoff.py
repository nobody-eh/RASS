#!/usr/bin/env python3
"""Sweep the target Wilson-LCB pass probability p_min.

This script consumes an existing budget-sweep or joint k x budget summary and
re-applies the repository's current recommendation rule for a list of p_min
targets, producing a tradeoff table and figure.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sweep_k_budget_selection as skb
import sweep_subset_budgets as sb


LOGGER = logging.getLogger("sweep_p_min_tradeoff")


def _parse_p_values(value: str) -> List[float]:
    out: List[float] = []
    for token in str(value).split(","):
        s = token.strip()
        if not s:
            continue
        v = float(s)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"p_min must be in [0, 1], got {v}")
        out.append(v)
    if not out:
        raise ValueError("No valid p_min values provided")
    return out


def _load_summary_inputs(
    summary_csv: Path,
    summary_json: Optional[Path],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(summary_csv)
    payload: Dict[str, object] = {}
    if summary_json is not None and summary_json.exists():
        with summary_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    return df, payload


def _infer_selection_kind(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    if "k" in df.columns:
        return "joint"
    return "budget"


def _select_joint_row(summary_df: pd.DataFrame, rec: Dict[str, object]) -> pd.Series:
    manifest = str(rec["recommended_manifest"])
    matches = summary_df[summary_df["best_manifest"].astype(str) == manifest].copy()
    if matches.empty:
        matches = summary_df[
            (pd.to_numeric(summary_df["k"], errors="coerce") == int(rec["recommended_k"]))
            & (
                pd.to_numeric(summary_df["budget_per_cluster"], errors="coerce")
                == int(rec["recommended_budget_per_cluster"])
            )
        ].copy()
    if matches.empty:
        raise RuntimeError(f"Could not resolve selected row for manifest: {manifest}")
    return matches.iloc[0].copy()


def _run_budget_selector(
    summary_df: pd.DataFrame,
    payload: Dict[str, object],
    p_min: float,
) -> Dict[str, object]:
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}
    args = SimpleNamespace(
        require_best_manifest_constraints=bool(
            config.get("require_best_manifest_constraints", True)
        ),
        max_total_subset=config.get("max_total_subset", None),
        target_joint_pass_rate=float(p_min),
        selection_use_lcb=bool(config.get("selection_use_lcb", True)),
    )
    selected_row, rec = sb._select_budget(summary_df.copy(), args)
    return {
        "recommended_row": selected_row,
        "recommendation": rec,
    }


def _run_joint_selector(
    summary_df: pd.DataFrame,
    payload: Dict[str, object],
    p_min: float,
) -> Dict[str, object]:
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}
    rec = skb._select_joint(
        summary_df.copy(),
        target_joint_pass_rate=float(p_min),
        use_lcb=bool(config.get("selection_use_lcb", True)),
        max_total_subset=config.get("max_total_subset", None),
    )
    selected_row = _select_joint_row(summary_df, rec)
    return {
        "recommended_row": selected_row,
        "recommendation": rec,
    }


def _sweep_p_min(
    summary_df: pd.DataFrame,
    payload: Dict[str, object],
    selection_kind: str,
    p_values: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p_min in p_values:
        try:
            if selection_kind == "joint":
                out = _run_joint_selector(summary_df, payload, p_min)
            else:
                out = _run_budget_selector(summary_df, payload, p_min)
        except RuntimeError:
            rows.append(
                {
                    "p_min": float(p_min),
                    "recommended_subset_size": np.nan,
                    "observed_pass_rate_at_reported_size": np.nan,
                    "wilson_lcb_at_reported_size": np.nan,
                    "no_feasible_subset_found": True,
                    "selection_mode": "no_feasible_rows",
                    "target_met": False,
                    "recommended_budget_per_cluster": np.nan,
                    "recommended_k": np.nan,
                    "recommended_manifest": "",
                }
            )
            continue

        selected_row = out["recommended_row"]
        rec = out["recommendation"]
        rows.append(
            {
                "p_min": float(p_min),
                "recommended_subset_size": int(rec["recommended_total_subset_size"]),
                "observed_pass_rate_at_reported_size": float(rec["recommended_joint_pass_rate"]),
                "wilson_lcb_at_reported_size": float(rec["recommended_joint_pass_rate_lcb"]),
                "no_feasible_subset_found": bool(not rec["target_met"]),
                "selection_mode": str(rec["selection_mode"]),
                "target_met": bool(rec["target_met"]),
                "recommended_budget_per_cluster": int(rec["recommended_budget_per_cluster"]),
                "recommended_k": (
                    int(rec["recommended_k"]) if "recommended_k" in rec else np.nan
                ),
                "recommended_manifest": str(rec["recommended_manifest"]),
                "row_joint_pass_rate": float(pd.to_numeric(selected_row["joint_pass_rate"], errors="coerce")),
                "row_joint_pass_rate_lcb": float(
                    pd.to_numeric(selected_row["joint_pass_rate_lcb"], errors="coerce")
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_latex_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{rrrrl}",
        "\\hline",
        "$p_{\\min}$ & Subset size & Pass rate & Wilson LCB & No feasible subset? \\\\",
        "\\hline",
    ]
    for row in df.to_dict("records"):
        size_val = row["recommended_subset_size"]
        size_txt = "-" if pd.isna(size_val) else str(int(size_val))
        pass_txt = "-" if pd.isna(row["observed_pass_rate_at_reported_size"]) else f"{float(row['observed_pass_rate_at_reported_size']):.3f}"
        lcb_txt = "-" if pd.isna(row["wilson_lcb_at_reported_size"]) else f"{float(row['wilson_lcb_at_reported_size']):.3f}"
        fail_txt = "Yes" if bool(row["no_feasible_subset_found"]) else "No"
        lines.append(
            f"{float(row['p_min']):.2f} & {size_txt} & {pass_txt} & {lcb_txt} & {fail_txt} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    current_p_min: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )

    plot_df = df.copy()
    plot_df["p_min"] = pd.to_numeric(plot_df["p_min"], errors="coerce")
    plot_df["recommended_subset_size"] = pd.to_numeric(
        plot_df["recommended_subset_size"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["p_min", "recommended_subset_size"]).copy()
    plot_df = plot_df.sort_values("p_min", kind="mergesort").reset_index(drop=True)
    plot_df["wilson_lcb_at_reported_size"] = pd.to_numeric(
        plot_df["wilson_lcb_at_reported_size"], errors="coerce"
    )
    plot_df["margin"] = plot_df["wilson_lcb_at_reported_size"] - plot_df["p_min"]
    plot_df["met_target"] = ~plot_df["no_feasible_subset_found"].astype(bool)

    x = np.arange(len(plot_df))
    x_labels = [f"{v:.2f}" for v in plot_df["p_min"].tolist()]

    fig, (ax_size, ax_margin) = plt.subplots(
        2,
        1,
        figsize=(8.7, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.18},
    )

    text = "#0f172a"
    muted = "#64748b"
    line_color = "#334155"
    pass_color = "#059669"
    fallback_color = "#dc2626"
    grid = "#e2e8f0"
    current_color = "#f8fafc"

    current_matches = np.where(np.isclose(plot_df["p_min"].to_numpy(dtype=float), float(current_p_min)))[0]
    current_idx = int(current_matches[0]) if len(current_matches) else None

    for ax in (ax_size, ax_margin):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=grid, linewidth=0.9, zorder=0)
        ax.tick_params(axis="both", length=0, colors=muted)
        if current_idx is not None:
            ax.axvspan(
                current_idx - 0.38,
                current_idx + 0.38,
                color=current_color,
                ec="#cbd5e1",
                lw=0.8,
                zorder=0,
            )

    ax_size.plot(x, plot_df["recommended_subset_size"], color=line_color, linewidth=2.1, zorder=2)
    ax_size.fill_between(
        x,
        plot_df["recommended_subset_size"].astype(float),
        color="#e2e8f0",
        alpha=0.45,
        zorder=1,
    )

    for idx, row in plot_df.iterrows():
        ok = bool(row["met_target"])
        color = pass_color if ok else fallback_color
        face = color if ok else "white"
        ax_size.scatter(
            idx,
            float(row["recommended_subset_size"]),
            s=92,
            facecolor=face,
            edgecolor=color,
            linewidth=2.0,
            zorder=4,
        )
        ax_size.text(
            idx,
            float(row["recommended_subset_size"]) + 0.55,
            f"{int(row['recommended_subset_size'])}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=text,
            fontweight="semibold",
        )

    y_min = max(0.0, float(plot_df["recommended_subset_size"].min()) - 3.0)
    y_max = float(plot_df["recommended_subset_size"].max()) + 4.0
    ax_size.set_ylim(y_min, y_max)
    ax_size.set_ylabel("Selected scenes")
    ax_size.set_title("Budget selected for each reliability target", fontweight="semibold", color=text, pad=8)

    if current_idx is not None:
        cur_row = plot_df.iloc[current_idx]
        status = "fallback" if bool(cur_row["no_feasible_subset_found"]) else "meets target"
        ax_size.annotate(
            f"paper setting\n$p_{{min}}={current_p_min:.2f}$, {status}",
            xy=(current_idx, float(cur_row["recommended_subset_size"])),
            xytext=(current_idx + 0.55, float(cur_row["recommended_subset_size"]) + 2.3),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": muted},
            ha="left",
            va="bottom",
            fontsize=9.2,
            color=text,
        )

    margins = plot_df["margin"].astype(float).tolist()
    bar_colors = [pass_color if v >= 0 else fallback_color for v in margins]
    ax_margin.bar(x, margins, color=bar_colors, width=0.58, zorder=3)
    ax_margin.axhline(0, color="#334155", linewidth=1.0, zorder=2)
    ax_margin.set_ylabel("LCB - target")
    ax_margin.set_xlabel("$p_{min}$ target")
    ax_margin.set_title("Reliability margin at reported budget", fontweight="semibold", color=text, pad=8)

    margin_abs = max(abs(float(np.nanmin(margins))), abs(float(np.nanmax(margins))), 0.03)
    ax_margin.set_ylim(-margin_abs * 1.22, margin_abs * 1.12)

    for idx, row in plot_df.iterrows():
        margin = float(row["margin"])
        va = "bottom" if margin >= 0 else "top"
        y_offset = margin_abs * 0.045 if margin >= 0 else -margin_abs * 0.045
        ax_margin.text(
            idx,
            margin + y_offset,
            f"{margin:+.3f}",
            ha="center",
            va=va,
            fontsize=8.8,
            color=text,
            fontweight="semibold",
        )

    ax_margin.set_xticks(x)
    ax_margin.set_xticklabels(x_labels)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=pass_color,
            markeredgecolor=pass_color,
            markersize=8,
            label="meets target",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=fallback_color,
            markeredgewidth=2,
            markersize=8,
            label="fallback",
        ),
    ]

    fig.suptitle("Reliability Target vs Selected Budget", fontsize=15.2, fontweight="semibold", color=text, y=0.975)
    fig.text(
        0.5,
        0.927,
        "Only the lowest target is met; higher targets reuse the best available 48-scene fallback in this grid.",
        ha="center",
        va="top",
        color="#475569",
        fontsize=9.4,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        frameon=False,
        fontsize=9.4,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.79, bottom=0.11)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_reproduction_note(
    path: Path,
    summary_csv: Path,
    summary_json: Optional[Path],
    selection_kind: str,
    p_values: Sequence[float],
    result_df: pd.DataFrame,
    output_dir: Path,
    run_command: str,
) -> None:
    lines = [
        "# p_min Sweep Reproduction",
        "",
        "## Inputs",
        f"- Summary CSV: `{summary_csv}`",
        f"- Summary JSON: `{summary_json}`" if summary_json is not None else "- Summary JSON: `(not provided)`",
        f"- Selection kind: `{selection_kind}`",
        f"- Swept values: `{', '.join(f'{v:.2f}' for v in p_values)}`",
        "",
        "## Command",
        "```bash",
        run_command,
        "```",
        "",
        "## Outputs",
        f"- `p_min_sweep_summary.csv`",
        f"- `p_min_sweep_summary.tex`",
        f"- `fig_p_min_tradeoff.pdf`",
        f"- `fig_p_min_tradeoff.png`",
        f"- `run_config.json`",
        "",
        "Interpretation note:",
        "- `no_feasible_subset_found=True` means no candidate within the searched range satisfied the requested Wilson-LCB target, so the selector fell back to the highest-LCB option allowed by the source sweep.",
        "",
    ]

    cur = result_df[np.isclose(pd.to_numeric(result_df["p_min"], errors="coerce"), 0.08)]
    if not cur.empty:
        row = cur.iloc[0]
        fail_txt = "is a fallback" if bool(row["no_feasible_subset_found"]) else "meets the target"
        lines += [
            "## Result Snapshot",
            (
                f"- At `p_min=0.08`, the selected subset size is **{int(row['recommended_subset_size'])}**; "
                f"the chosen point has observed pass rate **{float(row['observed_pass_rate_at_reported_size']):.3f}**, "
                f"Wilson LCB **{float(row['wilson_lcb_at_reported_size']):.3f}**, and {fail_txt}."
            ),
        ]

        lower = result_df[
            pd.to_numeric(result_df["p_min"], errors="coerce") < 0.08
        ].sort_values("p_min", kind="mergesort")
        if not lower.empty:
            prev = lower.iloc[-1]
            lines.append(
                f"- The nearest lower target in this sweep is `p_min={float(prev['p_min']):.2f}`, "
                f"which selects **{int(prev['recommended_subset_size'])}** scenes."
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("sweep_cluster_k/holdout_protocol_v3/joint_selection/k_budget_sweep_summary.csv"),
        help="Existing budget or joint-sweep summary CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("sweep_cluster_k/holdout_protocol_v3/joint_selection/k_budget_sweep_summary.json"),
        help="Optional JSON payload saved by the same sweep.",
    )
    parser.add_argument(
        "--selection-kind",
        choices=["auto", "budget", "joint"],
        default="auto",
        help="Which existing selector to reuse.",
    )
    parser.add_argument(
        "--p-min-values",
        type=str,
        default="0.05,0.08,0.10,0.20,0.30,0.50",
        help="Comma-separated p_min values to sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/p_min_tradeoff/holdout_protocol_v3_joint_selection"),
        help="Directory for the exported sweep artifacts.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    p_values = _parse_p_values(args.p_min_values)
    summary_df, payload = _load_summary_inputs(args.summary_csv, args.summary_json)
    selection_kind = _infer_selection_kind(summary_df, args.selection_kind)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = _sweep_p_min(summary_df, payload, selection_kind, p_values)

    result_df.to_csv(output_dir / "p_min_sweep_summary.csv", index=False)
    _write_latex_table(output_dir / "p_min_sweep_summary.tex", result_df)
    _save_figure(
        result_df,
        out_pdf=output_dir / "fig_p_min_tradeoff.pdf",
        out_png=output_dir / "fig_p_min_tradeoff.png",
        current_p_min=0.08,
    )

    run_command = (
        "python3 scripts/sweep_p_min_tradeoff.py "
        f"--summary-csv {args.summary_csv} "
        f"--summary-json {args.summary_json} "
        f"--selection-kind {selection_kind} "
        f"--p-min-values {','.join(f'{v:.2f}' for v in p_values)} "
        f"--output-dir {output_dir}"
    )
    _write_reproduction_note(
        output_dir / "README.md",
        summary_csv=args.summary_csv,
        summary_json=args.summary_json,
        selection_kind=selection_kind,
        p_values=p_values,
        result_df=result_df,
        output_dir=output_dir,
        run_command=run_command,
    )

    run_config = {
        "summary_csv": str(args.summary_csv),
        "summary_json": str(args.summary_json) if args.summary_json is not None else None,
        "selection_kind": selection_kind,
        "p_min_values": [float(v) for v in p_values],
        "source_config": payload.get("config", {}),
        "source_recommendation": payload.get("recommendation", {}),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    LOGGER.info("Saved p_min sweep outputs to %s", output_dir)


if __name__ == "__main__":
    main()
