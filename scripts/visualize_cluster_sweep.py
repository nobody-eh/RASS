#!/usr/bin/env python3
"""Create visual summaries for cluster-k sweep outputs.

This script consumes `k_sweep_summary.csv/json` plus per-k run artifacts
from `scripts/sweep_feature_clusters.py` and writes interactive HTML figures
plus a Markdown index.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LOGGER = logging.getLogger("visualize_cluster_sweep")


def _first_existing(run_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _resolve_run_dir(run_dir_value: str, sweep_dir: Path) -> Path:
    p = Path(str(run_dir_value))
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((sweep_dir / p).resolve())
        candidates.append((sweep_dir / p.name).resolve())
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _load_summary(
    summary_csv: Path, summary_json: Optional[Path]
) -> Tuple[pd.DataFrame, Optional[int], Dict[str, object]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise RuntimeError(f"Summary CSV is empty: {summary_csv}")

    rec_k: Optional[int] = None
    recommendation: Dict[str, object] = {}
    if summary_json is not None and summary_json.exists():
        with summary_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        recommendation = payload.get("recommendation", {}) or {}
        rk = recommendation.get("recommended_k")
        if rk is not None:
            rec_k = int(rk)

    if rec_k is None:
        if "rank_weighted" in df.columns:
            top = df.sort_values(["rank_weighted", "k"], kind="mergesort").iloc[0]
        elif "weighted_score" in df.columns:
            top = df.sort_values(
                ["weighted_score", "silhouette"], ascending=[False, False], kind="mergesort"
            ).iloc[0]
        else:
            top = df.sort_values(["k"], kind="mergesort").iloc[0]
        rec_k = int(top["k"])

    return df, rec_k, recommendation


def _write_html(fig: go.Figure, out_path: Path, title: str) -> None:
    fig.update_layout(template="plotly_white", title=title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), full_html=True, include_plotlyjs="cdn")


def _save_sweep_dashboard(df: pd.DataFrame, recommended_k: int, out_path: Path) -> None:
    plot_df = df.copy()
    plot_df["k"] = pd.to_numeric(plot_df["k"], errors="coerce")
    plot_df = plot_df.dropna(subset=["k"]).sort_values("k", kind="mergesort")
    ks = plot_df["k"].astype(int).to_numpy()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Weighted Score vs k",
            "Silhouette vs k",
            "Representative PSNR vs k",
            "Representative Count vs k",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    if "weighted_score" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=pd.to_numeric(plot_df["weighted_score"], errors="coerce"),
                mode="lines+markers",
                name="weighted_score",
                marker=dict(color="#1f77b4"),
            ),
            row=1,
            col=1,
        )

    if "silhouette" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=pd.to_numeric(plot_df["silhouette"], errors="coerce"),
                mode="lines+markers",
                name="silhouette",
                marker=dict(color="#ff7f0e"),
            ),
            row=1,
            col=2,
        )

    if "fi_psnr_mean" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=pd.to_numeric(plot_df["fi_psnr_mean"], errors="coerce"),
                mode="lines+markers",
                name="FI PSNR",
                marker=dict(color="#2ca02c"),
            ),
            row=2,
            col=1,
        )
    if "oc_psnr_mean" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=pd.to_numeric(plot_df["oc_psnr_mean"], errors="coerce"),
                mode="lines+markers",
                name="OC PSNR",
                marker=dict(color="#d62728"),
            ),
            row=2,
            col=1,
        )

    if "num_representatives" in plot_df.columns:
        reps = pd.to_numeric(plot_df["num_representatives"], errors="coerce").fillna(0).astype(int)
        colors = ["#e15759" if int(k) == int(recommended_k) else "#4e79a7" for k in ks]
        fig.add_trace(
            go.Bar(x=ks, y=reps, marker=dict(color=colors), name="# representatives"),
            row=2,
            col=2,
        )

    for (r, c) in ((1, 1), (1, 2), (2, 1), (2, 2)):
        fig.add_vline(
            x=int(recommended_k),
            line_dash="dash",
            line_color="black",
            opacity=0.6,
            row=r,
            col=c,
        )

    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_xaxes(title_text="k", row=2, col=1)
    fig.update_xaxes(title_text="k", row=2, col=2)
    fig.update_yaxes(title_text="weighted_score", row=1, col=1)
    fig.update_yaxes(title_text="silhouette", row=1, col=2)
    fig.update_yaxes(title_text="PSNR", row=2, col=1)
    fig.update_yaxes(title_text="# representatives", row=2, col=2)
    fig.update_layout(height=800, legend=dict(orientation="h"))

    _write_html(fig, out_path, f"K Sweep Dashboard (recommended k={recommended_k})")


def _save_cluster_histogram(hist_df: pd.DataFrame, selected_k: int, out_path: Path) -> None:
    df = hist_df.copy()
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df["num_scenes"] = pd.to_numeric(df["num_scenes"], errors="coerce")
    df = df.dropna(subset=["cluster", "num_scenes"]).sort_values("cluster", kind="mergesort")
    df["cluster"] = df["cluster"].astype(int)

    hover_data = np.where(
        "percent" in df.columns,
        pd.to_numeric(df["percent"], errors="coerce").round(3).astype(str) + "%",
        "",
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["cluster"],
                y=df["num_scenes"],
                marker=dict(color="#4e79a7"),
                hovertemplate="cluster=%{x}<br>num_scenes=%{y}<br>percent=%{customdata}<extra></extra>",
                customdata=hover_data,
            )
        ]
    )
    fig.update_xaxes(title_text="cluster")
    fig.update_yaxes(title_text="# scenes")
    _write_html(fig, out_path, f"Cluster Size Distribution (k={selected_k})")


def _save_rep_metric_chart(metrics_df: pd.DataFrame, selected_k: int, out_path: Path) -> None:
    df = metrics_df.copy()
    df["csv"] = df["csv"].astype(str).str.lower().str.strip()
    df["metric"] = df["metric"].astype(str).str.strip()
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PSNR", "SSIM"), horizontal_spacing=0.12)
    for idx, metric in enumerate(("PSNR", "SSIM"), start=1):
        sub = df[df["metric"] == metric].sort_values("csv", kind="mergesort")
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["csv"],
                y=sub["mean"],
                error_y=dict(type="data", array=sub["std"].fillna(0.0), visible=True),
                name=metric,
                showlegend=False,
                marker=dict(color="#59a14f" if metric == "PSNR" else "#f28e2b"),
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text="csv", row=1, col=idx)
        fig.update_yaxes(title_text="mean +/- std", row=1, col=idx)

    fig.update_layout(height=420)
    _write_html(fig, out_path, f"Representative Metrics (k={selected_k})")


def _save_cluster_profile_heatmap(
    profiles_df: pd.DataFrame, selected_k: int, top_n_features: int, out_path: Path
) -> None:
    df = profiles_df.copy()
    if "cluster" not in df.columns:
        raise RuntimeError("cluster_profiles CSV is missing 'cluster' column.")
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df = df.dropna(subset=["cluster"]).sort_values("cluster", kind="mergesort")

    numeric_cols: List[str] = []
    for c in df.columns:
        if c == "cluster":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)

    if not numeric_cols:
        raise RuntimeError("No numeric feature columns found in cluster_profiles CSV.")

    num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    feat_var = num.std(axis=0, ddof=0).sort_values(ascending=False)
    selected_feats = feat_var.head(max(1, int(top_n_features))).index.tolist()
    mat = num[selected_feats].to_numpy(dtype=float)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=mat,
                x=selected_feats,
                y=df["cluster"].astype(int).astype(str),
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="profile value"),
            )
        ]
    )
    fig.update_xaxes(title_text="feature")
    fig.update_yaxes(title_text="cluster")
    _write_html(
        fig,
        out_path,
        f"Cluster Profile Heatmap (k={selected_k}, top {len(selected_feats)} varying features)",
    )


def _save_subset_coverage(
    mapping_df: pd.DataFrame, manifest_df: pd.DataFrame, selected_k: int, out_path: Path
) -> None:
    full_map = mapping_df.copy()
    full_map["dish_id"] = full_map["dish_id"].astype(str)
    full_map["cluster"] = pd.to_numeric(full_map["cluster"], errors="coerce")
    full_map = full_map.dropna(subset=["cluster"])
    full_map["cluster"] = full_map["cluster"].astype(int)

    sub = manifest_df.copy()
    if "dish_id" not in sub.columns:
        raise RuntimeError("Subset manifest must contain a 'dish_id' column.")
    sub["dish_id"] = sub["dish_id"].astype(str)

    if "cluster" not in sub.columns:
        sub = sub.merge(full_map[["dish_id", "cluster"]], on="dish_id", how="left")
    sub["cluster"] = pd.to_numeric(sub["cluster"], errors="coerce")
    sub = sub.dropna(subset=["cluster"])
    sub["cluster"] = sub["cluster"].astype(int)

    full_counts = full_map.groupby("cluster").size().rename("total")
    sub_counts = sub.groupby("cluster").size().rename("selected")
    merged = pd.concat([full_counts, sub_counts], axis=1).fillna(0).reset_index()
    merged["total"] = merged["total"].astype(int)
    merged["selected"] = merged["selected"].astype(int)
    merged = merged.sort_values("cluster", kind="mergesort")

    coverage_pct = np.where(
        merged["total"].to_numpy() > 0,
        100.0 * merged["selected"].to_numpy() / merged["total"].to_numpy(),
        0.0,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Absolute counts", "Coverage percent"),
    )
    fig.add_trace(
        go.Bar(x=merged["cluster"], y=merged["total"], name="total", marker=dict(color="#bab0ab")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=merged["cluster"], y=merged["selected"], name="selected", marker=dict(color="#4e79a7")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=merged["cluster"], y=coverage_pct, name="coverage %", marker=dict(color="#f28e2b")),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="cluster", row=2, col=1)
    fig.update_yaxes(title_text="# scenes", row=1, col=1)
    fig.update_yaxes(title_text="coverage %", row=2, col=1)
    fig.update_layout(height=700)

    _write_html(fig, out_path, f"Subset Coverage by Cluster (k={selected_k})")


def _write_markdown(
    path: Path,
    selected_row: pd.Series,
    recommendation: Dict[str, object],
    figures: List[Tuple[str, Path]],
) -> None:
    lines: List[str] = [
        "# Cluster Sweep Visual Report",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Selected k: **{int(selected_row['k'])}**",
    ]
    if recommendation:
        lines += [
            f"- Recommended k (from JSON): **{recommendation.get('recommended_k')}**",
            f"- Budget max representatives: **{recommendation.get('budget_max_representatives')}**",
            f"- Budget applied: **{recommendation.get('budget_applied')}**",
        ]

    for field in (
        "weighted_score",
        "silhouette",
        "cluster_balance_min_over_max",
        "cluster_entropy",
        "num_representatives",
        "oc_psnr_mean",
        "fi_psnr_mean",
    ):
        if field in selected_row.index and pd.notna(selected_row[field]):
            val = selected_row[field]
            if isinstance(val, (float, np.floating)):
                lines.append(f"- {field}: `{float(val):.6f}`")
            else:
                lines.append(f"- {field}: `{val}`")

    lines += ["", "## Figures", ""]
    for title, p in figures:
        lines.append(f"- [{title}]({p.name})")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual summaries for k-sweep outputs.")
    parser.add_argument("--sweep-dir", type=Path, default=Path("sweep_cluster_k"))
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--selected-k", type=int, default=None, help="Override recommended k.")
    parser.add_argument("--top-features", type=int, default=15, help="Top varying profile features for heatmap.")
    parser.add_argument("--subset-manifest", type=Path, default=None, help="Optional subset manifest CSV with dish_id.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    sweep_dir = args.sweep_dir
    summary_csv = args.summary_csv or (sweep_dir / "k_sweep_summary.csv")
    summary_json = args.summary_json or (sweep_dir / "k_sweep_summary.json")
    out_dir = args.output_dir or (sweep_dir / "visuals")
    out_dir.mkdir(parents=True, exist_ok=True)

    df, rec_k, recommendation = _load_summary(summary_csv, summary_json)
    selected_k = int(args.selected_k) if args.selected_k is not None else int(rec_k)

    if "k" not in df.columns:
        raise RuntimeError("Summary CSV must contain 'k' column.")
    row_df = df[pd.to_numeric(df["k"], errors="coerce") == selected_k]
    if row_df.empty:
        raise RuntimeError(f"Selected k={selected_k} not found in summary CSV.")
    selected_row = row_df.iloc[0]

    run_dir = _resolve_run_dir(str(selected_row["run_dir"]), sweep_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir does not exist: {run_dir}")
    LOGGER.info("Using run dir for k=%d: %s", selected_k, run_dir)

    figures: List[Tuple[str, Path]] = []

    p_dashboard = out_dir / "k_sweep_dashboard.html"
    _save_sweep_dashboard(df, selected_k, p_dashboard)
    figures.append(("K Sweep Dashboard", p_dashboard))
    LOGGER.info("Saved: %s", p_dashboard)

    hist_path = run_dir / "cluster_size_histogram.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        p_hist = out_dir / f"k{selected_k}_cluster_histogram.html"
        _save_cluster_histogram(hist_df, selected_k, p_hist)
        figures.append((f"Cluster Size Distribution (k={selected_k})", p_hist))
        LOGGER.info("Saved: %s", p_hist)
    else:
        LOGGER.warning("Missing histogram CSV: %s", hist_path)

    metrics_path = _first_existing(run_dir, "*_reps_ingp_metrics_summary.csv")
    if metrics_path is not None:
        metrics_df = pd.read_csv(metrics_path)
        p_metrics = out_dir / f"k{selected_k}_rep_metrics.html"
        _save_rep_metric_chart(metrics_df, selected_k, p_metrics)
        figures.append((f"Representative Metrics (k={selected_k})", p_metrics))
        LOGGER.info("Saved: %s", p_metrics)
    else:
        LOGGER.warning("Missing reps metrics CSV under: %s", run_dir)

    profiles_path = _first_existing(run_dir, "*_cluster_profiles.csv")
    if profiles_path is not None:
        profiles_df = pd.read_csv(profiles_path)
        p_heat = out_dir / f"k{selected_k}_cluster_profile_heatmap.html"
        _save_cluster_profile_heatmap(profiles_df, selected_k, args.top_features, p_heat)
        figures.append((f"Cluster Profile Heatmap (k={selected_k})", p_heat))
        LOGGER.info("Saved: %s", p_heat)
    else:
        LOGGER.warning("Missing cluster profiles CSV under: %s", run_dir)

    if args.subset_manifest is not None and args.subset_manifest.exists():
        mapping_path = _first_existing(run_dir, "*_dish_cluster_mapping.csv")
        if mapping_path is None:
            LOGGER.warning("Skipping subset coverage: mapping CSV not found under %s", run_dir)
        else:
            mapping_df = pd.read_csv(mapping_path)
            manifest_df = pd.read_csv(args.subset_manifest)
            p_cov = out_dir / f"k{selected_k}_subset_coverage.html"
            _save_subset_coverage(mapping_df, manifest_df, selected_k, p_cov)
            figures.append((f"Subset Coverage (k={selected_k})", p_cov))
            LOGGER.info("Saved: %s", p_cov)
    elif args.subset_manifest is not None:
        LOGGER.warning("Subset manifest not found: %s", args.subset_manifest)

    md_path = out_dir / "visual_report.md"
    _write_markdown(md_path, selected_row, recommendation, figures)
    LOGGER.info("Saved: %s", md_path)


if __name__ == "__main__":
    main()
