#!/usr/bin/env python3
"""Reproducible end-to-end feature analysis pipeline.

Pipeline:
1) Load raw feature CSV (default: feats2.csv)
2) Normalize features (log/robust/standard/clip) -> feats_normalized.csv
3) Cluster normalized features + UMAP outputs
4) Select representative scenes
5) Summarize OC/FI NeRF metrics on representatives with robust column handling

This script is deterministic (sorted outputs + fixed random state).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("feature_analysis_pipeline")


def _setup_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_setup_import_path()


def _configure_runtime_env() -> None:
    # Some conda builds of numba/umap need an explicit writable cache dir.
    if "NUMBA_CACHE_DIR" not in os.environ:
        cache_dir = Path("/tmp/numba_cache")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
            LOGGER.info("Using NUMBA_CACHE_DIR=%s", cache_dir)
        except Exception as exc:
            LOGGER.warning("Could not set NUMBA_CACHE_DIR: %s", exc)
    # Prevent OpenMP shared-memory failures on some hosts.
    if "KMP_USE_SHM" not in os.environ:
        os.environ["KMP_USE_SHM"] = "0"


METRIC_COLUMNS = [
    "PSNR",
    "PSNR_MIN",
    "PSNR_MAX",
    "SSIM",
    "SSIM_MIN",
    "SSIM_MAX",
    "psnr_avgmse",
]


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def _canonicalize_metric_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Map heterogeneous metric column names to a canonical schema."""
    if df.empty:
        raise ValueError(f"{source_name} metrics CSV is empty")

    orig_cols = list(df.columns)
    norm_to_orig: Dict[str, str] = {}
    for c in orig_cols:
        n = _normalize_col_name(c)
        # keep first occurrence deterministically
        if n not in norm_to_orig:
            norm_to_orig[n] = c

    def pick_first(candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            if c in norm_to_orig:
                return norm_to_orig[c]
        return None

    # dish_id column
    dish_col = pick_first(("dishid", "sceneid", "id"))
    if dish_col is None:
        raise ValueError(
            f"{source_name}: unable to find dish identifier column in {orig_cols}"
        )

    # Canonical mapping with robust fallbacks for known files:
    # - OC: PSNR, PSNR_MIN, PSNR_MAX, SSIM, SSIM_MIN, SSIM_MAX, psnr_avgmse
    # - FI: PNSR, MIN, MAX, SSIM, MIN.1, MAX.1, psnr avgmse
    psnr_col = pick_first(("psnr", "pnsr"))
    ssim_col = pick_first(("ssim",))
    psnr_avgmse_col = pick_first(("psnravgmse",))

    psnr_min_col = pick_first(("psnrmin",))
    psnr_max_col = pick_first(("psnrmax",))
    ssim_min_col = pick_first(("ssimmin",))
    ssim_max_col = pick_first(("ssimmax",))

    # Fallbacks for FI-style MIN/MAX columns.
    min_col = pick_first(("min",))
    max_col = pick_first(("max",))
    min1_col = pick_first(("min1",))
    max1_col = pick_first(("max1",))

    if psnr_min_col is None and min_col is not None:
        psnr_min_col = min_col
    if psnr_max_col is None and max_col is not None:
        psnr_max_col = max_col
    if ssim_min_col is None and min1_col is not None:
        ssim_min_col = min1_col
    if ssim_max_col is None and max1_col is not None:
        ssim_max_col = max1_col

    mapping = {
        "dish_id": dish_col,
        "PSNR": psnr_col,
        "PSNR_MIN": psnr_min_col,
        "PSNR_MAX": psnr_max_col,
        "SSIM": ssim_col,
        "SSIM_MIN": ssim_min_col,
        "SSIM_MAX": ssim_max_col,
        "psnr_avgmse": psnr_avgmse_col,
    }

    out = pd.DataFrame()
    out["dish_id"] = df[mapping["dish_id"]].astype(str)
    for metric in METRIC_COLUMNS:
        src_col = mapping[metric]
        if src_col is None:
            LOGGER.warning(
                "%s: metric column not found for canonical field %s",
                source_name,
                metric,
            )
            out[metric] = np.nan
        else:
            out[metric] = pd.to_numeric(df[src_col], errors="coerce")
    return out


def _compute_rep_metric_summary(
    reps: List[str], metric_df: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    rep_set = set(map(str, reps))
    df_sub = metric_df[metric_df["dish_id"].astype(str).isin(rep_set)].copy()
    if df_sub.empty:
        LOGGER.warning("%s: no representative dishes matched metrics table", source_name)
        return pd.DataFrame(columns=["csv", "metric", "mean", "std", "count"])

    rows: List[Dict[str, object]] = []
    for metric in METRIC_COLUMNS:
        s = pd.to_numeric(df_sub[metric], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "csv": source_name,
                "metric": metric,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "count": int(s.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _save_representatives(path: Path, reps: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rid in reps:
            f.write(f"{rid}\n")


def _resolve_optional_path(path_value: Optional[str], default_path: Path) -> Path:
    if path_value is None or str(path_value).strip() == "":
        return default_path
    return Path(path_value)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cluster_histogram(mapping_df: pd.DataFrame) -> pd.DataFrame:
    hist = mapping_df.groupby("cluster", dropna=False).size().reset_index(name="num_scenes")
    hist = hist.sort_values("cluster", kind="mergesort").reset_index(drop=True)
    total = int(hist["num_scenes"].sum())
    hist["percent"] = (
        (100.0 * hist["num_scenes"] / total) if total > 0 else 0.0
    )
    return hist


def _write_analysis_markdown(path: Path, payload: Dict[str, Any]) -> None:
    overview = payload["overview"]
    cluster_hist = payload["cluster_histogram"]
    rep_summary = payload["representatives"]
    metric_rows = payload["representative_metric_summary"]

    lines = [
        "# Feature Analysis Report",
        "",
        "## Overview",
        f"- Scenes: **{overview['num_scenes']}**",
        f"- Features: **{overview['num_features']}**",
        f"- Clustering method: **{overview['cluster_method']}**",
        f"- Number of clusters: **{overview['n_clusters']}**",
        "",
        "## Normalization",
        f"- Log1p features: **{overview['num_log_features']}**",
        f"- Robust-scaled features: **{overview['num_robust_features']}**",
        f"- Standard-scaled features: **{overview['num_standard_features']}**",
        "",
        "## Cluster Histogram",
        "",
        "| cluster | num_scenes | percent |",
        "|---:|---:|---:|",
    ]
    for row in cluster_hist:
        lines.append(
            f"| {row['cluster']} | {row['num_scenes']} | {float(row['percent']):.2f} |"
        )

    lines += [
        "",
        "## Representatives",
        f"- Total representatives: **{rep_summary['num_representatives']}**",
        f"- Method: **{rep_summary['method']}**",
        f"- Per-cluster target: **{rep_summary['n_per_cluster']}**",
        "",
        "| cluster | rep_count |",
        "|---:|---:|",
    ]
    for row in rep_summary["counts_per_cluster"]:
        lines.append(f"| {row['cluster']} | {row['rep_count']} |")

    lines += [
        "",
        "## Representative Metrics (OC/FI)",
        "",
        "| csv | metric | mean | std | count |",
        "|---|---|---:|---:|---:|",
    ]
    if metric_rows:
        for row in metric_rows:
            lines.append(
                f"| {row['csv']} | {row['metric']} | {float(row['mean']):.6f} | {float(row['std']):.6f} | {int(row['count'])} |"
            )
    else:
        lines.append("| - | - | - | - | - |")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_runtime_env()

    try:
        import feats_norm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for normalization pipeline. "
            "Ensure project Python deps are installed."
        ) from exc

    try:
        import cluster as cluster_mod
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for clustering pipeline (likely umap-learn/plotly/sklearn). "
            "Install required packages before running."
        ) from exc

    np.random.seed(args.random_seed)

    raw_csv = Path(args.input_csv)
    normalized_csv = Path(args.normalized_csv)
    output_prefix = args.output_prefix
    cluster_hist_path = _resolve_optional_path(
        args.cluster_hist_csv, Path(f"{output_prefix}_cluster_size_histogram.csv")
    )
    analysis_report_json_path = _resolve_optional_path(
        args.analysis_report_json, Path(f"{output_prefix}_analysis_report.json")
    )
    analysis_report_md_path = _resolve_optional_path(
        args.analysis_report_md, Path(f"{output_prefix}_analysis_report.md")
    )

    if not raw_csv.exists():
        raise FileNotFoundError(f"Input features CSV not found: {raw_csv}")

    _ensure_parent(normalized_csv)
    _ensure_parent(Path(args.clustered_features_csv))
    _ensure_parent(cluster_hist_path)
    _ensure_parent(analysis_report_json_path)
    _ensure_parent(analysis_report_md_path)

    LOGGER.info("Loading raw features: %s", raw_csv)
    df_raw = pd.read_csv(raw_csv)
    if "dish_id" not in df_raw.columns:
        raise ValueError("Input CSV must contain dish_id column")

    LOGGER.info("Raw shape: %s", df_raw.shape)

    # Step 1: Normalize
    LOGGER.info("Running normalization")
    df_norm, norm_info = feats_norm.normalize_features(
        df_raw,
        drop_cols=args.drop_cols,
        id_col=args.id_col,
        skew_thresh=args.skew_thresh,
        range_ratio_thresh=args.range_ratio_thresh,
        clip_std=args.clip_std,
    )
    df_norm.to_csv(normalized_csv, index=False)
    LOGGER.info("Saved normalized features: %s", normalized_csv)
    LOGGER.info("Log-transformed features: %d", len(norm_info["log_feats"]))
    LOGGER.info("Robust-scaled features: %d", len(norm_info["robust_feats"]))
    LOGGER.info("Standard-scaled features: %d", len(norm_info["standard_feats"]))

    # Step 2: Cluster
    LOGGER.info("Running clustering")
    ids, x_scaled, _, _ = cluster_mod.preprocess_features(
        df_norm, drop_cols=args.cluster_drop_cols
    )

    if args.n_clusters is None:
        LOGGER.info("Selecting k by silhouette search [%d, %d]", args.k_min, args.k_max)
        best_k = cluster_mod.choose_clusters(x_scaled, k_min=args.k_min, k_max=args.k_max)
    else:
        best_k = args.n_clusters
    LOGGER.info("Using clusters: %d", best_k)

    labels, _ = cluster_mod.cluster_data(
        x_scaled, method=args.method, n_clusters=best_k
    )

    emb3d = cluster_mod.embed_umap_3d(
        x_scaled,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_seed,
    )

    cluster_mod.build_and_save_3d(
        df_norm,
        emb3d,
        labels,
        output_prefix,
        include_images=args.include_images,
        base_images_dir=args.base_images_dir,
        static_png=not args.disable_static_png,
    )

    mapping_df = pd.DataFrame({"dish_id": df_norm["dish_id"].astype(str), "cluster": labels})
    mapping_df = mapping_df.sort_values("dish_id", kind="mergesort").reset_index(drop=True)
    mapping_df.to_csv(args.clustered_features_csv, index=False)
    LOGGER.info("Saved clustered features mapping: %s", args.clustered_features_csv)
    cluster_hist_df = _cluster_histogram(mapping_df)
    cluster_hist_df.to_csv(cluster_hist_path, index=False)
    LOGGER.info("Saved cluster histogram: %s", cluster_hist_path)

    # Step 3: Representatives
    if args.rep_method == "centroid":
        reps = cluster_mod.pick_representatives_centroid(
            ids, x_scaled, labels, n_per_cluster=args.n_per_cluster
        )
    else:
        reps = cluster_mod.pick_representatives_medoid(ids, x_scaled, labels)
    reps = [str(x) for x in reps]

    reps_path = Path(f"{output_prefix}_representatives_{args.rep_method}.txt")
    _ensure_parent(reps_path)
    _save_representatives(reps_path, reps)
    LOGGER.info("Saved representatives: %s (count=%d)", reps_path, len(reps))

    # Step 4: OC/FI representative metric summary (with robust column canonicalization)
    oc_path = Path(args.oc_csv)
    fi_path = Path(args.fi_csv)
    if not oc_path.exists():
        raise FileNotFoundError(f"OC metrics CSV not found: {oc_path}")
    if not fi_path.exists():
        raise FileNotFoundError(f"FI metrics CSV not found: {fi_path}")

    LOGGER.info("Loading metrics tables: %s, %s", oc_path, fi_path)
    oc_df = _canonicalize_metric_columns(pd.read_csv(oc_path), "oc")
    fi_df = _canonicalize_metric_columns(pd.read_csv(fi_path), "fi")

    oc_summary = _compute_rep_metric_summary(reps, oc_df, "oc")
    fi_summary = _compute_rep_metric_summary(reps, fi_df, "fi")
    summary_df = pd.concat([oc_summary, fi_summary], ignore_index=True)
    summary_df = summary_df.sort_values(["csv", "metric"], kind="mergesort")

    reps_metrics_path = Path(f"{output_prefix}_reps_ingp_metrics_summary.csv")
    _ensure_parent(reps_metrics_path)
    summary_df.to_csv(reps_metrics_path, index=False)
    LOGGER.info("Saved representative metrics summary: %s", reps_metrics_path)

    # Save run metadata for reproducibility.
    run_meta = {
        "input_csv": str(raw_csv),
        "normalized_csv": str(normalized_csv),
        "output_prefix": output_prefix,
        "cluster_method": args.method,
        "n_clusters": int(best_k),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "rep_method": args.rep_method,
        "n_per_cluster": int(args.n_per_cluster),
        "random_seed": int(args.random_seed),
        "oc_csv": str(oc_path),
        "fi_csv": str(fi_path),
        "num_scenes": int(df_norm.shape[0]),
        "num_features": int(df_norm.shape[1] - 1),
    }
    meta_path = Path(f"{output_prefix}_run_metadata.json")
    _ensure_parent(meta_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)
    LOGGER.info("Saved run metadata: %s", meta_path)

    rep_counts_df = (
        mapping_df[mapping_df["dish_id"].isin(reps)]
        .groupby("cluster", dropna=False)
        .size()
        .reset_index(name="rep_count")
        .sort_values("cluster", kind="mergesort")
    )

    report_payload: Dict[str, Any] = {
        "overview": {
            "num_scenes": int(df_norm.shape[0]),
            "num_features": int(df_norm.shape[1] - 1),
            "cluster_method": args.method,
            "n_clusters": int(best_k),
            "num_log_features": int(len(norm_info["log_feats"])),
            "num_robust_features": int(len(norm_info["robust_feats"])),
            "num_standard_features": int(len(norm_info["standard_feats"])),
            "random_seed": int(args.random_seed),
        },
        "normalization": {
            "log_features": list(norm_info["log_feats"]),
            "robust_features": list(norm_info["robust_feats"]),
            "standard_features": list(norm_info["standard_feats"]),
        },
        "cluster_histogram": cluster_hist_df.to_dict(orient="records"),
        "representatives": {
            "method": args.rep_method,
            "n_per_cluster": int(args.n_per_cluster),
            "num_representatives": int(len(reps)),
            "dish_ids": reps,
            "counts_per_cluster": rep_counts_df.to_dict(orient="records"),
        },
        "representative_metric_summary": summary_df.to_dict(orient="records"),
        "artifacts": {
            "normalized_csv": str(normalized_csv),
            "clustered_features_csv": str(args.clustered_features_csv),
            "cluster_histogram_csv": str(cluster_hist_path),
            "representatives_txt": str(reps_path),
            "representative_metrics_csv": str(reps_metrics_path),
            "run_metadata_json": str(meta_path),
            "umap_html": f"{output_prefix}_3d_umap.html",
            "umap_static_png": None if args.disable_static_png else f"{output_prefix}_3d_static.png",
            "analysis_report_json": str(analysis_report_json_path),
            "analysis_report_md": str(analysis_report_md_path),
        },
    }

    with analysis_report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, sort_keys=True)
    LOGGER.info("Saved analysis report JSON: %s", analysis_report_json_path)

    _write_analysis_markdown(analysis_report_md_path, report_payload)
    LOGGER.info("Saved analysis report Markdown: %s", analysis_report_md_path)

    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible feature analysis pipeline (normalize + cluster + reps)."
    )
    parser.add_argument("--input-csv", default="feats2.csv", help="Raw feature CSV input.")
    parser.add_argument(
        "--normalized-csv",
        default="feats_normalized.csv",
        help="Path to write normalized features CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        default="clustered_scenes",
        help="Prefix for clustering artifacts.",
    )
    parser.add_argument(
        "--clustered-features-csv",
        default="clustered_feats2.csv",
        help="Path to write dish_id->cluster mapping.",
    )
    parser.add_argument(
        "--cluster-hist-csv",
        default=None,
        help="Path to write cluster size histogram CSV. "
        "Default: <output-prefix>_cluster_size_histogram.csv",
    )
    parser.add_argument(
        "--analysis-report-json",
        default=None,
        help="Path to write analysis report JSON. "
        "Default: <output-prefix>_analysis_report.json",
    )
    parser.add_argument(
        "--analysis-report-md",
        default=None,
        help="Path to write analysis report Markdown. "
        "Default: <output-prefix>_analysis_report.md",
    )

    # Normalization params
    parser.add_argument("--id-col", default="dish_id")
    parser.add_argument("--drop-cols", nargs="*", default=None)
    parser.add_argument("--skew-thresh", type=float, default=1.0)
    parser.add_argument("--range-ratio-thresh", type=float, default=100.0)
    parser.add_argument("--clip-std", type=float, default=3.0)

    # Clustering params
    parser.add_argument("--method", choices=["kmeans", "agglomerative"], default="kmeans")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--cluster-drop-cols", nargs="*", default=None)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--disable-static-png", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--base-images-dir", default=None)

    # Representative selection + metrics
    parser.add_argument("--rep-method", choices=["centroid", "medoid"], default="centroid")
    parser.add_argument("--n-per-cluster", type=int, default=4)
    parser.add_argument("--oc-csv", default="ingp_oc.csv")
    parser.add_argument("--fi-csv", default="ingp_fi.csv")

    parser.add_argument("--random-seed", type=int, default=0)
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
    run_pipeline(args)


if __name__ == "__main__":
    main()
