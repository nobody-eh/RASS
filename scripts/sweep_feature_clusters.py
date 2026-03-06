#!/usr/bin/env python3
"""Sweep clustering k values and summarize subset-selection tradeoffs.

This script runs `scripts/run_feature_analysis_pipeline.py` for each requested
cluster count and aggregates the outputs into one comparison table.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("sweep_feature_clusters")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_runtime_env() -> None:
    if "KMP_USE_SHM" not in os.environ:
        os.environ["KMP_USE_SHM"] = "0"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if k not in os.environ:
            os.environ[k] = "1"


def _parse_k_values(value: str) -> List[int]:
    out: List[int] = []
    for tok in value.split(","):
        s = tok.strip()
        if not s:
            continue
        k = int(s)
        if k < 2:
            raise ValueError(f"Invalid k={k}. k must be >= 2.")
        out.append(k)
    if not out:
        raise ValueError("No valid k values provided.")
    return sorted(set(out))


def _safe_metric(df: pd.DataFrame, csv_name: str, metric: str) -> float:
    if df.empty:
        return float("nan")
    mask = (df["csv"] == csv_name) & (df["metric"] == metric)
    if not mask.any():
        return float("nan")
    v = pd.to_numeric(df.loc[mask, "mean"], errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return float(v.iloc[0])


def _cluster_entropy_from_hist(hist_df: pd.DataFrame) -> float:
    if hist_df.empty:
        return float("nan")
    p = pd.to_numeric(hist_df["percent"], errors="coerce").to_numpy(dtype=float) / 100.0
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    h = -np.sum(p * np.log(p))
    h_max = math.log(max(2, p.size))
    return float(h / h_max) if h_max > 0 else float("nan")


def _compute_silhouette(normalized_csv: Path, clustered_features_csv: Path) -> float:
    if not normalized_csv.exists() or not clustered_features_csv.exists():
        return float("nan")

    code = r"""
import sys
import pandas as pd
from sklearn.metrics import silhouette_score

norm_csv = sys.argv[1]
map_csv = sys.argv[2]

df_norm = pd.read_csv(norm_csv)
df_map = pd.read_csv(map_csv)
if "dish_id" not in df_norm.columns or "dish_id" not in df_map.columns or "cluster" not in df_map.columns:
    print("nan")
    raise SystemExit(0)

merged = df_norm.merge(df_map[["dish_id", "cluster"]], on="dish_id", how="inner")
if merged.empty:
    print("nan")
    raise SystemExit(0)

labels = pd.to_numeric(merged["cluster"], errors="coerce").dropna()
if labels.nunique() < 2:
    print("nan")
    raise SystemExit(0)

feat_cols = [c for c in merged.columns if c not in ("dish_id", "cluster")]
X = merged[feat_cols].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean(axis=0))
y = pd.to_numeric(merged["cluster"], errors="coerce").astype(int).to_numpy()
print(float(silhouette_score(X.to_numpy(), y)))
"""
    env = os.environ.copy()
    env.setdefault("KMP_USE_SHM", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    proc = subprocess.run(
        [sys.executable, "-c", code, str(normalized_csv), str(clustered_features_csv)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        LOGGER.warning(
            "Silhouette computation failed for %s (code=%d): %s",
            clustered_features_csv,
            proc.returncode,
            (proc.stderr or "").strip(),
        )
        return float("nan")
    txt = (proc.stdout or "").strip()
    try:
        return float(txt)
    except Exception:
        return float("nan")


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    LOGGER.info("Running: %s", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_markdown(path: Path, rows: List[Dict[str, Any]], recommendation: Dict[str, Any]) -> None:
    lines = [
        "# K Sweep Summary",
        "",
        f"- Recommended k: **{recommendation['recommended_k']}**",
        f"- Budget max representatives: **{recommendation['budget_max_representatives']}**",
        f"- Budget applied: **{recommendation['budget_applied']}**",
        f"- Recommended weighted score: **{float(recommendation['recommended_weighted_score']):.6f}**",
        "",
        "| rank_w | rank_s | k | weighted_score | silhouette | min_cluster | max_cluster | balance | entropy | reps | oc_psnr | fi_psnr |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {rank_weighted} | {rank_silhouette} | {k} | {weighted_score:.6f} | "
            "{silhouette:.6f} | {min_cluster_size} | {max_cluster_size} | "
            "{cluster_balance_min_over_max:.4f} | {cluster_entropy:.4f} | "
            "{num_representatives} | {oc_psnr_mean:.4f} | {fi_psnr_mean:.4f} |".format(
                **r
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _norm01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx):
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    if float(mx) - float(mn) <= 1e-12:
        return pd.Series(np.ones(len(s), dtype=float), index=s.index)
    return (s - mn) / (mx - mn)


def _apply_weighted_scoring(
    df: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    out["silhouette_norm"] = _norm01(out["silhouette"])
    out["balance_norm"] = _norm01(out["cluster_balance_min_over_max"])
    out["entropy_norm"] = _norm01(out["cluster_entropy"])
    out["fi_psnr_norm"] = _norm01(out["fi_psnr_mean"])
    out["oc_psnr_norm"] = _norm01(out["oc_psnr_mean"])
    out["compactness_norm"] = 1.0 - _norm01(out["num_representatives"])

    weights = {
        "silhouette_norm": float(args.weight_silhouette),
        "balance_norm": float(args.weight_balance),
        "entropy_norm": float(args.weight_entropy),
        "fi_psnr_norm": float(args.weight_fi_psnr),
        "oc_psnr_norm": float(args.weight_oc_psnr),
        "compactness_norm": float(args.weight_compactness),
    }
    weight_sum = sum(max(0.0, w) for w in weights.values())
    if weight_sum <= 0.0:
        raise ValueError("At least one weight must be > 0.")

    score = np.zeros(len(out), dtype=float)
    for col, w in weights.items():
        if w <= 0:
            continue
        score += w * out[col].to_numpy(dtype=float)
    out["weighted_score"] = score / weight_sum

    out = out.sort_values(
        ["silhouette", "cluster_balance_min_over_max", "fi_psnr_mean", "oc_psnr_mean"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    out["rank_silhouette"] = np.arange(1, len(out) + 1)

    out = out.sort_values(
        ["weighted_score", "silhouette", "cluster_balance_min_over_max", "fi_psnr_mean"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    out["rank_weighted"] = np.arange(1, len(out) + 1)

    eligible = out
    budget_applied = False
    if args.budget_max_representatives is not None:
        budget_applied = True
        eligible = out[out["num_representatives"] <= int(args.budget_max_representatives)]
        if eligible.empty:
            eligible = out
            budget_applied = False

    recommended = eligible.iloc[0].to_dict()
    details = {
        "recommended_k": int(recommended["k"]),
        "recommended_num_representatives": int(recommended["num_representatives"]),
        "recommended_weighted_score": float(recommended["weighted_score"]),
        "budget_max_representatives": args.budget_max_representatives,
        "budget_applied": budget_applied,
        "weights": {
            "silhouette": float(args.weight_silhouette),
            "balance": float(args.weight_balance),
            "entropy": float(args.weight_entropy),
            "fi_psnr": float(args.weight_fi_psnr),
            "oc_psnr": float(args.weight_oc_psnr),
            "compactness": float(args.weight_compactness),
        },
    }
    return out, details


def _build_command(args: argparse.Namespace, k: int, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = run_dir / f"{args.prefix_base}_k{k}"

    artifacts = {
        "normalized_csv": run_dir / "feats_normalized.csv",
        "clustered_features_csv": run_dir / "clustered_feats2.csv",
        "cluster_hist_csv": run_dir / "cluster_size_histogram.csv",
        "analysis_report_json": run_dir / "analysis_report.json",
        "analysis_report_md": run_dir / "analysis_report.md",
        "reps_metrics_csv": Path(f"{output_prefix}_reps_ingp_metrics_summary.csv"),
    }

    script = _repo_root() / "scripts" / "run_feature_analysis_pipeline.py"
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--input-csv",
        str(args.input_csv),
        "--normalized-csv",
        str(artifacts["normalized_csv"]),
        "--output-prefix",
        str(output_prefix),
        "--clustered-features-csv",
        str(artifacts["clustered_features_csv"]),
        "--cluster-hist-csv",
        str(artifacts["cluster_hist_csv"]),
        "--analysis-report-json",
        str(artifacts["analysis_report_json"]),
        "--analysis-report-md",
        str(artifacts["analysis_report_md"]),
        "--method",
        args.method,
        "--n-clusters",
        str(k),
        "--n-neighbors",
        str(args.n_neighbors),
        "--min-dist",
        str(args.min_dist),
        "--rep-method",
        args.rep_method,
        "--n-per-cluster",
        str(args.n_per_cluster),
        "--oc-csv",
        str(args.oc_csv),
        "--fi-csv",
        str(args.fi_csv),
        "--random-seed",
        str(args.random_seed),
        "--log-level",
        args.child_log_level,
    ]
    if args.cluster_drop_cols:
        cmd += ["--cluster-drop-cols"] + list(args.cluster_drop_cols)
    if args.disable_static_png:
        cmd += ["--disable-static-png"]
    if args.include_images:
        cmd += ["--include-images"]
        if args.base_images_dir:
            cmd += ["--base-images-dir", str(args.base_images_dir)]
    return {"cmd": cmd, "artifacts": artifacts}


def run_sweep(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    root = _repo_root()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    k_values = _parse_k_values(args.k_values)
    LOGGER.info("Sweeping k values: %s", k_values)

    for k in k_values:
        run_dir = out_dir / f"k_{k}"
        prepared = _build_command(args, k, run_dir)
        cmd = prepared["cmd"]
        artifacts = prepared["artifacts"]

        can_skip = (
            args.skip_existing
            and artifacts["analysis_report_json"].exists()
            and artifacts["clustered_features_csv"].exists()
            and artifacts["normalized_csv"].exists()
            and artifacts["reps_metrics_csv"].exists()
        )
        if can_skip:
            LOGGER.info("Skipping k=%d (artifacts already exist)", k)
        else:
            _run_cmd(cmd, cwd=root)

        with artifacts["analysis_report_json"].open("r", encoding="utf-8") as f:
            report = json.load(f)

        hist_df = pd.read_csv(artifacts["cluster_hist_csv"])
        reps_df = pd.read_csv(artifacts["reps_metrics_csv"])
        silhouette = _compute_silhouette(
            artifacts["normalized_csv"],
            artifacts["clustered_features_csv"],
        )

        min_cluster = int(hist_df["num_scenes"].min()) if not hist_df.empty else 0
        max_cluster = int(hist_df["num_scenes"].max()) if not hist_df.empty else 0
        balance = (
            float(min_cluster / max_cluster) if max_cluster > 0 else float("nan")
        )

        row = {
            "k": int(k),
            "silhouette": float(silhouette),
            "min_cluster_size": int(min_cluster),
            "max_cluster_size": int(max_cluster),
            "cluster_balance_min_over_max": float(balance),
            "cluster_entropy": float(_cluster_entropy_from_hist(hist_df)),
            "num_representatives": int(report["representatives"]["num_representatives"]),
            "oc_psnr_mean": _safe_metric(reps_df, "oc", "PSNR"),
            "oc_ssim_mean": _safe_metric(reps_df, "oc", "SSIM"),
            "fi_psnr_mean": _safe_metric(reps_df, "fi", "PSNR"),
            "fi_ssim_mean": _safe_metric(reps_df, "fi", "SSIM"),
            "run_dir": str(run_dir),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No sweep results produced.")

    df, recommendation = _apply_weighted_scoring(pd.DataFrame(rows), args)
    cols = [
        "rank_weighted",
        "rank_silhouette",
        "k",
        "weighted_score",
        "silhouette",
        "min_cluster_size",
        "max_cluster_size",
        "cluster_balance_min_over_max",
        "cluster_entropy",
        "num_representatives",
        "oc_psnr_mean",
        "oc_ssim_mean",
        "fi_psnr_mean",
        "fi_ssim_mean",
        "run_dir",
    ]
    return df[cols], recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep clustering k values and summarize results."
    )
    parser.add_argument("--input-csv", type=Path, default=Path("feats2.csv"))
    parser.add_argument("--oc-csv", type=Path, default=Path("ingp_oc.csv"))
    parser.add_argument("--fi-csv", type=Path, default=Path("ingp_fi.csv"))
    parser.add_argument("--k-values", type=str, default="4,6,8,10")
    parser.add_argument("--output-dir", type=Path, default=Path("sweep_cluster_k"))
    parser.add_argument("--prefix-base", type=str, default="clustered_scenes")

    parser.add_argument("--method", choices=["kmeans", "agglomerative"], default="kmeans")
    parser.add_argument("--cluster-drop-cols", nargs="*", default=None)
    parser.add_argument("--rep-method", choices=["centroid", "medoid"], default="centroid")
    parser.add_argument("--n-per-cluster", type=int, default=2)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=0)

    parser.add_argument(
        "--disable-static-png",
        dest="disable_static_png",
        action="store_true",
        default=True,
        help="Disable per-run static PNG generation (faster, default).",
    )
    parser.add_argument(
        "--enable-static-png",
        dest="disable_static_png",
        action="store_false",
        help="Enable per-run static PNG generation.",
    )
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--base-images-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--summary-csv-name", type=str, default="k_sweep_summary.csv")
    parser.add_argument("--summary-json-name", type=str, default="k_sweep_summary.json")
    parser.add_argument("--summary-md-name", type=str, default="k_sweep_summary.md")
    parser.add_argument(
        "--budget-max-representatives",
        type=int,
        default=None,
        help="If set, recommend k only among rows with num_representatives <= budget.",
    )
    parser.add_argument("--weight-silhouette", type=float, default=0.35)
    parser.add_argument("--weight-balance", type=float, default=0.20)
    parser.add_argument("--weight-entropy", type=float, default=0.10)
    parser.add_argument("--weight-fi-psnr", type=float, default=0.20)
    parser.add_argument("--weight-oc-psnr", type=float, default=0.10)
    parser.add_argument(
        "--weight-compactness",
        type=float,
        default=0.05,
        help="Higher value favors fewer representatives.",
    )
    parser.add_argument("--child-log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_runtime_env()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    df, recommendation = run_sweep(args)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / args.summary_csv_name
    summary_json = out_dir / args.summary_json_name
    summary_md = out_dir / args.summary_md_name

    df.to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as f:
        payload = {
            "recommendation": {k: _to_jsonable(v) for k, v in recommendation.items()},
            "rows": [{k: _to_jsonable(v) for k, v in r.items()} for r in df.to_dict(orient="records")],
        }
        json.dump(payload, f, indent=2, sort_keys=False)
    _write_markdown(summary_md, df.to_dict(orient="records"), recommendation)

    best = df.iloc[0].to_dict()
    LOGGER.info("Saved summary CSV: %s", summary_csv)
    LOGGER.info("Saved summary JSON: %s", summary_json)
    LOGGER.info("Saved summary MD: %s", summary_md)
    LOGGER.info(
        "Top weighted row k=%s | weighted_score=%.6f | silhouette=%.6f | reps=%d",
        best["k"],
        float(best["weighted_score"]),
        float(best["silhouette"]),
        int(best["num_representatives"]),
    )
    LOGGER.info(
        "Recommended k=%s (budget<=%s, applied=%s)",
        recommendation["recommended_k"],
        recommendation["budget_max_representatives"],
        recommendation["budget_applied"],
    )


if __name__ == "__main__":
    main()
