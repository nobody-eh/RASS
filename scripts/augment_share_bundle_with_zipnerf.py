#!/usr/bin/env python3
"""Augment share-bundle figure CSVs with ZipNeRF metrics.

This script:
1) Parses ZipNeRF metrics from either plain CSV or RTF-wrapped CSV text.
2) Normalizes columns to: dish_id, PSNR, SSIM.
3) Appends ZipNeRF rows to:
   - fig2_global_gaps_vs_threshold.csv
   - fig3_holdout_tune_test_gaps.csv
   - fig4_refinement_before_after.csv
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


LOGGER = logging.getLogger("augment_share_bundle_with_zipnerf")


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def _canonicalize_metrics(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{source_name}: metrics table is empty")

    orig_cols = list(df.columns)
    norm_to_orig: Dict[str, str] = {}
    for c in orig_cols:
        n = _normalize_col_name(c)
        if n not in norm_to_orig:
            norm_to_orig[n] = c

    def pick_first(candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if c in norm_to_orig:
                return norm_to_orig[c]
        return None

    dish_col = pick_first(("dishid", "sceneid", "id", "experimentname"))
    psnr_col = pick_first(("psnr", "pnsr"))
    ssim_col = pick_first(("ssim",))
    if dish_col is None or psnr_col is None or ssim_col is None:
        raise ValueError(
            f"{source_name}: required columns not found. "
            f"Need dish/psnr/ssim in {orig_cols}"
        )

    out = pd.DataFrame()
    out["dish_id"] = df[dish_col].astype(str).str.strip()
    out["PSNR"] = pd.to_numeric(df[psnr_col], errors="coerce")
    out["SSIM"] = pd.to_numeric(df[ssim_col], errors="coerce")
    out = out.dropna(subset=["dish_id", "PSNR", "SSIM"])
    out = out[out["dish_id"] != ""].copy()
    out = out.groupby("dish_id", as_index=False)[["PSNR", "SSIM"]].mean()
    return out


def _parse_rtf_csv_rows(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    data_rows: List[Dict[str, object]] = []
    for line in text.splitlines():
        if "dish_" not in line or "," not in line:
            continue
        start = line.find("dish_")
        end = line.find("\\cell", start)
        if end < 0:
            end = len(line)
        csv_line = line[start:end].strip()
        parts = [p.strip() for p in csv_line.split(",")]
        if len(parts) < 5:
            continue
        data_rows.append(
            {
                "experiment_name": parts[0],
                "method_name": parts[1],
                "psnr": parts[2],
                "ssim": parts[4],
            }
        )
    if not data_rows:
        raise ValueError(f"No dish rows parsed from RTF file: {path}")
    return pd.DataFrame(data_rows)


def load_zipnerf_metrics(path: Path) -> pd.DataFrame:
    text_head = path.read_text(encoding="utf-8", errors="ignore")[:128]
    if "{\\rtf" in text_head.lower():
        raw_df = _parse_rtf_csv_rows(path)
    else:
        raw_df = pd.read_csv(path)
    out = _canonicalize_metrics(raw_df, "zipnerf")
    return out


def load_manifest_ids(path: Path) -> Set[str]:
    df = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in df.columns}
    col = None
    for cand in ("dishid", "sceneid", "id", "experimentname"):
        if cand in norm_to_orig:
            col = norm_to_orig[cand]
            break
    if col is None:
        raise ValueError(f"Unable to find dish_id column in manifest: {path}")
    ids = set(df[col].astype(str).str.strip())
    return {x for x in ids if x}


def load_split_filter(path: Path, eval_split: str) -> Set[str]:
    df = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in df.columns}
    dish_col = None
    split_col = None
    for cand in ("dishid", "sceneid", "id", "experimentname"):
        if cand in norm_to_orig:
            dish_col = norm_to_orig[cand]
            break
    for cand in ("split", "subset", "partition"):
        if cand in norm_to_orig:
            split_col = norm_to_orig[cand]
            break
    if dish_col is None or split_col is None:
        raise ValueError(f"Split CSV missing dish_id/split columns: {path}")
    sub = df[df[split_col].astype(str).str.lower() == eval_split.lower()].copy()
    return set(sub[dish_col].astype(str).str.strip())


def summarize_gaps(
    metrics_df: pd.DataFrame,
    subset_ids: Set[str],
    source_name: str,
    psnr_thr: float,
    ssim_thr: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for metric, thr in (("PSNR", psnr_thr), ("SSIM", ssim_thr)):
        full = pd.to_numeric(metrics_df[metric], errors="coerce").dropna()
        sub = pd.to_numeric(
            metrics_df[metrics_df["dish_id"].isin(subset_ids)][metric], errors="coerce"
        ).dropna()
        if full.empty or sub.empty:
            LOGGER.warning(
                "%s/%s: empty full or subset slice (full=%d, subset=%d)",
                source_name,
                metric,
                len(full),
                len(sub),
            )
            continue
        full_mean = float(full.mean())
        subset_mean = float(sub.mean())
        diff = subset_mean - full_mean
        abs_diff = float(abs(diff))
        out.append(
            {
                "csv": source_name,
                "metric": metric,
                "full_mean": full_mean,
                "subset_mean": subset_mean,
                "diff_subset_minus_full": float(diff),
                "abs_diff": abs_diff,
                "threshold": float(thr),
                "pass": bool(abs_diff <= thr),
            }
        )
    return out


def write_fig2(
    fig2_path: Path,
    zip_metrics_all: pd.DataFrame,
    subset_ids: Set[str],
    psnr_thr: float,
    ssim_thr: float,
    source_name: str,
) -> None:
    fig2 = pd.read_csv(fig2_path)
    fig2 = fig2[fig2["csv"].astype(str).str.lower() != source_name.lower()].copy()
    rows = summarize_gaps(zip_metrics_all, subset_ids, source_name, psnr_thr, ssim_thr)
    fig2 = pd.concat([fig2, pd.DataFrame(rows)], ignore_index=True)
    fig2["csv"] = fig2["csv"].astype(str).str.lower()
    fig2["metric"] = fig2["metric"].astype(str).str.upper()
    fig2 = fig2.sort_values(["metric", "csv"], kind="mergesort").reset_index(drop=True)
    fig2.to_csv(fig2_path, index=False)
    LOGGER.info("Updated fig2 with %d %s rows: %s", len(rows), source_name, fig2_path)


def write_fig3(
    fig3_path: Path,
    zip_metrics_all: pd.DataFrame,
    split_csv: Path,
    tune_manifest: Path,
    test_manifest: Path,
    psnr_thr: float,
    ssim_thr: float,
    source_name: str,
) -> None:
    fig3 = pd.read_csv(fig3_path)
    fig3 = fig3[fig3["csv"].astype(str).str.lower() != source_name.lower()].copy()

    split_tune_ids = load_split_filter(split_csv, "tune")
    split_test_ids = load_split_filter(split_csv, "test")
    tune_df = zip_metrics_all[zip_metrics_all["dish_id"].isin(split_tune_ids)].copy()
    test_df = zip_metrics_all[zip_metrics_all["dish_id"].isin(split_test_ids)].copy()

    tune_rows = summarize_gaps(
        tune_df, load_manifest_ids(tune_manifest), source_name, psnr_thr, ssim_thr
    )
    test_rows = summarize_gaps(
        test_df, load_manifest_ids(test_manifest), source_name, psnr_thr, ssim_thr
    )

    out_rows: List[Dict[str, object]] = []
    for split_name, rows in (("tune", tune_rows), ("test", test_rows)):
        for r in rows:
            out_rows.append(
                {
                    "split": split_name,
                    "csv": r["csv"],
                    "metric": r["metric"],
                    "diff_subset_minus_full": r["diff_subset_minus_full"],
                    "abs_diff": r["abs_diff"],
                    "threshold": r["threshold"],
                    "pass": r["pass"],
                }
            )
    fig3 = pd.concat([fig3, pd.DataFrame(out_rows)], ignore_index=True)
    fig3["csv"] = fig3["csv"].astype(str).str.lower()
    fig3["metric"] = fig3["metric"].astype(str).str.upper()
    fig3 = fig3.sort_values(["split", "metric", "csv"], kind="mergesort").reset_index(drop=True)
    fig3.to_csv(fig3_path, index=False)
    LOGGER.info("Updated fig3 with %d %s rows: %s", len(out_rows), source_name, fig3_path)


def write_fig4(
    fig4_path: Path,
    zip_metrics_all: pd.DataFrame,
    before_manifest: Path,
    after_manifest: Path,
    psnr_thr: float,
    ssim_thr: float,
    source_name: str,
) -> None:
    fig4 = pd.read_csv(fig4_path)
    fig4 = fig4[fig4["csv"].astype(str).str.lower() != source_name.lower()].copy()

    before_rows = summarize_gaps(
        zip_metrics_all, load_manifest_ids(before_manifest), source_name, psnr_thr, ssim_thr
    )
    after_rows = summarize_gaps(
        zip_metrics_all, load_manifest_ids(after_manifest), source_name, psnr_thr, ssim_thr
    )

    out_rows: List[Dict[str, object]] = []
    for phase, rows in (("before", before_rows), ("after", after_rows)):
        for r in rows:
            out_rows.append(
                {
                    "phase": phase,
                    "csv": r["csv"],
                    "metric": r["metric"],
                    "diff_subset_minus_full": r["diff_subset_minus_full"],
                    "abs_diff": r["abs_diff"],
                    "threshold": r["threshold"],
                    "pass": r["pass"],
                }
            )
    fig4 = pd.concat([fig4, pd.DataFrame(out_rows)], ignore_index=True)
    fig4["csv"] = fig4["csv"].astype(str).str.lower()
    fig4["metric"] = fig4["metric"].astype(str).str.upper()
    fig4 = fig4.sort_values(["phase", "metric", "csv"], kind="mergesort").reset_index(drop=True)
    fig4.to_csv(fig4_path, index=False)
    LOGGER.info("Updated fig4 with %d %s rows: %s", len(out_rows), source_name, fig4_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment share-bundle figure CSVs with ZipNeRF.")
    p.add_argument("--zipnerf-input", type=Path, default=Path("latex/zipnerf.csv"))
    p.add_argument("--zipnerf-clean-output", type=Path, default=Path("zipnerf_metrics.csv"))
    p.add_argument(
        "--share-bundle-dir",
        type=Path,
        default=Path("sweep_cluster_k/share_bundle_prism_20260304"),
    )
    p.add_argument(
        "--global-subset-manifest",
        type=Path,
        default=Path("sweep_cluster_k/budget_sweep_k6_auto_v2/recommended_subset.csv"),
    )
    p.add_argument(
        "--holdout-split-csv",
        type=Path,
        default=Path("sweep_cluster_k/holdout_split_k6.csv"),
    )
    p.add_argument(
        "--holdout-tune-manifest",
        type=Path,
        default=Path("sweep_cluster_k/holdout_protocol_v3/tune_budget_eval/recommended_subset.csv"),
    )
    p.add_argument(
        "--holdout-test-manifest",
        type=Path,
        default=Path("sweep_cluster_k/holdout_protocol_v3/test_budget_eval/recommended_subset.csv"),
    )
    p.add_argument("--refine-before-manifest", type=Path, default=Path("subset_k6_manifest.csv"))
    p.add_argument(
        "--refine-after-manifest",
        type=Path,
        default=Path("sweep_cluster_k/refined_subset_k6_manifest.csv"),
    )
    p.add_argument("--source-name", default="zipnerf")
    p.add_argument("--psnr-threshold", type=float, default=0.5)
    p.add_argument("--ssim-threshold", type=float, default=0.01)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    zip_df = load_zipnerf_metrics(args.zipnerf_input)
    args.zipnerf_clean_output.parent.mkdir(parents=True, exist_ok=True)
    zip_df.to_csv(args.zipnerf_clean_output, index=False)
    LOGGER.info(
        "Parsed ZipNeRF metrics: %d scenes -> %s",
        len(zip_df),
        args.zipnerf_clean_output,
    )

    share_dir = args.share_bundle_dir
    fig2 = share_dir / "fig2_global_gaps_vs_threshold.csv"
    fig3 = share_dir / "fig3_holdout_tune_test_gaps.csv"
    fig4 = share_dir / "fig4_refinement_before_after.csv"
    for p in (fig2, fig3, fig4):
        if not p.exists():
            raise FileNotFoundError(f"Missing bundle file: {p}")

    write_fig2(
        fig2,
        zip_df,
        load_manifest_ids(args.global_subset_manifest),
        args.psnr_threshold,
        args.ssim_threshold,
        args.source_name,
    )
    write_fig3(
        fig3,
        zip_df,
        args.holdout_split_csv,
        args.holdout_tune_manifest,
        args.holdout_test_manifest,
        args.psnr_threshold,
        args.ssim_threshold,
        args.source_name,
    )
    write_fig4(
        fig4,
        zip_df,
        args.refine_before_manifest,
        args.refine_after_manifest,
        args.psnr_threshold,
        args.ssim_threshold,
        args.source_name,
    )

    LOGGER.info("ZipNeRF augmentation complete.")


if __name__ == "__main__":
    main()
