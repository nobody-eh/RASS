#!/usr/bin/env python3
"""Validate subset quality by metric matching against the full set.

The subset is considered validated if OC/FI PSNR/SSIM means stay close to
full-dataset means under user-defined thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("validate_subset_metric_matching")


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def _canonicalize_metric_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{source_name} metrics CSV is empty")

    orig_cols = list(df.columns)
    norm_to_orig: Dict[str, str] = {}
    for c in orig_cols:
        n = _normalize_col_name(c)
        if n not in norm_to_orig:
            norm_to_orig[n] = c

    def pick_first(candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            if c in norm_to_orig:
                return norm_to_orig[c]
        return None

    dish_col = pick_first(("dishid", "sceneid", "id"))
    if dish_col is None:
        raise ValueError(
            f"{source_name}: unable to find dish identifier column in {orig_cols}"
        )
    psnr_col = pick_first(("psnr", "pnsr"))
    ssim_col = pick_first(("ssim",))
    if psnr_col is None or ssim_col is None:
        raise ValueError(
            f"{source_name}: unable to find PSNR/SSIM columns in {orig_cols}"
        )

    out = pd.DataFrame()
    out["dish_id"] = df[dish_col].astype(str)
    out["PSNR"] = pd.to_numeric(df[psnr_col], errors="coerce")
    out["SSIM"] = pd.to_numeric(df[ssim_col], errors="coerce")
    out = out.dropna(subset=["PSNR", "SSIM"]).reset_index(drop=True)
    return out


def _merge_metrics_with_clusters(metrics: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = metrics.merge(mapping[["dish_id", "cluster"]], on="dish_id", how="inner")
    merged["cluster"] = pd.to_numeric(merged["cluster"], errors="coerce")
    merged = merged.dropna(subset=["cluster"]).copy()
    merged["cluster"] = merged["cluster"].astype(int)
    return merged


def _summarize_gaps(
    df: pd.DataFrame, subset_ids: set[str], source_name: str
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for metric in ("PSNR", "SSIM"):
        full = pd.to_numeric(df[metric], errors="coerce").dropna()
        sub = pd.to_numeric(
            df[df["dish_id"].isin(subset_ids)][metric], errors="coerce"
        ).dropna()
        if full.empty or sub.empty:
            continue
        full_mean = float(full.mean())
        subset_mean = float(sub.mean())
        diff = subset_mean - full_mean
        rel = (100.0 * diff / full_mean) if abs(full_mean) > 1e-12 else np.nan
        out.append(
            {
                "csv": source_name,
                "metric": metric,
                "full_n": int(full.shape[0]),
                "subset_n": int(sub.shape[0]),
                "full_mean": full_mean,
                "subset_mean": subset_mean,
                "diff_subset_minus_full": float(diff),
                "abs_diff": float(abs(diff)),
                "relative_diff_percent": float(rel),
            }
        )
    return out


def _compute_cluster_gap_table(
    df: pd.DataFrame, subset_ids: set[str], source_name: str
) -> pd.DataFrame:
    full_means = (
        df.groupby("cluster", dropna=False)[["PSNR", "SSIM"]]
        .mean()
        .rename(columns={"PSNR": "full_psnr", "SSIM": "full_ssim"})
    )
    sub_means = (
        df[df["dish_id"].isin(subset_ids)]
        .groupby("cluster", dropna=False)[["PSNR", "SSIM"]]
        .mean()
        .rename(columns={"PSNR": "subset_psnr", "SSIM": "subset_ssim"})
    )
    out = full_means.join(sub_means, how="left").reset_index()
    out["gap_psnr"] = out["subset_psnr"] - out["full_psnr"]
    out["gap_ssim"] = out["subset_ssim"] - out["full_ssim"]
    out["csv"] = source_name
    out = out[
        [
            "csv",
            "cluster",
            "full_psnr",
            "subset_psnr",
            "gap_psnr",
            "full_ssim",
            "subset_ssim",
            "gap_ssim",
        ]
    ]
    out = out.sort_values(["csv", "cluster"], kind="mergesort").reset_index(drop=True)
    return out


def _subset_cluster_counts(
    mapping_df: pd.DataFrame, subset_ids: set[str]
) -> Dict[int, int]:
    sub = mapping_df[mapping_df["dish_id"].isin(subset_ids)].copy()
    sub["cluster"] = pd.to_numeric(sub["cluster"], errors="coerce")
    sub = sub.dropna(subset=["cluster"]).copy()
    sub["cluster"] = sub["cluster"].astype(int)
    vc = sub["cluster"].value_counts().sort_index()
    return {int(k): int(v) for k, v in vc.items()}


def _balanced_simulation(
    df: pd.DataFrame,
    cluster_target_counts: Dict[int, int],
    observed_diff: float,
    metric: str,
    num_simulations: int,
    random_seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_seed)
    full_mean = float(pd.to_numeric(df[metric], errors="coerce").dropna().mean())

    groups: Dict[int, np.ndarray] = {}
    for cluster, count in cluster_target_counts.items():
        g = df[df["cluster"] == int(cluster)][metric]
        vals = pd.to_numeric(g, errors="coerce").dropna().to_numpy(dtype=float)
        if vals.shape[0] < count:
            raise ValueError(
                f"Cluster {cluster} has {vals.shape[0]} rows for metric {metric}, "
                f"but subset requires {count}."
            )
        groups[int(cluster)] = vals

    diffs = np.empty(int(num_simulations), dtype=float)
    cluster_items = sorted(groups.items(), key=lambda kv: kv[0])
    for i in range(int(num_simulations)):
        parts: List[np.ndarray] = []
        for cluster, vals in cluster_items:
            k = int(cluster_target_counts[cluster])
            idx = rng.choice(vals.shape[0], size=k, replace=False)
            parts.append(vals[idx])
        sample = np.concatenate(parts) if parts else np.array([], dtype=float)
        sample_mean = float(sample.mean()) if sample.size > 0 else np.nan
        diffs[i] = sample_mean - full_mean

    pctl = float(100.0 * np.mean(diffs <= observed_diff))
    two_sided = float(2.0 * min(pctl / 100.0, 1.0 - pctl / 100.0))
    q025, q25, q50, q75, q975 = np.percentile(diffs, [2.5, 25, 50, 75, 97.5])
    return {
        "metric": metric,
        "observed_diff": float(observed_diff),
        "sim_mean_diff": float(np.mean(diffs)),
        "sim_std_diff": float(np.std(diffs, ddof=0)),
        "sim_q2_5": float(q025),
        "sim_q25": float(q25),
        "sim_q50": float(q50),
        "sim_q75": float(q75),
        "sim_q97_5": float(q975),
        "observed_percentile": float(pctl),
        "two_sided_pvalue": float(two_sided),
        "num_simulations": int(num_simulations),
    }


def _validate_thresholds(
    gap_rows: List[Dict[str, object]],
    max_abs_psnr_gap: float,
    max_abs_ssim_gap: float,
) -> Tuple[bool, List[Dict[str, object]]]:
    checks: List[Dict[str, object]] = []
    for row in gap_rows:
        metric = str(row["metric"]).upper()
        abs_diff = float(row["abs_diff"])
        thr = max_abs_psnr_gap if metric == "PSNR" else max_abs_ssim_gap
        ok = abs_diff <= float(thr)
        checks.append(
            {
                "csv": row["csv"],
                "metric": metric,
                "abs_diff": abs_diff,
                "threshold": float(thr),
                "pass": bool(ok),
            }
        )
    overall = all(c["pass"] for c in checks) if checks else False
    return overall, checks


def _validate_per_cluster_thresholds(
    cluster_gap_df: pd.DataFrame,
    max_per_cluster_psnr_gap: Optional[float],
    max_per_cluster_ssim_gap: Optional[float],
) -> Tuple[bool, List[Dict[str, object]]]:
    checks: List[Dict[str, object]] = []
    if cluster_gap_df.empty:
        return False, checks

    for csv_name, sub in cluster_gap_df.groupby("csv", dropna=False):
        max_psnr = float(np.nanmax(np.abs(pd.to_numeric(sub["gap_psnr"], errors="coerce"))))
        max_ssim = float(np.nanmax(np.abs(pd.to_numeric(sub["gap_ssim"], errors="coerce"))))

        if max_per_cluster_psnr_gap is not None:
            checks.append(
                {
                    "csv": str(csv_name),
                    "metric": "PSNR",
                    "max_abs_gap": max_psnr,
                    "threshold": float(max_per_cluster_psnr_gap),
                    "pass": bool(max_psnr <= float(max_per_cluster_psnr_gap)),
                }
            )
        if max_per_cluster_ssim_gap is not None:
            checks.append(
                {
                    "csv": str(csv_name),
                    "metric": "SSIM",
                    "max_abs_gap": max_ssim,
                    "threshold": float(max_per_cluster_ssim_gap),
                    "pass": bool(max_ssim <= float(max_per_cluster_ssim_gap)),
                }
            )

    overall = all(c["pass"] for c in checks) if checks else True
    return overall, checks


def _apply_split_filter(
    df: pd.DataFrame, split_df: pd.DataFrame, eval_split: str
) -> pd.DataFrame:
    if eval_split == "all":
        valid_ids = set(split_df["dish_id"].astype(str))
    else:
        valid_ids = set(
            split_df[split_df["split"].astype(str) == str(eval_split)]["dish_id"].astype(str)
        )
    out = df[df["dish_id"].astype(str).isin(valid_ids)].copy()
    return out


def _write_markdown(
    path: Path,
    summary: Dict[str, object],
    gap_df: pd.DataFrame,
    cluster_gap_df: pd.DataFrame,
) -> None:
    lines: List[str] = [
        "# Subset Metric-Matching Validation",
        "",
        f"- Subset manifest: `{summary['subset_manifest']}`",
        f"- Cluster mapping: `{summary['cluster_mapping_csv']}`",
        f"- Subset size: **{summary['subset_size']}**",
        f"- Eval split: **{summary['eval_split']}**",
        f"- Overall pass: **{summary['overall_pass']}**",
        "",
        "## Global Gaps (subset vs full)",
        "",
        "| csv | metric | full_mean | subset_mean | diff | abs_diff | rel_diff_% |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in gap_df.iterrows():
        lines.append(
            "| {csv} | {metric} | {full_mean:.6f} | {subset_mean:.6f} | "
            "{diff_subset_minus_full:.6f} | {abs_diff:.6f} | {relative_diff_percent:.3f} |".format(
                **row.to_dict()
            )
        )

    lines += [
        "",
        "## Threshold Checks",
        "",
        "| csv | metric | abs_diff | threshold | pass |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["threshold_checks"]:
        lines.append(
            f"| {row['csv']} | {row['metric']} | {float(row['abs_diff']):.6f} | "
            f"{float(row['threshold']):.6f} | {bool(row['pass'])} |"
        )

    if summary["per_cluster_threshold_checks"]:
        lines += [
            "",
            "## Per-Cluster Threshold Checks",
            "",
            "| csv | metric | max_abs_cluster_gap | threshold | pass |",
            "|---|---|---:|---:|---:|",
        ]
        for row in summary["per_cluster_threshold_checks"]:
            lines.append(
                f"| {row['csv']} | {row['metric']} | {float(row['max_abs_gap']):.6f} | "
                f"{float(row['threshold']):.6f} | {bool(row['pass'])} |"
            )

    lines += [
        "",
        "## Balanced-Random Baseline",
        "",
        "| csv | metric | observed_diff | q2.5 | q50 | q97.5 | percentile | p(two-sided) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["random_baseline"]:
        lines.append(
            f"| {row['csv']} | {row['metric']} | {float(row['observed_diff']):.6f} | "
            f"{float(row['sim_q2_5']):.6f} | {float(row['sim_q50']):.6f} | {float(row['sim_q97_5']):.6f} | "
            f"{float(row['observed_percentile']):.2f} | {float(row['two_sided_pvalue']):.4f} |"
        )

    lines += [
        "",
        "## Per-Cluster Gaps",
        "",
        "| csv | cluster | full_psnr | subset_psnr | gap_psnr | full_ssim | subset_ssim | gap_ssim |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in cluster_gap_df.iterrows():
        lines.append(
            f"| {row['csv']} | {int(row['cluster'])} | "
            f"{float(row['full_psnr']):.6f} | {float(row['subset_psnr']):.6f} | {float(row['gap_psnr']):.6f} | "
            f"{float(row['full_ssim']):.6f} | {float(row['subset_ssim']):.6f} | {float(row['gap_ssim']):.6f} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate subset metric matching against full set."
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        required=True,
        help="Subset manifest CSV (must contain dish_id; cluster optional).",
    )
    parser.add_argument(
        "--cluster-mapping-csv",
        type=Path,
        required=True,
        help="Dish->cluster mapping CSV from selected-k run.",
    )
    parser.add_argument("--oc-csv", type=Path, default=Path("ingp_oc.csv"))
    parser.add_argument("--fi-csv", type=Path, default=Path("ingp_fi.csv"))
    parser.add_argument("--num-simulations", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-abs-psnr-gap", type=float, default=0.5)
    parser.add_argument("--max-abs-ssim-gap", type=float, default=0.01)
    parser.add_argument(
        "--max-per-cluster-psnr-gap",
        type=float,
        default=None,
        help="Optional per-cluster max abs PSNR gap.",
    )
    parser.add_argument(
        "--max-per-cluster-ssim-gap",
        type=float,
        default=None,
        help="Optional per-cluster max abs SSIM gap.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional split CSV with columns: dish_id, split",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["all", "selection", "tune", "test"],
        default="all",
        help="Split partition to evaluate on when split CSV is provided.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("subset_metric_matching_report.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("subset_metric_matching_report.md"),
    )
    parser.add_argument(
        "--output-gap-csv",
        type=Path,
        default=Path("subset_metric_matching_gaps.csv"),
    )
    parser.add_argument(
        "--output-cluster-gap-csv",
        type=Path,
        default=Path("subset_metric_matching_cluster_gaps.csv"),
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

    subset_df = pd.read_csv(args.subset_manifest)
    if "dish_id" not in subset_df.columns:
        raise ValueError("subset manifest must contain 'dish_id' column")
    subset_ids = set(subset_df["dish_id"].astype(str))
    if not subset_ids:
        raise ValueError("subset manifest has no dish_id entries")

    mapping_df = pd.read_csv(args.cluster_mapping_csv)
    if "dish_id" not in mapping_df.columns or "cluster" not in mapping_df.columns:
        raise ValueError("cluster mapping CSV must contain dish_id, cluster columns")
    mapping_df = mapping_df[["dish_id", "cluster"]].copy()
    mapping_df["dish_id"] = mapping_df["dish_id"].astype(str)

    oc = _canonicalize_metric_columns(pd.read_csv(args.oc_csv), "oc")
    fi = _canonicalize_metric_columns(pd.read_csv(args.fi_csv), "fi")

    oc_m = _merge_metrics_with_clusters(oc, mapping_df)
    fi_m = _merge_metrics_with_clusters(fi, mapping_df)

    if args.split_csv is not None:
        split_df = pd.read_csv(args.split_csv)
        if "dish_id" not in split_df.columns or "split" not in split_df.columns:
            raise ValueError("split CSV must contain dish_id and split columns")
        split_df["dish_id"] = split_df["dish_id"].astype(str)
        split_df["split"] = split_df["split"].astype(str)
        oc_m = _apply_split_filter(oc_m, split_df, args.eval_split)
        fi_m = _apply_split_filter(fi_m, split_df, args.eval_split)
        if oc_m.empty or fi_m.empty:
            raise RuntimeError(
                f"No rows left after applying split filter eval_split={args.eval_split}"
            )
        subset_ids = subset_ids.intersection(set(split_df["dish_id"].astype(str)))

    gaps = _summarize_gaps(oc_m, subset_ids, "oc") + _summarize_gaps(fi_m, subset_ids, "fi")
    if not gaps:
        raise RuntimeError(
            "No subset-vs-full overlap after filters. Check subset manifest, split, and mapping."
        )
    gap_df = pd.DataFrame(gaps).sort_values(["csv", "metric"], kind="mergesort")

    cluster_gap_df = pd.concat(
        [
            _compute_cluster_gap_table(oc_m, subset_ids, "oc"),
            _compute_cluster_gap_table(fi_m, subset_ids, "fi"),
        ],
        ignore_index=True,
    ).sort_values(["csv", "cluster"], kind="mergesort")

    cluster_target_counts = _subset_cluster_counts(mapping_df, subset_ids)
    if not cluster_target_counts:
        raise RuntimeError("No subset rows matched cluster mapping.")

    random_rows: List[Dict[str, object]] = []
    for source_name, df in (("oc", oc_m), ("fi", fi_m)):
        for metric in ("PSNR", "SSIM"):
            row = gap_df[(gap_df["csv"] == source_name) & (gap_df["metric"] == metric)]
            if row.empty:
                continue
            observed_diff = float(row.iloc[0]["diff_subset_minus_full"])
            sim = _balanced_simulation(
                df=df,
                cluster_target_counts=cluster_target_counts,
                observed_diff=observed_diff,
                metric=metric,
                num_simulations=int(args.num_simulations),
                random_seed=int(args.random_seed),
            )
            sim["csv"] = source_name
            random_rows.append(sim)

    overall_pass, threshold_checks = _validate_thresholds(
        gaps,
        max_abs_psnr_gap=float(args.max_abs_psnr_gap),
        max_abs_ssim_gap=float(args.max_abs_ssim_gap),
    )
    per_cluster_pass, per_cluster_threshold_checks = _validate_per_cluster_thresholds(
        cluster_gap_df,
        max_per_cluster_psnr_gap=args.max_per_cluster_psnr_gap,
        max_per_cluster_ssim_gap=args.max_per_cluster_ssim_gap,
    )
    overall_combined_pass = bool(overall_pass and per_cluster_pass)

    summary: Dict[str, object] = {
        "subset_manifest": str(args.subset_manifest),
        "cluster_mapping_csv": str(args.cluster_mapping_csv),
        "oc_csv": str(args.oc_csv),
        "fi_csv": str(args.fi_csv),
        "split_csv": None if args.split_csv is None else str(args.split_csv),
        "eval_split": str(args.eval_split),
        "subset_size": int(len(subset_ids)),
        "subset_cluster_counts": {str(k): int(v) for k, v in sorted(cluster_target_counts.items())},
        "num_simulations": int(args.num_simulations),
        "random_seed": int(args.random_seed),
        "thresholds": {
            "max_abs_psnr_gap": float(args.max_abs_psnr_gap),
            "max_abs_ssim_gap": float(args.max_abs_ssim_gap),
            "max_per_cluster_psnr_gap": args.max_per_cluster_psnr_gap,
            "max_per_cluster_ssim_gap": args.max_per_cluster_ssim_gap,
        },
        "overall_pass": bool(overall_combined_pass),
        "overall_global_pass": bool(overall_pass),
        "overall_per_cluster_pass": bool(per_cluster_pass),
        "threshold_checks": threshold_checks,
        "per_cluster_threshold_checks": per_cluster_threshold_checks,
        "random_baseline": random_rows,
        "global_gaps": gaps,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_gap_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_cluster_gap_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)

    gap_df.to_csv(args.output_gap_csv, index=False)
    cluster_gap_df.to_csv(args.output_cluster_gap_csv, index=False)
    _write_markdown(args.output_markdown, summary, gap_df, cluster_gap_df)

    LOGGER.info("Subset size: %d", len(subset_ids))
    LOGGER.info("Overall pass: %s", overall_combined_pass)
    LOGGER.info("Saved JSON: %s", args.output_json)
    LOGGER.info("Saved Markdown: %s", args.output_markdown)
    LOGGER.info("Saved global gaps CSV: %s", args.output_gap_csv)
    LOGGER.info("Saved cluster gaps CSV: %s", args.output_cluster_gap_csv)


if __name__ == "__main__":
    main()
