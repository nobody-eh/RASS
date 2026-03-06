#!/usr/bin/env python3
"""Sweep subset budgets and estimate metric-matching quality.

For each budget `b` (scenes per cluster), this script performs balanced
cluster sampling simulations and reports expected/p95 metric gaps for:
- OC PSNR
- OC SSIM
- FI PSNR
- FI SSIM

It also reports joint pass rate under user-defined thresholds and writes
one best-sample manifest per budget (lowest normalized gap objective).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from statistics import NormalDist
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("sweep_subset_budgets")


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
    out[f"{source_name}_psnr"] = pd.to_numeric(df[psnr_col], errors="coerce")
    out[f"{source_name}_ssim"] = pd.to_numeric(df[ssim_col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def _parse_budgets(value: str) -> List[int]:
    out: List[int] = []
    for token in value.split(","):
        s = token.strip()
        if not s:
            continue
        b = int(s)
        if b < 1:
            raise ValueError(f"Invalid budget {b}; must be >= 1")
        out.append(b)
    if not out:
        raise ValueError("No valid budgets provided")
    return sorted(set(out))


def _to_jsonable(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def _wilson_lower_bound(p_hat: float, n: int, confidence_level: float) -> float:
    if n <= 0:
        return 0.0
    cl = min(max(float(confidence_level), 0.5001), 0.999999)
    z = float(NormalDist().inv_cdf(0.5 + cl / 2.0))
    n_f = float(n)
    denom = 1.0 + (z * z) / n_f
    center = (p_hat + (z * z) / (2.0 * n_f)) / denom
    margin = (
        z
        * np.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n_f)) / n_f)
        / denom
    )
    return float(max(0.0, center - margin))


def _load_runtime_table(runtime_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(runtime_csv)
    if df.empty:
        raise ValueError(f"runtime CSV is empty: {runtime_csv}")
    cols = {_normalize_col_name(c): c for c in df.columns}
    id_col = cols.get("dishid") or cols.get("sceneid") or cols.get("id")
    rt_col = (
        cols.get("runtimesec")
        or cols.get("runtime")
        or cols.get("seconds")
        or cols.get("timesec")
        or cols.get("elapsedsec")
    )
    if id_col is None or rt_col is None:
        raise ValueError(
            f"runtime CSV must contain id + runtime columns. Found: {list(df.columns)}"
        )
    out = pd.DataFrame()
    out["dish_id"] = df[id_col].astype(str)
    out["runtime_sec"] = pd.to_numeric(df[rt_col], errors="coerce")
    out = out.dropna(subset=["runtime_sec"]).reset_index(drop=True)
    return out


def _load_split_ids(split_csv: Path, eval_split: str) -> set[str]:
    df = pd.read_csv(split_csv)
    if "dish_id" not in df.columns or "split" not in df.columns:
        raise ValueError("split CSV must contain dish_id and split columns")
    df["dish_id"] = df["dish_id"].astype(str)
    df["split"] = df["split"].astype(str)
    if eval_split == "all":
        return set(df["dish_id"].astype(str))
    return set(df[df["split"] == str(eval_split)]["dish_id"].astype(str))


def _make_markdown(
    path: Path,
    rows: List[Dict[str, object]],
    config: Dict[str, object],
    recommendation: Optional[Dict[str, object]],
) -> None:
    lines = [
        "# Budget Sweep Metric-Matching Summary",
        "",
        f"- Cluster mapping: `{config['cluster_mapping_csv']}`",
        f"- OC CSV: `{config['oc_csv']}`",
        f"- FI CSV: `{config['fi_csv']}`",
        f"- Split: `{config['eval_split']}`",
        f"- Simulations per budget: **{config['num_simulations']}**",
        f"- Thresholds: PSNR <= **{config['max_abs_psnr_gap']}**, SSIM <= **{config['max_abs_ssim_gap']}**",
        f"- Target joint pass rate: **{config['target_joint_pass_rate']}**",
    "",
    ]
    if recommendation is not None:
        lines += [
            "",
            "## Auto Selection",
            f"- Recommended budget per cluster: **{recommendation['recommended_budget_per_cluster']}**",
            f"- Recommended total subset: **{recommendation['recommended_total_subset_size']}**",
            f"- Recommended joint pass rate: **{recommendation['recommended_joint_pass_rate']:.4f}**",
            f"- Selection mode: **{recommendation['selection_mode']}**",
            f"- Target met: **{recommendation['target_met']}**",
            f"- Max-total constraint applied: **{recommendation['max_total_subset_applied']}**",
            f"- Recommended manifest: `{recommendation['recommended_manifest']}`",
            f"- Exported manifest copy: `{recommendation['exported_manifest']}`",
            "",
        ]

    lines += [
        "| rank | rec | budget_per_cluster | total_subset | feasible | joint_pass_rate | joint_pass_lcb | best_global_pass | best_cluster_pass | exp_abs_oc_psnr | exp_abs_oc_ssim | exp_abs_fi_psnr | exp_abs_fi_ssim | p95_abs_fi_psnr | exp_runtime_mean | best_manifest |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {rank} | {recommended} | {budget_per_cluster} | {total_subset_size} | {feasible} | {joint_pass_rate:.4f} | {joint_pass_rate_lcb:.4f} | {best_global_pass} | {best_per_cluster_pass} | "
            "{exp_abs_oc_psnr:.4f} | {exp_abs_oc_ssim:.4f} | {exp_abs_fi_psnr:.4f} | {exp_abs_fi_ssim:.4f} | "
            "{p95_abs_fi_psnr:.4f} | {exp_runtime_mean:.4f} | {best_manifest} |".format(**r)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_combined(
    mapping_csv: Path,
    oc_csv: Path,
    fi_csv: Path,
    runtime_csv: Optional[Path],
    split_csv: Optional[Path] = None,
    eval_split: str = "all",
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], int, bool]:
    mapping = pd.read_csv(mapping_csv)
    if "dish_id" not in mapping.columns or "cluster" not in mapping.columns:
        raise ValueError("cluster mapping CSV must contain dish_id, cluster")
    mapping = mapping[["dish_id", "cluster"]].copy()
    mapping["dish_id"] = mapping["dish_id"].astype(str)
    mapping["cluster"] = pd.to_numeric(mapping["cluster"], errors="coerce")
    mapping = mapping.dropna(subset=["cluster"]).copy()
    mapping["cluster"] = mapping["cluster"].astype(int)
    if split_csv is not None:
        keep_ids = _load_split_ids(split_csv, eval_split)
        mapping = mapping[mapping["dish_id"].isin(keep_ids)].copy()

    oc = _canonicalize_metric_columns(pd.read_csv(oc_csv), "oc")
    fi = _canonicalize_metric_columns(pd.read_csv(fi_csv), "fi")

    df = mapping.merge(oc, on="dish_id", how="inner").merge(fi, on="dish_id", how="inner")
    runtime_available = False
    if runtime_csv is not None:
        runtime_df = _load_runtime_table(runtime_csv)
        df = df.merge(runtime_df, on="dish_id", how="left")
        runtime_available = "runtime_sec" in df.columns and df["runtime_sec"].notna().any()
    if df.empty:
        raise RuntimeError("No overlap among cluster mapping and OC/FI metrics")
    df = df.sort_values("dish_id", kind="mergesort").reset_index(drop=True)
    clusters = sorted(df["cluster"].unique().tolist())
    groups = {int(c): df.index[df["cluster"] == int(c)].to_numpy(dtype=int) for c in clusters}
    return df, groups, len(clusters), runtime_available


def _simulate_budget(
    df: pd.DataFrame,
    groups: Dict[int, np.ndarray],
    budget_per_cluster: int,
    num_simulations: int,
    seed: int,
    max_abs_psnr_gap: float,
    max_abs_ssim_gap: float,
    confidence_level: float,
    max_per_cluster_psnr_gap: Optional[float],
    max_per_cluster_ssim_gap: Optional[float],
    weight_runtime: float,
) -> Dict[str, object]:
    metrics = ["oc_psnr", "oc_ssim", "fi_psnr", "fi_ssim"]
    full_means = df[metrics].mean().to_numpy(dtype=float)
    cluster_mean_full = (
        df.groupby("cluster", dropna=False)[metrics]
        .mean()
        .sort_index(kind="mergesort")
    )

    min_cluster_size = min(int(v.shape[0]) for v in groups.values())
    if budget_per_cluster > min_cluster_size:
        return {
            "budget_per_cluster": int(budget_per_cluster),
            "total_subset_size": int(budget_per_cluster * len(groups)),
            "feasible": False,
            "reason": f"budget>{min_cluster_size} (min available per cluster)",
        }

    rng = np.random.default_rng(seed)
    diffs = np.empty((int(num_simulations), 4), dtype=float)
    runtime_means = np.full(int(num_simulations), np.nan, dtype=float)
    threshold_vec = np.array(
        [max_abs_psnr_gap, max_abs_ssim_gap, max_abs_psnr_gap, max_abs_ssim_gap],
        dtype=float,
    )
    has_runtime = "runtime_sec" in df.columns and df["runtime_sec"].notna().any()
    full_runtime_mean = (
        float(pd.to_numeric(df["runtime_sec"], errors="coerce").mean())
        if has_runtime
        else np.nan
    )
    best_obj = float("inf")
    best_indices: Optional[np.ndarray] = None

    cluster_items = sorted(groups.items(), key=lambda kv: kv[0])
    for i in range(int(num_simulations)):
        sampled_parts: List[np.ndarray] = []
        for _, idx in cluster_items:
            pick = rng.choice(idx, size=int(budget_per_cluster), replace=False)
            sampled_parts.append(pick)
        sample_idx = np.concatenate(sampled_parts)

        sample_mean = df.iloc[sample_idx][metrics].mean().to_numpy(dtype=float)
        diff = sample_mean - full_means
        diffs[i, :] = diff
        if has_runtime:
            rt_mean = float(pd.to_numeric(df.iloc[sample_idx]["runtime_sec"], errors="coerce").mean())
            runtime_means[i] = rt_mean

        obj = float(np.sum(np.abs(diff) / threshold_vec))
        if has_runtime and full_runtime_mean > 1e-12 and weight_runtime > 0.0:
            obj += float(weight_runtime) * max(0.0, rt_mean / full_runtime_mean)
        if obj < best_obj:
            best_obj = obj
            best_indices = sample_idx.copy()

    assert best_indices is not None

    abs_diffs = np.abs(diffs)
    pass_mask = (
        (abs_diffs[:, 0] <= max_abs_psnr_gap)
        & (abs_diffs[:, 1] <= max_abs_ssim_gap)
        & (abs_diffs[:, 2] <= max_abs_psnr_gap)
        & (abs_diffs[:, 3] <= max_abs_ssim_gap)
    )
    joint_pass_rate = float(np.mean(pass_mask))
    joint_pass_rate_lcb = _wilson_lower_bound(
        p_hat=joint_pass_rate,
        n=int(num_simulations),
        confidence_level=confidence_level,
    )

    best_df = df.iloc[best_indices][["dish_id", "cluster"]].copy()
    best_df = best_df.sort_values(["cluster", "dish_id"], kind="mergesort").reset_index(drop=True)
    best_metric_means = df.iloc[best_indices][metrics].mean().to_numpy(dtype=float)
    best_metric_diff = best_metric_means - full_means
    best_global_pass = bool(
        (abs(best_metric_diff[0]) <= max_abs_psnr_gap)
        and (abs(best_metric_diff[1]) <= max_abs_ssim_gap)
        and (abs(best_metric_diff[2]) <= max_abs_psnr_gap)
        and (abs(best_metric_diff[3]) <= max_abs_ssim_gap)
    )

    # Per-cluster gap check on best subset manifest.
    best_cluster = (
        df.iloc[best_indices]
        .groupby("cluster", dropna=False)[metrics]
        .mean()
        .sort_index(kind="mergesort")
    )
    aligned = cluster_mean_full.join(best_cluster, lsuffix="_full", rsuffix="_subset")
    per_cluster_psnr_gaps = np.concatenate(
        [
            np.abs(aligned["oc_psnr_subset"] - aligned["oc_psnr_full"]).to_numpy(dtype=float),
            np.abs(aligned["fi_psnr_subset"] - aligned["fi_psnr_full"]).to_numpy(dtype=float),
        ]
    )
    per_cluster_ssim_gaps = np.concatenate(
        [
            np.abs(aligned["oc_ssim_subset"] - aligned["oc_ssim_full"]).to_numpy(dtype=float),
            np.abs(aligned["fi_ssim_subset"] - aligned["fi_ssim_full"]).to_numpy(dtype=float),
        ]
    )
    max_per_cluster_psnr = float(np.max(per_cluster_psnr_gaps)) if per_cluster_psnr_gaps.size else np.nan
    max_per_cluster_ssim = float(np.max(per_cluster_ssim_gaps)) if per_cluster_ssim_gaps.size else np.nan
    best_per_cluster_pass = True
    if max_per_cluster_psnr_gap is not None:
        best_per_cluster_pass = best_per_cluster_pass and (max_per_cluster_psnr <= float(max_per_cluster_psnr_gap))
    if max_per_cluster_ssim_gap is not None:
        best_per_cluster_pass = best_per_cluster_pass and (max_per_cluster_ssim <= float(max_per_cluster_ssim_gap))

    return {
        "budget_per_cluster": int(budget_per_cluster),
        "total_subset_size": int(budget_per_cluster * len(groups)),
        "feasible": True,
        "reason": "",
        "joint_pass_rate": joint_pass_rate,
        "joint_pass_rate_lcb": float(joint_pass_rate_lcb),
        "exp_abs_oc_psnr": float(np.mean(abs_diffs[:, 0])),
        "exp_abs_oc_ssim": float(np.mean(abs_diffs[:, 1])),
        "exp_abs_fi_psnr": float(np.mean(abs_diffs[:, 2])),
        "exp_abs_fi_ssim": float(np.mean(abs_diffs[:, 3])),
        "p95_abs_oc_psnr": float(np.percentile(abs_diffs[:, 0], 95)),
        "p95_abs_oc_ssim": float(np.percentile(abs_diffs[:, 1], 95)),
        "p95_abs_fi_psnr": float(np.percentile(abs_diffs[:, 2], 95)),
        "p95_abs_fi_ssim": float(np.percentile(abs_diffs[:, 3], 95)),
        "exp_runtime_mean": float(np.nanmean(runtime_means)) if has_runtime else np.nan,
        "p95_runtime_mean": float(np.nanpercentile(runtime_means, 95)) if has_runtime else np.nan,
        "best_runtime_mean": (
            float(pd.to_numeric(df.iloc[best_indices]["runtime_sec"], errors="coerce").mean())
            if has_runtime
            else np.nan
        ),
        "best_global_pass": bool(best_global_pass),
        "best_per_cluster_pass": bool(best_per_cluster_pass),
        "max_per_cluster_psnr_gap_best": max_per_cluster_psnr,
        "max_per_cluster_ssim_gap_best": max_per_cluster_ssim,
        "best_objective": float(best_obj),
        "best_subset_df": best_df,
    }


def _select_budget(
    summary_df: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.Series, Dict[str, object]]:
    feasible_df = summary_df[summary_df["feasible"] == True].copy()  # noqa: E712
    if feasible_df.empty:
        raise RuntimeError("No feasible budgets available for recommendation.")

    candidate_df = feasible_df
    passing_constraints_applied = False
    if args.require_best_manifest_constraints:
        constrained = candidate_df[
            (candidate_df["best_global_pass"] == True)  # noqa: E712
            & (candidate_df["best_per_cluster_pass"] == True)  # noqa: E712
        ]
        if not constrained.empty:
            candidate_df = constrained
            passing_constraints_applied = True

    max_total_subset_applied = False
    if args.max_total_subset is not None:
        constrained = candidate_df[
            pd.to_numeric(candidate_df["total_subset_size"], errors="coerce")
            <= int(args.max_total_subset)
        ]
        if not constrained.empty:
            candidate_df = constrained
            max_total_subset_applied = True

    target = float(args.target_joint_pass_rate)
    metric_col = "joint_pass_rate_lcb" if args.selection_use_lcb else "joint_pass_rate"
    meets_target = candidate_df[pd.to_numeric(candidate_df[metric_col], errors="coerce") >= target]

    if not meets_target.empty:
        selection_mode = "min_size_meeting_target"
        selected = (
            meets_target.sort_values(
                ["total_subset_size", metric_col, "best_objective"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            .iloc[0]
            .copy()
        )
        target_met = True
    else:
        selection_mode = "max_metric_fallback"
        selected = (
            candidate_df.sort_values(
                [metric_col, "total_subset_size", "best_objective"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .iloc[0]
            .copy()
        )
        target_met = False

    recommendation = {
        "recommended_budget_per_cluster": int(selected["budget_per_cluster"]),
        "recommended_total_subset_size": int(selected["total_subset_size"]),
        "recommended_joint_pass_rate": float(selected["joint_pass_rate"]),
        "recommended_joint_pass_rate_lcb": float(selected["joint_pass_rate_lcb"]),
        "selection_mode": selection_mode,
        "selection_metric": metric_col,
        "target_joint_pass_rate": target,
        "target_met": target_met,
        "max_total_subset": args.max_total_subset,
        "max_total_subset_applied": max_total_subset_applied,
        "require_best_manifest_constraints": bool(args.require_best_manifest_constraints),
        "passing_constraints_applied": passing_constraints_applied,
        "recommended_manifest": str(selected["best_manifest"]),
    }
    return selected, recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep multiple subset budgets and estimate metric-matching quality."
    )
    parser.add_argument(
        "--cluster-mapping-csv",
        type=Path,
        required=True,
        help="dish_id->cluster mapping CSV (selected k)",
    )
    parser.add_argument("--oc-csv", type=Path, default=Path("ingp_oc.csv"))
    parser.add_argument("--fi-csv", type=Path, default=Path("ingp_fi.csv"))
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=None,
        help="Optional CSV with dish_id + runtime_sec for runtime-aware selection.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Optional split CSV with columns dish_id, split.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["all", "selection", "tune", "test"],
        default="all",
        help="Split partition used for sweep (if split CSV is provided).",
    )
    parser.add_argument("--budgets", type=str, default="2,4,6,8,10")
    parser.add_argument("--num-simulations", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-abs-psnr-gap", type=float, default=0.5)
    parser.add_argument("--max-abs-ssim-gap", type=float, default=0.01)
    parser.add_argument(
        "--max-per-cluster-psnr-gap",
        type=float,
        default=None,
        help="Optional per-cluster max abs PSNR gap on best manifest.",
    )
    parser.add_argument(
        "--max-per-cluster-ssim-gap",
        type=float,
        default=None,
        help="Optional per-cluster max abs SSIM gap on best manifest.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for Wilson lower bound of joint pass rate.",
    )
    parser.add_argument(
        "--selection-use-lcb",
        action="store_true",
        default=True,
        help="Use LCB of joint pass rate for auto-selection target checks (default: true).",
    )
    parser.add_argument(
        "--selection-use-mean",
        dest="selection_use_lcb",
        action="store_false",
        help="Use mean joint pass rate (instead of LCB) for target checks.",
    )
    parser.add_argument(
        "--target-joint-pass-rate",
        type=float,
        default=0.10,
        help="Auto-selection target for joint pass rate.",
    )
    parser.add_argument(
        "--max-total-subset",
        type=int,
        default=None,
        help="If set, auto-selector prefers budgets with total subset <= this value.",
    )
    parser.add_argument(
        "--require-best-manifest-constraints",
        action="store_true",
        default=True,
        help="Enforce best-manifest global/per-cluster constraint pass before selection.",
    )
    parser.add_argument(
        "--allow-nonpassing-best-manifest",
        dest="require_best_manifest_constraints",
        action="store_false",
        help="Allow selection from budgets whose best manifest does not pass constraints.",
    )
    parser.add_argument(
        "--weight-runtime",
        type=float,
        default=0.0,
        help="Weight for runtime term in best-manifest objective (only if runtime CSV provided).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sweep_cluster_k/budget_sweep"),
    )
    parser.add_argument(
        "--summary-csv-name",
        type=str,
        default="budget_sweep_summary.csv",
    )
    parser.add_argument(
        "--summary-json-name",
        type=str,
        default="budget_sweep_summary.json",
    )
    parser.add_argument(
        "--summary-md-name",
        type=str,
        default="budget_sweep_summary.md",
    )
    parser.add_argument(
        "--manifests-dir-name",
        type=str,
        default="manifests",
    )
    parser.add_argument(
        "--recommended-manifest-name",
        type=str,
        default="recommended_subset.csv",
        help="Filename for copied recommended subset manifest.",
    )
    parser.add_argument(
        "--recommendation-json-name",
        type=str,
        default="budget_recommendation.json",
        help="Filename for recommendation metadata JSON.",
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

    budgets = _parse_budgets(args.budgets)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = out_dir / args.manifests_dir_name
    manifests_dir.mkdir(parents=True, exist_ok=True)

    df, groups, n_clusters, runtime_available = _prepare_combined(
        mapping_csv=args.cluster_mapping_csv,
        oc_csv=args.oc_csv,
        fi_csv=args.fi_csv,
        runtime_csv=args.runtime_csv,
        split_csv=args.split_csv,
        eval_split=args.eval_split,
    )
    min_cluster_size = min(int(v.shape[0]) for v in groups.values())
    LOGGER.info(
        "Prepared combined table: rows=%d | clusters=%d | min cluster size=%d",
        df.shape[0],
        n_clusters,
        min_cluster_size,
    )

    rows: List[Dict[str, object]] = []
    for i, b in enumerate(budgets):
        LOGGER.info("Evaluating budget_per_cluster=%d", b)
        res = _simulate_budget(
            df=df,
            groups=groups,
            budget_per_cluster=b,
            num_simulations=int(args.num_simulations),
            seed=int(args.random_seed) + (i * 97),
            max_abs_psnr_gap=float(args.max_abs_psnr_gap),
            max_abs_ssim_gap=float(args.max_abs_ssim_gap),
            confidence_level=float(args.confidence_level),
            max_per_cluster_psnr_gap=args.max_per_cluster_psnr_gap,
            max_per_cluster_ssim_gap=args.max_per_cluster_ssim_gap,
            weight_runtime=float(args.weight_runtime),
        )
        if not bool(res["feasible"]):
            row = {
                "budget_per_cluster": int(res["budget_per_cluster"]),
                "total_subset_size": int(res["total_subset_size"]),
                "feasible": False,
                "reason": str(res["reason"]),
                "joint_pass_rate": 0.0,
                "joint_pass_rate_lcb": 0.0,
                "exp_abs_oc_psnr": np.nan,
                "exp_abs_oc_ssim": np.nan,
                "exp_abs_fi_psnr": np.nan,
                "exp_abs_fi_ssim": np.nan,
                "p95_abs_oc_psnr": np.nan,
                "p95_abs_oc_ssim": np.nan,
                "p95_abs_fi_psnr": np.nan,
                "p95_abs_fi_ssim": np.nan,
                "exp_runtime_mean": np.nan,
                "p95_runtime_mean": np.nan,
                "best_runtime_mean": np.nan,
                "best_global_pass": False,
                "best_per_cluster_pass": False,
                "max_per_cluster_psnr_gap_best": np.nan,
                "max_per_cluster_ssim_gap_best": np.nan,
                "best_objective": np.nan,
                "best_manifest": "",
            }
            rows.append(row)
            continue

        best_manifest = manifests_dir / f"subset_budget_{int(b)}.csv"
        best_subset_df = res.pop("best_subset_df")
        assert isinstance(best_subset_df, pd.DataFrame)
        best_subset_df.to_csv(best_manifest, index=False)

        row = {k: v for k, v in res.items() if k != "best_subset_df"}
        row["best_manifest"] = str(best_manifest)
        rows.append(row)

    if not rows:
        raise RuntimeError("No budget results produced")

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        ["feasible", "joint_pass_rate", "total_subset_size"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected_row, recommendation = _select_budget(summary_df, args)
    recommended_manifest_src = Path(str(recommendation["recommended_manifest"]))
    recommended_manifest_dst = out_dir / args.recommended_manifest_name
    if not recommended_manifest_src.exists():
        raise FileNotFoundError(
            f"Recommended manifest does not exist: {recommended_manifest_src}"
        )
    shutil.copyfile(recommended_manifest_src, recommended_manifest_dst)
    recommendation["exported_manifest"] = str(recommended_manifest_dst)

    summary_df["recommended"] = (
        pd.to_numeric(summary_df["budget_per_cluster"], errors="coerce").astype(int)
        == int(recommendation["recommended_budget_per_cluster"])
    )
    summary_df["rank"] = np.arange(1, summary_df.shape[0] + 1)
    summary_df = summary_df[
        [
            "rank",
            "recommended",
            "budget_per_cluster",
            "total_subset_size",
            "feasible",
            "reason",
            "joint_pass_rate",
            "joint_pass_rate_lcb",
            "best_global_pass",
            "best_per_cluster_pass",
            "exp_abs_oc_psnr",
            "exp_abs_oc_ssim",
            "exp_abs_fi_psnr",
            "exp_abs_fi_ssim",
            "p95_abs_oc_psnr",
            "p95_abs_oc_ssim",
            "p95_abs_fi_psnr",
            "p95_abs_fi_ssim",
            "exp_runtime_mean",
            "p95_runtime_mean",
            "best_runtime_mean",
            "max_per_cluster_psnr_gap_best",
            "max_per_cluster_ssim_gap_best",
            "best_objective",
            "best_manifest",
        ]
    ]

    summary_csv = out_dir / args.summary_csv_name
    summary_json = out_dir / args.summary_json_name
    summary_md = out_dir / args.summary_md_name
    recommendation_json = out_dir / args.recommendation_json_name
    summary_df.to_csv(summary_csv, index=False)

    config = {
        "cluster_mapping_csv": str(args.cluster_mapping_csv),
        "oc_csv": str(args.oc_csv),
        "fi_csv": str(args.fi_csv),
        "runtime_csv": None if args.runtime_csv is None else str(args.runtime_csv),
        "runtime_available": bool(runtime_available),
        "split_csv": None if args.split_csv is None else str(args.split_csv),
        "eval_split": str(args.eval_split),
        "budgets": budgets,
        "num_simulations": int(args.num_simulations),
        "random_seed": int(args.random_seed),
        "max_abs_psnr_gap": float(args.max_abs_psnr_gap),
        "max_abs_ssim_gap": float(args.max_abs_ssim_gap),
        "max_per_cluster_psnr_gap": args.max_per_cluster_psnr_gap,
        "max_per_cluster_ssim_gap": args.max_per_cluster_ssim_gap,
        "confidence_level": float(args.confidence_level),
        "selection_use_lcb": bool(args.selection_use_lcb),
        "require_best_manifest_constraints": bool(args.require_best_manifest_constraints),
        "weight_runtime": float(args.weight_runtime),
        "target_joint_pass_rate": float(args.target_joint_pass_rate),
        "max_total_subset": args.max_total_subset,
        "n_clusters": int(n_clusters),
        "combined_rows": int(df.shape[0]),
        "min_cluster_size": int(min_cluster_size),
    }
    payload = {
        "config": config,
        "recommendation": {k: _to_jsonable(v) for k, v in recommendation.items()},
        "rows": [
            {
                k: _to_jsonable(v)
                for k, v in row.items()
            }
            for row in summary_df.to_dict(orient="records")
        ],
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)

    with recommendation_json.open("w", encoding="utf-8") as f:
        json.dump(
            {k: _to_jsonable(v) for k, v in recommendation.items()},
            f,
            indent=2,
            sort_keys=False,
        )

    _make_markdown(
        summary_md,
        summary_df.to_dict(orient="records"),
        config,
        recommendation,
    )

    LOGGER.info("Saved summary CSV: %s", summary_csv)
    LOGGER.info("Saved summary JSON: %s", summary_json)
    LOGGER.info("Saved recommendation JSON: %s", recommendation_json)
    LOGGER.info("Saved summary Markdown: %s", summary_md)
    LOGGER.info(
        "Recommended budget=%s | total_subset=%s | joint_pass_rate=%.4f | mode=%s",
        recommendation["recommended_budget_per_cluster"],
        recommendation["recommended_total_subset_size"],
        float(recommendation["recommended_joint_pass_rate"]),
        recommendation["selection_mode"],
    )


if __name__ == "__main__":
    main()
