#!/usr/bin/env python3
"""Refine a subset manifest with cluster-preserving metric-aware swaps."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("refine_subset_metric_matching")


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def _canonicalize_metric_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{source_name} metrics CSV is empty")
    cols = {_normalize_col_name(c): c for c in df.columns}
    dish_col = cols.get("dishid") or cols.get("sceneid") or cols.get("id")
    psnr_col = cols.get("psnr") or cols.get("pnsr")
    ssim_col = cols.get("ssim")
    if dish_col is None or psnr_col is None or ssim_col is None:
        raise ValueError(f"{source_name}: missing required columns in {list(df.columns)}")
    out = pd.DataFrame()
    out["dish_id"] = df[dish_col].astype(str)
    out[f"{source_name}_psnr"] = pd.to_numeric(df[psnr_col], errors="coerce")
    out[f"{source_name}_ssim"] = pd.to_numeric(df[ssim_col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def _load_runtime(runtime_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if runtime_csv is None:
        return None
    df = pd.read_csv(runtime_csv)
    if df.empty:
        return None
    cols = {_normalize_col_name(c): c for c in df.columns}
    dish_col = cols.get("dishid") or cols.get("sceneid") or cols.get("id")
    rt_col = (
        cols.get("runtimesec")
        or cols.get("runtime")
        or cols.get("seconds")
        or cols.get("timesec")
        or cols.get("elapsedsec")
    )
    if dish_col is None or rt_col is None:
        raise ValueError(f"runtime CSV missing id/runtime cols: {list(df.columns)}")
    out = pd.DataFrame()
    out["dish_id"] = df[dish_col].astype(str)
    out["runtime_sec"] = pd.to_numeric(df[rt_col], errors="coerce")
    out = out.dropna(subset=["runtime_sec"]).reset_index(drop=True)
    return out


def _prepare_combined(
    mapping_csv: Path, oc_csv: Path, fi_csv: Path, runtime_csv: Optional[Path]
) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_csv)
    if "dish_id" not in mapping.columns or "cluster" not in mapping.columns:
        raise ValueError("cluster mapping CSV must contain dish_id and cluster")
    mapping = mapping[["dish_id", "cluster"]].copy()
    mapping["dish_id"] = mapping["dish_id"].astype(str)
    mapping["cluster"] = pd.to_numeric(mapping["cluster"], errors="coerce")
    mapping = mapping.dropna(subset=["cluster"]).copy()
    mapping["cluster"] = mapping["cluster"].astype(int)

    oc = _canonicalize_metric_columns(pd.read_csv(oc_csv), "oc")
    fi = _canonicalize_metric_columns(pd.read_csv(fi_csv), "fi")

    df = mapping.merge(oc, on="dish_id", how="inner").merge(fi, on="dish_id", how="inner")
    runtime_df = _load_runtime(runtime_csv)
    if runtime_df is not None:
        df = df.merge(runtime_df, on="dish_id", how="left")
    if df.empty:
        raise RuntimeError("No overlap between mapping and OC/FI metrics")
    df = df.sort_values("dish_id", kind="mergesort").reset_index(drop=True)
    return df


def _compute_gap_table(df: pd.DataFrame, subset_ids: set[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for prefix in ("oc", "fi"):
        for metric in ("psnr", "ssim"):
            col = f"{prefix}_{metric}"
            full = pd.to_numeric(df[col], errors="coerce").dropna()
            sub = pd.to_numeric(df[df["dish_id"].isin(subset_ids)][col], errors="coerce").dropna()
            full_mean = float(full.mean())
            sub_mean = float(sub.mean())
            diff = sub_mean - full_mean
            rows.append(
                {
                    "csv": prefix,
                    "metric": metric.upper(),
                    "full_mean": full_mean,
                    "subset_mean": sub_mean,
                    "diff_subset_minus_full": diff,
                    "abs_diff": abs(diff),
                }
            )
    return pd.DataFrame(rows).sort_values(["csv", "metric"], kind="mergesort").reset_index(drop=True)


def _objective(
    sum_vec: np.ndarray,
    subset_size: int,
    full_mean_vec: np.ndarray,
    threshold_vec: np.ndarray,
    runtime_sum: float,
    full_runtime_mean: float,
    weight_runtime: float,
) -> float:
    mean_vec = sum_vec / float(subset_size)
    base = float(np.sum(np.abs(mean_vec - full_mean_vec) / threshold_vec))
    if weight_runtime > 0.0 and full_runtime_mean > 1e-12 and not np.isnan(runtime_sum):
        runtime_mean = runtime_sum / float(subset_size)
        base += float(weight_runtime) * max(0.0, runtime_mean / full_runtime_mean)
    return base


def _local_refine(
    df: pd.DataFrame,
    cluster_target_counts: Dict[int, int],
    init_ids: List[str],
    max_iter: int,
    candidate_eval_limit: int,
    random_seed: int,
    max_abs_psnr_gap: float,
    max_abs_ssim_gap: float,
    weight_runtime: float,
) -> Tuple[List[str], float]:
    metrics = ["oc_psnr", "oc_ssim", "fi_psnr", "fi_ssim"]
    full_mean = df[metrics].mean().to_numpy(dtype=float)
    threshold = np.array(
        [max_abs_psnr_gap, max_abs_ssim_gap, max_abs_psnr_gap, max_abs_ssim_gap],
        dtype=float,
    )

    work = df.set_index("dish_id", drop=False)
    cluster_to_ids = {
        int(c): work[work["cluster"] == int(c)]["dish_id"].astype(str).tolist()
        for c in sorted(cluster_target_counts.keys())
    }
    metric_vec_by_id = {
        dish_id: work.loc[dish_id, metrics].to_numpy(dtype=float)
        for dish_id in work.index.astype(str)
    }
    if "runtime_sec" in work.columns and work["runtime_sec"].notna().any():
        runtime_by_id = {
            dish_id: float(pd.to_numeric(work.loc[dish_id, "runtime_sec"], errors="coerce"))
            for dish_id in work.index.astype(str)
        }
        full_runtime_mean = float(pd.to_numeric(work["runtime_sec"], errors="coerce").mean())
    else:
        runtime_by_id = {dish_id: np.nan for dish_id in work.index.astype(str)}
        full_runtime_mean = np.nan

    rng = np.random.default_rng(random_seed)
    current = list(init_ids)
    current_set = set(current)
    subset_size = len(current)
    sum_vec = np.sum(np.stack([metric_vec_by_id[s] for s in current], axis=0), axis=0)
    runtime_sum = float(np.nansum([runtime_by_id[s] for s in current]))
    current_obj = _objective(
        sum_vec=sum_vec,
        subset_size=subset_size,
        full_mean_vec=full_mean,
        threshold_vec=threshold,
        runtime_sum=runtime_sum,
        full_runtime_mean=full_runtime_mean,
        weight_runtime=weight_runtime,
    )

    for _ in range(int(max_iter)):
        improved = False
        for c, target_count in sorted(cluster_target_counts.items()):
            selected_in_cluster = [s for s in current if int(work.loc[s, "cluster"]) == int(c)]
            if len(selected_in_cluster) != int(target_count):
                continue
            pool = [s for s in cluster_to_ids[int(c)] if s not in current_set]
            if candidate_eval_limit > 0 and len(pool) > candidate_eval_limit:
                idx = rng.choice(len(pool), size=int(candidate_eval_limit), replace=False)
                pool = [pool[i] for i in idx]

            for old_id in list(selected_in_cluster):
                if old_id not in current_set:
                    continue
                old_vec = metric_vec_by_id[old_id]
                old_rt = runtime_by_id[old_id]
                best_local_obj = current_obj
                best_local_id: Optional[str] = None
                best_local_vec: Optional[np.ndarray] = None
                best_local_rt = old_rt

                for cand in pool:
                    cand_vec = metric_vec_by_id[cand]
                    cand_rt = runtime_by_id[cand]
                    new_sum = sum_vec - old_vec + cand_vec
                    new_runtime = runtime_sum - old_rt + cand_rt
                    new_obj = _objective(
                        sum_vec=new_sum,
                        subset_size=subset_size,
                        full_mean_vec=full_mean,
                        threshold_vec=threshold,
                        runtime_sum=new_runtime,
                        full_runtime_mean=full_runtime_mean,
                        weight_runtime=weight_runtime,
                    )
                    if new_obj + 1e-12 < best_local_obj:
                        best_local_obj = new_obj
                        best_local_id = cand
                        best_local_vec = cand_vec
                        best_local_rt = cand_rt

                if best_local_id is not None and best_local_vec is not None:
                    idx_in_current = current.index(old_id)
                    current[idx_in_current] = best_local_id
                    current_set.remove(old_id)
                    current_set.add(best_local_id)
                    sum_vec = sum_vec - old_vec + best_local_vec
                    runtime_sum = runtime_sum - old_rt + best_local_rt
                    current_obj = best_local_obj
                    improved = True
        if not improved:
            break

    return current, float(current_obj)


def _write_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Subset Refinement Report",
        "",
        f"- Input manifest: `{payload['input_manifest']}`",
        f"- Output manifest: `{payload['output_manifest']}`",
        f"- Subset size: **{payload['subset_size']}**",
        f"- Objective before: **{payload['objective_before']:.6f}**",
        f"- Objective after: **{payload['objective_after']:.6f}**",
        "",
        "## Global Gaps Before",
        "",
        "| csv | metric | full_mean | subset_mean | diff | abs_diff |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["global_gaps_before"]:
        lines.append(
            f"| {row['csv']} | {row['metric']} | {float(row['full_mean']):.6f} | "
            f"{float(row['subset_mean']):.6f} | {float(row['diff_subset_minus_full']):.6f} | {float(row['abs_diff']):.6f} |"
        )
    lines += [
        "",
        "## Global Gaps After",
        "",
        "| csv | metric | full_mean | subset_mean | diff | abs_diff |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["global_gaps_after"]:
        lines.append(
            f"| {row['csv']} | {row['metric']} | {float(row['full_mean']):.6f} | "
            f"{float(row['subset_mean']):.6f} | {float(row['diff_subset_minus_full']):.6f} | {float(row['abs_diff']):.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine subset manifest via cluster-preserving metric-aware swaps."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--cluster-mapping-csv", type=Path, required=True)
    parser.add_argument("--oc-csv", type=Path, default=Path("ingp_oc.csv"))
    parser.add_argument("--fi-csv", type=Path, default=Path("ingp_fi.csv"))
    parser.add_argument("--runtime-csv", type=Path, default=None)
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--candidate-eval-limit", type=int, default=150)
    parser.add_argument("--random-restarts", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-abs-psnr-gap", type=float, default=0.5)
    parser.add_argument("--max-abs-ssim-gap", type=float, default=0.01)
    parser.add_argument("--weight-runtime", type=float, default=0.0)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("subset_refined_manifest.csv"),
    )
    parser.add_argument(
        "--output-report-json",
        type=Path,
        default=Path("subset_refinement_report.json"),
    )
    parser.add_argument(
        "--output-report-md",
        type=Path,
        default=Path("subset_refinement_report.md"),
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

    df = _prepare_combined(
        mapping_csv=args.cluster_mapping_csv,
        oc_csv=args.oc_csv,
        fi_csv=args.fi_csv,
        runtime_csv=args.runtime_csv,
    )

    in_df = pd.read_csv(args.input_manifest)
    if "dish_id" not in in_df.columns:
        raise ValueError("input manifest must contain dish_id")
    init_ids = [str(x) for x in in_df["dish_id"].astype(str).tolist()]
    if not init_ids:
        raise ValueError("input manifest is empty")

    df_ids = set(df["dish_id"].astype(str))
    init_ids = [x for x in init_ids if x in df_ids]
    if not init_ids:
        raise RuntimeError("No input manifest rows overlap with combined metric table")

    map_df = pd.read_csv(args.cluster_mapping_csv)
    map_df["dish_id"] = map_df["dish_id"].astype(str)
    cluster_counts = (
        map_df[map_df["dish_id"].isin(init_ids)]
        .groupby("cluster")
        .size()
        .sort_index(kind="mergesort")
    )
    cluster_target_counts = {int(k): int(v) for k, v in cluster_counts.items()}

    rng = np.random.default_rng(int(args.random_seed))
    all_by_cluster = {
        int(c): df[df["cluster"] == int(c)]["dish_id"].astype(str).tolist()
        for c in sorted(cluster_target_counts.keys())
    }

    candidates: List[Tuple[List[str], float]] = []
    # restart 0: current manifest
    refined0, obj0 = _local_refine(
        df=df,
        cluster_target_counts=cluster_target_counts,
        init_ids=init_ids,
        max_iter=int(args.max_iter),
        candidate_eval_limit=int(args.candidate_eval_limit),
        random_seed=int(args.random_seed),
        max_abs_psnr_gap=float(args.max_abs_psnr_gap),
        max_abs_ssim_gap=float(args.max_abs_ssim_gap),
        weight_runtime=float(args.weight_runtime),
    )
    candidates.append((refined0, obj0))

    for r in range(int(args.random_restarts)):
        random_ids: List[str] = []
        for c, n in sorted(cluster_target_counts.items()):
            ids = all_by_cluster[int(c)]
            pick = rng.choice(ids, size=int(n), replace=False)
            random_ids.extend([str(x) for x in pick])
        refined, obj = _local_refine(
            df=df,
            cluster_target_counts=cluster_target_counts,
            init_ids=random_ids,
            max_iter=int(args.max_iter),
            candidate_eval_limit=int(args.candidate_eval_limit),
            random_seed=int(args.random_seed) + 101 * (r + 1),
            max_abs_psnr_gap=float(args.max_abs_psnr_gap),
            max_abs_ssim_gap=float(args.max_abs_ssim_gap),
            weight_runtime=float(args.weight_runtime),
        )
        candidates.append((refined, obj))

    best_ids, best_obj = min(candidates, key=lambda x: x[1])
    before_gap = _compute_gap_table(df, set(init_ids))
    after_gap = _compute_gap_table(df, set(best_ids))

    out_manifest = df[df["dish_id"].isin(best_ids)][["dish_id", "cluster"]].copy()
    out_manifest = out_manifest.sort_values(["cluster", "dish_id"], kind="mergesort").reset_index(drop=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report_md.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.to_csv(args.output_manifest, index=False)

    payload: Dict[str, object] = {
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "subset_size": int(len(best_ids)),
        "objective_before": float(obj0),
        "objective_after": float(best_obj),
        "random_restarts": int(args.random_restarts),
        "max_iter": int(args.max_iter),
        "candidate_eval_limit": int(args.candidate_eval_limit),
        "global_gaps_before": before_gap.to_dict(orient="records"),
        "global_gaps_after": after_gap.to_dict(orient="records"),
    }

    with args.output_report_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    _write_markdown(args.output_report_md, payload)

    LOGGER.info("Saved refined manifest: %s", args.output_manifest)
    LOGGER.info("Saved refinement report JSON: %s", args.output_report_json)
    LOGGER.info("Saved refinement report Markdown: %s", args.output_report_md)
    LOGGER.info("Objective before=%.6f | after=%.6f", float(obj0), float(best_obj))


if __name__ == "__main__":
    main()

