#!/usr/bin/env python3
"""Generate strong-accept empirical results for the BASS manuscript.

This script aims to recover the highest-value results that can be reproduced
from the repository's current assets:
1) Larger-budget reliability frontier for balanced BASS-style candidates
2) Failure-mode comparison at 48 scenes
3) Threshold-sensitivity table
4) Distributional fidelity plots for the exported BASS-48 subset
5) Expanded cross-method ranking table, when per-scene method logs exist

The script is conservative by design:
- it never overwrites raw source files
- it uses deterministic scene-ID ordering
- it writes a sanity report that explicitly records anything not generated
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp, spearmanr


LOGGER = logging.getLogger("strong_accept_results")

DEFAULT_K = 6
DEFAULT_P_MIN = 0.08
DEFAULT_BUDGETS_PER_CLUSTER = [8, 10, 12, 16, 20]
DEFAULT_BASELINE_CONFIG = Path("sweep_cluster_k/baseline_comparison_lpips_ks/baseline_eval_config.json")
DEFAULT_CANDIDATE_ROOT = Path("sweep_cluster_k/budget_sweep_k6_auto_v2")
DEFAULT_K6_MAPPING = Path("sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv")
DEFAULT_DESCRIPTOR_CANDIDATES = [
    Path("sweep_cluster_k/k_6/feats_normalized.csv"),
    Path("feats_normalized.csv"),
]
DEFAULT_ZIPNERF_CANDIDATES = [
    Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx"),
    Path("zipnerf_metrics.csv"),
]
DEFAULT_SUBSET_CANDIDATES = [
    DEFAULT_CANDIDATE_ROOT / "recommended_subset.csv",
    Path("sweep_cluster_k/budget_sweep_k6_auto/recommended_subset.csv"),
    Path("sweep_cluster_k/holdout_protocol_v3/joint_selection/recommended_subset.csv"),
]
DEFAULT_METHOD_SEARCH: Dict[str, List[Path]] = {
    "Instant-NGP": [
        Path("ingp_fi.csv"),
        Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/instant-ngp_nerf.xlsx"),
    ],
    "Zip-NeRF": [
        Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx"),
        Path("zipnerf_metrics.csv"),
    ],
    "Feature-Splatting": [
        Path("sweep_cluster_k/holdout_protocol_v2/joint_selection/feature_splatting.csv"),
    ],
    "Nerfacto": [],
    "TensoRF": [],
    "DVGO": [],
    "Plenoxels": [],
    "3DGS": [],
    "Mip-NeRF": [],
}
THRESHOLD_SETTINGS: Dict[str, Dict[str, float]] = {
    "strict": {
        "psnr_tol": 0.25,
        "ssim_tol": 0.005,
        "lpips_tol": 0.005,
        "ks_tol": 0.10,
    },
    "default": {
        "psnr_tol": 0.50,
        "ssim_tol": 0.010,
        "lpips_tol": 0.010,
        "ks_tol": 0.14,
    },
    "relaxed": {
        "psnr_tol": 0.75,
        "ssim_tol": 0.015,
        "lpips_tol": 0.015,
        "ks_tol": 0.18,
    },
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: Path
    sheet_name: Optional[str] = None


@dataclass
class MethodTable:
    name: str
    path: Path
    df: pd.DataFrame


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def _pick_first(norm_to_orig: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        if cand in norm_to_orig:
            return norm_to_orig[cand]
    return None


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _wilson_lower_bound(p_hat: float, n: int, confidence_level: float = 0.95) -> float:
    if n <= 0:
        return 0.0
    z = float(NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0))
    n_f = float(n)
    denom = 1.0 + (z * z) / n_f
    center = (p_hat + (z * z) / (2.0 * n_f)) / denom
    margin = (
        z
        * np.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n_f)) / n_f)
        / denom
    )
    return float(max(0.0, center - margin))


def _latex_escape(text: str) -> str:
    out = str(text)
    out = out.replace("\\", "\\textbackslash{}")
    out = out.replace("_", "\\_")
    out = out.replace("%", "\\%")
    out = out.replace("&", "\\&")
    out = out.replace("#", "\\#")
    return out


def _path_exists(path: Optional[Path]) -> bool:
    return path is not None and path.exists()


def _detect_candidate_root(repo_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return _resolve_path(str(explicit), repo_root)
    for cand in (DEFAULT_CANDIDATE_ROOT, Path("sweep_cluster_k/budget_sweep_k6_auto")):
        path = (repo_root / cand).resolve()
        if path.exists():
            return path
    return None


def _detect_descriptors(repo_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return _resolve_path(str(explicit), repo_root)
    for cand in DEFAULT_DESCRIPTOR_CANDIDATES:
        path = (repo_root / cand).resolve()
        if path.exists():
            return path
    return None


def _detect_zipnerf_log(repo_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return _resolve_path(str(explicit), repo_root)
    for cand in DEFAULT_ZIPNERF_CANDIDATES:
        path = (repo_root / cand).resolve()
        if path.exists():
            return path
    return None


def _detect_subset(repo_root: Path, explicit: Optional[Path], candidate_root: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return _resolve_path(str(explicit), repo_root)
    if candidate_root is not None:
        cand = (candidate_root / "recommended_subset.csv").resolve()
        if cand.exists():
            return cand
    for cand in DEFAULT_SUBSET_CANDIDATES:
        path = (repo_root / cand).resolve()
        if path.exists():
            return path
    return None


def _detect_k6_mapping(repo_root: Path, candidate_root: Optional[Path]) -> Optional[Path]:
    if candidate_root is not None:
        summary_json = candidate_root / "budget_sweep_summary.json"
        if summary_json.exists():
            payload = _load_json(summary_json)
            config = payload.get("config", {})
            if isinstance(config, dict) and "cluster_mapping_csv" in config:
                path = _resolve_path(str(config["cluster_mapping_csv"]), repo_root)
                if path.exists():
                    return path
    path = (repo_root / DEFAULT_K6_MAPPING).resolve()
    return path if path.exists() else None


def _read_table(path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        if sheet_name is not None:
            return pd.read_excel(path, sheet_name=sheet_name)
        workbook = pd.ExcelFile(path)
        if not workbook.sheet_names:
            raise ValueError(f"No sheets found in workbook: {path}")
        if "zipnerf" in workbook.sheet_names:
            return workbook.parse("zipnerf")
        return workbook.parse(workbook.sheet_names[0])
    raise ValueError(f"Unsupported table type: {path}")


def _load_mapping_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in raw.columns}
    id_col = _pick_first(norm_to_orig, ("dishid", "sceneid", "id"))
    cluster_col = _pick_first(norm_to_orig, ("cluster", "regime"))
    if id_col is None or cluster_col is None:
        raise ValueError(f"{path}: expected dish ID and cluster columns")
    out = pd.DataFrame()
    out["dish_id"] = raw[id_col].astype(str).str.strip()
    out["cluster"] = pd.to_numeric(raw[cluster_col], errors="coerce")
    out = out.dropna(subset=["dish_id", "cluster"]).copy()
    out = out[out["dish_id"] != ""].copy()
    out["cluster"] = out["cluster"].astype(int)
    out = out.drop_duplicates(subset=["dish_id"], keep="first").reset_index(drop=True)
    return out


def _load_zipnerf_metrics(path: Path) -> pd.DataFrame:
    raw = _read_table(path, sheet_name=None)
    norm_to_orig = {_normalize_col_name(c): c for c in raw.columns}
    id_col = _pick_first(norm_to_orig, ("experimentname", "dishid", "sceneid", "id"))
    psnr_col = _pick_first(norm_to_orig, ("psnr", "psnravgfrommse", "psnravgmse"))
    ssim_col = _pick_first(norm_to_orig, ("ssim",))
    lpips_col = _pick_first(norm_to_orig, ("lpips",))
    if id_col is None or psnr_col is None or ssim_col is None:
        raise ValueError(f"{path}: unable to detect Zip-NeRF ID / PSNR / SSIM columns")
    out = pd.DataFrame()
    out["dish_id"] = raw[id_col].astype(str).str.strip()
    out["psnr"] = pd.to_numeric(raw[psnr_col], errors="coerce")
    out["ssim"] = pd.to_numeric(raw[ssim_col], errors="coerce")
    out["lpips"] = (
        pd.to_numeric(raw[lpips_col], errors="coerce")
        if lpips_col is not None
        else np.nan
    )
    out = out.dropna(subset=["dish_id", "psnr", "ssim"]).copy()
    out = out[out["dish_id"] != ""].copy()
    out = (
        out.groupby("dish_id", as_index=False)[["psnr", "ssim", "lpips"]]
        .mean(numeric_only=True)
        .sort_values("dish_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def _load_subset_ids(path: Path) -> List[str]:
    raw = pd.read_csv(path)
    norm_to_orig = {_normalize_col_name(c): c for c in raw.columns}
    id_col = _pick_first(norm_to_orig, ("dishid", "experimentname", "sceneid", "id"))
    if id_col is None:
        raise ValueError(f"{path}: unable to find subset identifier column")
    ids = sorted(set(raw[id_col].astype(str).str.strip()) - {""})
    if not ids:
        raise ValueError(f"{path}: subset manifest did not contain any scene IDs")
    return ids


def _canonicalize_method_table(spec: MethodSpec) -> MethodTable:
    raw = _read_table(spec.path, spec.sheet_name)
    if raw.empty:
        raise ValueError(f"{spec.name}: metrics table is empty")

    norm_to_orig = {_normalize_col_name(c): c for c in raw.columns}
    id_col = _pick_first(norm_to_orig, ("dishid", "experimentname", "sceneid", "scene", "id"))
    psnr_col = _pick_first(
        norm_to_orig,
        ("psnravgfrommse", "psnravgmse", "psnr", "pnsr", "meanpsnr"),
    )
    ssim_col = _pick_first(norm_to_orig, ("ssim", "ssimmean", "meanssim"))
    lpips_col = _pick_first(norm_to_orig, ("lpips", "lpipsmean", "meanlpips"))

    if id_col is None or psnr_col is None or ssim_col is None:
        raise ValueError(f"{spec.name}: missing required ID / PSNR / SSIM columns")

    out = pd.DataFrame()
    out["scene_id"] = raw[id_col].astype(str).str.strip()
    out["psnr"] = pd.to_numeric(raw[psnr_col], errors="coerce")
    out["ssim"] = pd.to_numeric(raw[ssim_col], errors="coerce")
    out["lpips"] = (
        pd.to_numeric(raw[lpips_col], errors="coerce")
        if lpips_col is not None
        else np.nan
    )
    out = out[out["scene_id"] != ""].copy()
    out = (
        out.groupby("scene_id", as_index=False)[["psnr", "ssim", "lpips"]]
        .mean(numeric_only=True)
        .sort_values("scene_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return MethodTable(name=spec.name, path=spec.path, df=out)


def _detect_sheet_name(path: Path) -> Optional[str]:
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        return None
    workbook = pd.ExcelFile(path)
    if "zipnerf" in workbook.sheet_names:
        return "zipnerf"
    return workbook.sheet_names[0] if workbook.sheet_names else None


def _infer_method_name(path: Path) -> str:
    stem = path.stem.lower()
    if "ingp" in stem or "instant-ngp" in stem or "instantngp" in stem:
        return "Instant-NGP"
    if "zipnerf" in stem:
        return "Zip-NeRF"
    if "feature" in stem and "splat" in stem:
        return "Feature-Splatting"
    if "nerfacto" in stem:
        return "Nerfacto"
    if "tensorrf" in stem:
        return "TensoRF"
    if "dvgo" in stem or "directvoxgo" in stem:
        return "DVGO"
    if "plenoxel" in stem:
        return "Plenoxels"
    if "3dgs" in stem or "gaussian" in stem:
        return "3DGS"
    if "mipnerf" in stem or "mip-nerf" in stem:
        return "Mip-NeRF"
    return path.stem


def _discover_method_specs(repo_root: Path, explicit_paths: Sequence[Path]) -> Tuple[List[MethodSpec], List[str], List[str]]:
    missing_candidates: List[str] = []
    discovered: List[MethodSpec] = []
    seen_names: set[str] = set()

    if explicit_paths:
        for raw in explicit_paths:
            path = _resolve_path(str(raw), repo_root)
            if not path.exists():
                missing_candidates.append(str(path))
                continue
            name = _infer_method_name(path)
            if name in seen_names:
                continue
            discovered.append(MethodSpec(name=name, path=path, sheet_name=_detect_sheet_name(path)))
            seen_names.add(name)
        return discovered, missing_candidates, []

    searched_notes: List[str] = []
    for method_name, candidates in DEFAULT_METHOD_SEARCH.items():
        found = False
        for cand in candidates:
            path = (repo_root / cand).resolve()
            searched_notes.append(str(cand))
            if path.exists():
                discovered.append(MethodSpec(name=method_name, path=path, sheet_name=_detect_sheet_name(path)))
                seen_names.add(method_name)
                found = True
                break
        if found or candidates:
            continue

        search_hits = sorted(
            p
            for p in repo_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
            and method_name.lower().replace("-", "") in p.stem.lower().replace("-", "")
        )
        if search_hits:
            path = search_hits[0]
            discovered.append(MethodSpec(name=method_name, path=path, sheet_name=_detect_sheet_name(path)))
            seen_names.add(method_name)
        else:
            missing_candidates.append(method_name)

    return discovered, [], missing_candidates


def _merge_mapping_and_metrics(mapping_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = mapping_df.merge(metrics_df, on="dish_id", how="inner")
    df = df.dropna(subset=["psnr", "ssim", "lpips"]).copy()
    df = df.sort_values("dish_id", kind="mergesort").reset_index(drop=True)
    return df


def _metric_eval_for_ids(
    df: pd.DataFrame,
    subset_ids: Sequence[str],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    subset_set = set(str(v) for v in subset_ids)
    subset = df[df["dish_id"].isin(subset_set)].copy()
    if subset.empty:
        raise ValueError("Subset has no overlap with evaluation dataframe")

    full_vals = {
        "psnr": pd.to_numeric(df["psnr"], errors="coerce").dropna().to_numpy(dtype=float),
        "ssim": pd.to_numeric(df["ssim"], errors="coerce").dropna().to_numpy(dtype=float),
        "lpips": pd.to_numeric(df["lpips"], errors="coerce").dropna().to_numpy(dtype=float),
    }
    subset_vals = {
        "psnr": pd.to_numeric(subset["psnr"], errors="coerce").dropna().to_numpy(dtype=float),
        "ssim": pd.to_numeric(subset["ssim"], errors="coerce").dropna().to_numpy(dtype=float),
        "lpips": pd.to_numeric(subset["lpips"], errors="coerce").dropna().to_numpy(dtype=float),
    }

    psnr_gap = float(np.mean(subset_vals["psnr"]) - np.mean(full_vals["psnr"]))
    ssim_gap = float(np.mean(subset_vals["ssim"]) - np.mean(full_vals["ssim"]))
    lpips_gap = float(np.mean(subset_vals["lpips"]) - np.mean(full_vals["lpips"]))
    abs_psnr_gap = float(abs(psnr_gap))
    abs_ssim_gap = float(abs(ssim_gap))
    abs_lpips_gap = float(abs(lpips_gap))
    ks_psnr = float(ks_2samp(full_vals["psnr"], subset_vals["psnr"]).statistic)
    ks_ssim = float(ks_2samp(full_vals["ssim"], subset_vals["ssim"]).statistic)
    ks_lpips = float(ks_2samp(full_vals["lpips"], subset_vals["lpips"]).statistic)
    max_ks = float(max(ks_psnr, ks_ssim, ks_lpips))

    mean_pass = (
        abs_psnr_gap <= thresholds["psnr_tol"]
        and abs_ssim_gap <= thresholds["ssim_tol"]
        and abs_lpips_gap <= thresholds["lpips_tol"]
    )
    ks_pass = max_ks <= thresholds["ks_tol"]
    joint_pass = bool(mean_pass and ks_pass)

    mean_objective = float(
        max(
            abs_psnr_gap / thresholds["psnr_tol"],
            abs_ssim_gap / thresholds["ssim_tol"],
            abs_lpips_gap / thresholds["lpips_tol"],
        )
    )
    joint_objective = float(max(mean_objective, max_ks / thresholds["ks_tol"]))

    return {
        "subset_size": int(len(subset_set)),
        "evaluated_subset_size": int(subset["dish_id"].nunique()),
        "psnr_gap": psnr_gap,
        "ssim_gap": ssim_gap,
        "lpips_gap": lpips_gap,
        "abs_psnr_gap": abs_psnr_gap,
        "abs_ssim_gap": abs_ssim_gap,
        "abs_lpips_gap": abs_lpips_gap,
        "ks_psnr": ks_psnr,
        "ks_ssim": ks_ssim,
        "ks_lpips": ks_lpips,
        "max_ks": max_ks,
        "mean_constraints_pass": bool(mean_pass),
        "ks_pass": bool(ks_pass),
        "joint_pass": bool(joint_pass),
        "mean_objective": mean_objective,
        "joint_objective": joint_objective,
    }


def _subset_manifest_df(mapping_df: pd.DataFrame, subset_ids: Sequence[str]) -> pd.DataFrame:
    subset_set = set(str(v) for v in subset_ids)
    out = mapping_df[mapping_df["dish_id"].isin(subset_set)][["dish_id", "cluster"]].copy()
    out = out.sort_values(["cluster", "dish_id"], kind="mergesort").reset_index(drop=True)
    return out


def _save_subset_manifest(path: Path, mapping_df: pd.DataFrame, subset_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _subset_manifest_df(mapping_df, subset_ids).to_csv(path, index=False)


def _simulate_trials(
    df: pd.DataFrame,
    groups: Dict[int, np.ndarray],
    budget_per_cluster: int,
    num_trials: int,
    seed: int,
    mode: str,
    thresholds: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    if mode not in {"balanced", "uniform"}:
        raise ValueError(f"Unsupported mode: {mode}")

    rng = np.random.default_rng(seed + (10000 if mode == "uniform" else 0) + int(budget_per_cluster))
    dish_ids = df["dish_id"].astype(str).to_numpy()
    total_subset_size = int(len(groups) * int(budget_per_cluster))
    all_indices = np.arange(df.shape[0], dtype=int)

    rows: List[Dict[str, object]] = []
    best_joint: Optional[Dict[str, object]] = None
    best_mean: Optional[Dict[str, object]] = None

    for trial in range(int(num_trials)):
        if mode == "balanced":
            picks = [
                rng.choice(groups[cluster], size=int(budget_per_cluster), replace=False)
                for cluster in sorted(groups)
            ]
            idx = np.sort(np.concatenate(picks))
        else:
            idx = np.sort(rng.choice(all_indices, size=total_subset_size, replace=False))

        subset_ids = dish_ids[idx].tolist()
        stats = _metric_eval_for_ids(df, subset_ids, thresholds)
        row = {
            "trial": int(trial),
            "mode": mode,
            "b": int(budget_per_cluster),
            "budget_scenes": int(total_subset_size),
            **stats,
        }
        rows.append(row)

        joint_key = (
            float(row["joint_objective"]),
            float(row["mean_objective"]),
            int(trial),
        )
        mean_key = (
            float(row["mean_objective"]),
            float(row["joint_objective"]),
            int(trial),
        )
        if best_joint is None or joint_key < best_joint["key"]:
            best_joint = {
                "key": joint_key,
                "subset_ids": subset_ids,
                "stats": dict(row),
            }
        if best_mean is None or mean_key < best_mean["key"]:
            best_mean = {
                "key": mean_key,
                "subset_ids": subset_ids,
                "stats": dict(row),
            }

    trial_df = pd.DataFrame(rows).sort_values(["budget_scenes", "trial"], kind="mergesort").reset_index(drop=True)
    if best_joint is None or best_mean is None:
        raise RuntimeError("Simulation did not produce any candidates")
    return trial_df, best_joint, best_mean


def _build_groups(mapping_metrics_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    groups: Dict[int, np.ndarray] = {}
    for cluster in sorted(mapping_metrics_df["cluster"].unique().tolist()):
        groups[int(cluster)] = mapping_metrics_df.index[mapping_metrics_df["cluster"] == int(cluster)].to_numpy(dtype=int)
    return groups


def _recommend_budget(
    balanced_trials_by_b: Dict[int, pd.DataFrame],
    thresholds: Dict[str, float],
    p_min: float,
    confidence_level: float,
) -> Tuple[str, float, str]:
    for b in sorted(balanced_trials_by_b):
        trial_df = balanced_trials_by_b[b]
        pass_mask = (
            (trial_df["abs_psnr_gap"] <= thresholds["psnr_tol"])
            & (trial_df["abs_ssim_gap"] <= thresholds["ssim_tol"])
            & (trial_df["abs_lpips_gap"] <= thresholds["lpips_tol"])
            & (trial_df["max_ks"] <= thresholds["ks_tol"])
        )
        n_trials = int(trial_df.shape[0])
        n_pass = int(pass_mask.sum())
        p_hat = float(n_pass / n_trials) if n_trials > 0 else float("nan")
        lcb = _wilson_lower_bound(p_hat, n_trials, confidence_level)
        budget_scenes = int(trial_df["budget_scenes"].iloc[0])
        if lcb >= float(p_min):
            return str(budget_scenes), float(lcb), "target met"
    max_budget = max(int(df["budget_scenes"].iloc[0]) for df in balanced_trials_by_b.values())
    return f"not reached up to {max_budget} scenes", float("nan"), "target not reached"


def _top_ranking_signature(score_map: Dict[str, float], higher_is_better: bool) -> Tuple[str, ...]:
    ordered = sorted(
        score_map.items(),
        key=lambda kv: (-kv[1], kv[0]) if higher_is_better else (kv[1], kv[0]),
    )
    return tuple(name for name, _ in ordered)


def _compute_ranking_preserved(
    method_tables: Sequence[MethodTable],
    subset_ids: Sequence[str],
) -> Optional[bool]:
    if len(method_tables) < 2:
        return None
    shared_ids = set(str(v) for v in subset_ids)
    for table in method_tables:
        shared_ids &= set(table.df["scene_id"].astype(str))
    if len(shared_ids) < 2:
        return None

    subset_shared = set(str(v) for v in subset_ids) & shared_ids
    if not subset_shared:
        return None

    metrics = [("psnr", True), ("ssim", True)]
    preserved_all = True
    for metric_name, higher_is_better in metrics:
        full_scores: Dict[str, float] = {}
        subset_scores: Dict[str, float] = {}
        for table in method_tables:
            full_scores[table.name] = float(
                table.df[table.df["scene_id"].isin(shared_ids)][metric_name].mean()
            )
            subset_scores[table.name] = float(
                table.df[table.df["scene_id"].isin(subset_shared)][metric_name].mean()
            )
        if _top_ranking_signature(full_scores, higher_is_better) != _top_ranking_signature(
            subset_scores, higher_is_better
        ):
            preserved_all = False
    return bool(preserved_all)


def _write_latex_table(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_reliability_frontier_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{rrrrrrrrr}",
        "\\hline",
        "Scenes & $k$ & $b$ & Trials & Passes & Pass rate & Wilson LCB & $|\\Delta|$ LPIPS & Max KS \\\\",
        "\\hline",
    ]
    for row in df.to_dict("records"):
        lines.append(
            "{} & {} & {} & {} & {} & {:.3f} & {:.3f} & {:.4f} & {:.3f} \\\\".format(
                int(row["budget_scenes"]),
                int(row["k"]),
                int(row["b"]),
                int(row["n_trials"]),
                int(row["n_pass"]),
                float(row["empirical_pass_rate"]),
                float(row["wilson_lcb_95"]),
                float(row["lpips_mean_gap"]),
                float(row["max_ks"]),
            )
        )
    lines += ["\\hline", "\\end{tabular}"]
    _write_latex_table(path, lines)


def _make_failure_mode_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Method & Mean-pass & KS-pass & Joint-pass & Wilson LCB & Max KS & Rank preserved \\\\",
        "\\hline",
    ]
    for row in df.to_dict("records"):
        rank_txt = (
            "Yes" if bool(row["ranking_preserved"]) else "No"
        ) if not pd.isna(row["ranking_preserved"]) else "--"
        mean_txt = "--" if pd.isna(row["mean_constraints_pass_rate"]) else f"{float(row['mean_constraints_pass_rate']):.3f}"
        ks_txt = "--" if pd.isna(row["ks_pass_rate"]) else f"{float(row['ks_pass_rate']):.3f}"
        joint_txt = "--" if pd.isna(row["joint_pass_rate"]) else f"{float(row['joint_pass_rate']):.3f}"
        lcb_txt = "--" if pd.isna(row["wilson_lcb_95"]) else f"{float(row['wilson_lcb_95']):.3f}"
        max_ks_txt = "--" if pd.isna(row["exported_or_best_subset_max_ks"]) else f"{float(row['exported_or_best_subset_max_ks']):.3f}"
        lines.append(
            f"{_latex_escape(row['method'])} & {mean_txt} & {ks_txt} & {joint_txt} & {lcb_txt} & {max_ks_txt} & {rank_txt} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}"]
    _write_latex_table(path, lines)


def _make_threshold_sensitivity_table(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\hline",
        "Setting & PSNR tol. & SSIM tol. & LPIPS tol. & KS tol. & Recommendation \\\\",
        "\\hline",
    ]
    for row in df.to_dict("records"):
        rec = _latex_escape(str(row["recommended_budget"]))
        lines.append(
            "{} & {:.2f} & {:.3f} & {:.3f} & {:.2f} & {} \\\\".format(
                _latex_escape(row["setting"]),
                float(row["psnr_tol"]),
                float(row["ssim_tol"]),
                float(row["lpips_tol"]),
                float(row["ks_tol"]),
                rec,
            )
        )
    lines += ["\\hline", "\\end{tabular}"]
    _write_latex_table(path, lines)


def _make_expanded_ranking_table(
    path: Path,
    method_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\hline",
        "Method & Full PSNR & BASS-48 PSNR & Full rank & Subset rank & Full SSIM & BASS-48 SSIM & Full rank & Subset rank \\\\",
        "\\hline",
    ]
    for row in method_df.to_dict("records"):
        lines.append(
            "{} & {:.3f} & {:.3f} & {} & {} & {:.4f} & {:.4f} & {} & {} \\\\".format(
                _latex_escape(row["method"]),
                float(row["full_psnr"]),
                float(row["bass48_psnr"]),
                int(row["full_psnr_rank"]),
                int(row["bass48_psnr_rank"]),
                float(row["full_ssim"]),
                float(row["bass48_ssim"]),
                int(row["full_ssim_rank"]),
                int(row["bass48_ssim_rank"]),
            )
        )
    lines += ["\\hline"]
    for row in corr_df.to_dict("records"):
        lines.append(
            "\\multicolumn{9}{l}{%s: Spearman $\\rho$ = %.3f, Kendall $\\tau$ = %.3f} \\\\" % (
                _latex_escape(str(row["metric"]).upper()),
                float(row["spearman_rho"]),
                float(row["kendall_tau"]),
            )
        )
    lines += ["\\hline", "\\end{tabular}"]
    _write_latex_table(path, lines)


def _save_reliability_figure(
    df: pd.DataFrame,
    out_pdf: Path,
    p_min: float,
) -> None:
    plot_df = df.sort_values("budget_scenes", kind="mergesort").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.1, 3.9))
    ax.plot(
        plot_df["budget_scenes"],
        plot_df["empirical_pass_rate"],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="Empirical pass rate",
    )
    ax.plot(
        plot_df["budget_scenes"],
        plot_df["wilson_lcb_95"],
        marker="s",
        linewidth=2.0,
        color="#d62728",
        label="Wilson LCB (95%)",
    )
    ax.axhline(float(p_min), linestyle="--", color="#555555", linewidth=1.1, label="$p_{\\min}=0.08$")
    ax.set_xlabel("Subset size")
    ax.set_ylabel("Pass probability")
    ax.set_title("Cost-Reliability Frontier for RASS")
    ax.set_xticks(plot_df["budget_scenes"].tolist())
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.14), frameon=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _ecdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.sort(values.astype(float))
    y = np.arange(1, vals.shape[0] + 1, dtype=float) / float(vals.shape[0])
    return vals, y


def _save_distribution_figure(
    full_df: pd.DataFrame,
    subset_ids: Sequence[str],
    out_pdf: Path,
) -> pd.DataFrame:
    subset_set = set(str(v) for v in subset_ids)
    subset_df = full_df[full_df["dish_id"].isin(subset_set)].copy()
    metrics = [
        ("psnr", "PSNR", "#1f77b4"),
        ("ssim", "SSIM", "#2ca02c"),
        ("lpips", "LPIPS", "#d62728"),
    ]
    rows: List[Dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    for ax, (metric_key, metric_label, color) in zip(axes, metrics):
        full_vals = pd.to_numeric(full_df[metric_key], errors="coerce").dropna().to_numpy(dtype=float)
        subset_vals = pd.to_numeric(subset_df[metric_key], errors="coerce").dropna().to_numpy(dtype=float)
        x_full, y_full = _ecdf_xy(full_vals)
        x_sub, y_sub = _ecdf_xy(subset_vals)
        ks_val = float(ks_2samp(full_vals, subset_vals).statistic)
        ax.step(x_full, y_full, where="post", color="#444444", linewidth=1.8, label="Full")
        ax.step(x_sub, y_sub, where="post", color=color, linewidth=1.8, label="BASS-48")
        ax.set_title(metric_label)
        ax.set_xlabel(metric_label)
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.25)
        ax.text(
            0.98,
            0.06,
            f"KS={ks_val:.3f}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=10,
        )
        rows.append(
            {
                "metric": metric_label,
                "full_count": int(full_vals.shape[0]),
                "subset_count": int(subset_vals.shape[0]),
                "full_mean": float(np.mean(full_vals)),
                "subset_mean": float(np.mean(subset_vals)),
                "abs_mean_gap": float(abs(float(np.mean(subset_vals) - np.mean(full_vals)))),
                "ks_distance": ks_val,
            }
        )
    axes[0].legend(loc="lower right")
    fig.suptitle("Distributional Fidelity of the Exported BASS-48 Subset", fontsize=12)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _metric_rank_correlation(
    method_df: pd.DataFrame,
    metric: str,
) -> Tuple[float, float]:
    full_col = f"full_{metric}_rank"
    subset_col = f"bass48_{metric}_rank"
    rho = spearmanr(method_df[full_col], method_df[subset_col]).statistic
    tau = kendalltau(method_df[full_col], method_df[subset_col]).statistic
    return float(rho), float(tau)


def _compute_expanded_ranking(
    method_tables: Sequence[MethodTable],
    subset_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    if len(method_tables) < 2:
        return pd.DataFrame(), pd.DataFrame(), [], []

    shared_full: Optional[set[str]] = None
    for table in method_tables:
        cur_ids = set(table.df["scene_id"].astype(str))
        if shared_full is None:
            shared_full = cur_ids
        else:
            shared_full &= cur_ids
    shared_full = shared_full or set()

    subset_set = set(str(v) for v in subset_ids)
    shared_subset = shared_full & subset_set
    if len(shared_full) < 2 or len(shared_subset) < 2:
        return pd.DataFrame(), pd.DataFrame(), sorted(shared_full), sorted(shared_subset)

    rows: List[Dict[str, object]] = []
    for table in method_tables:
        full = table.df[table.df["scene_id"].isin(shared_full)].copy()
        subset = table.df[table.df["scene_id"].isin(shared_subset)].copy()
        rows.append(
            {
                "method": table.name,
                "full_psnr": float(full["psnr"].mean()),
                "bass48_psnr": float(subset["psnr"].mean()),
                "full_ssim": float(full["ssim"].mean()),
                "bass48_ssim": float(subset["ssim"].mean()),
                "full_lpips": (
                    float(full["lpips"].mean()) if not full["lpips"].isna().all() else float("nan")
                ),
                "bass48_lpips": (
                    float(subset["lpips"].mean()) if not subset["lpips"].isna().all() else float("nan")
                ),
                "shared_full_size": int(len(shared_full)),
                "shared_subset_size": int(len(shared_subset)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("method", kind="mergesort").reset_index(drop=True)
    out["full_psnr_rank"] = out["full_psnr"].rank(method="dense", ascending=False).astype(int)
    out["bass48_psnr_rank"] = out["bass48_psnr"].rank(method="dense", ascending=False).astype(int)
    out["full_ssim_rank"] = out["full_ssim"].rank(method="dense", ascending=False).astype(int)
    out["bass48_ssim_rank"] = out["bass48_ssim"].rank(method="dense", ascending=False).astype(int)

    corr_rows = []
    for metric in ("psnr", "ssim"):
        rho, tau = _metric_rank_correlation(out, metric)
        corr_rows.append(
            {
                "metric": metric,
                "spearman_rho": rho,
                "kendall_tau": tau,
                "shared_full_size": int(len(shared_full)),
                "shared_subset_size": int(len(shared_subset)),
                "num_methods": int(len(method_tables)),
            }
        )
    if not out["full_lpips"].isna().any() and not out["bass48_lpips"].isna().any():
        out["full_lpips_rank"] = out["full_lpips"].rank(method="dense", ascending=True).astype(int)
        out["bass48_lpips_rank"] = out["bass48_lpips"].rank(method="dense", ascending=True).astype(int)
        rho = spearmanr(out["full_lpips_rank"], out["bass48_lpips_rank"]).statistic
        tau = kendalltau(out["full_lpips_rank"], out["bass48_lpips_rank"]).statistic
        corr_rows.append(
            {
                "metric": "lpips",
                "spearman_rho": float(rho),
                "kendall_tau": float(tau),
                "shared_full_size": int(len(shared_full)),
                "shared_subset_size": int(len(shared_subset)),
                "num_methods": int(len(method_tables)),
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    return out, corr_df, sorted(shared_full), sorted(shared_subset)


def _format_caption_snippets(
    repo_root: Path,
    tables_dir: Path,
    figures_dir: Path,
    include_ranking: bool,
) -> str:
    def rel(path: Path) -> str:
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    lines = [
            "% Strong-accept additions",
            "\\begin{figure}[t]",
            "  \\centering",
            f"  \\includegraphics[width=0.92\\linewidth]{{{rel(figures_dir / 'reliability_frontier.pdf')}}}",
            "  \\caption{Cost--reliability frontier for BASS. Larger subsets increase the Wilson lower confidence bound under the same joint fidelity criterion, exposing the trade-off between benchmark cost and repeated-sampling reliability.}",
            "  \\label{fig:reliability_frontier}",
            "\\end{figure}",
            "",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\input{{{rel(tables_dir / 'reliability_frontier.tex')}}}",
            "  \\caption{Cost--reliability frontier for BASS. Larger subsets increase the Wilson lower confidence bound under the same joint fidelity criterion, exposing the trade-off between benchmark cost and repeated-sampling reliability.}",
            "  \\label{tab:reliability_frontier}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\input{{{rel(tables_dir / 'failure_mode_comparison.tex')}}}",
            "  \\caption{Failure modes prevented by joint risk control. Mean-only acceptance can preserve average PSNR/SSIM/LPIPS while allowing distributional drift; the proposed joint criterion rejects such subsets through the KS guardrail and repeated-sampling Wilson LCB.}",
            "  \\label{tab:failure_mode_comparison}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\input{{{rel(tables_dir / 'threshold_sensitivity.tex')}}}",
            "  \\caption{Threshold sensitivity of the BASS recommendation under strict, default, and relaxed joint fidelity settings.}",
            "  \\label{tab:threshold_sensitivity}",
            "\\end{table}",
            "",
            "\\begin{figure}[t]",
            "  \\centering",
            f"  \\includegraphics[width=\\linewidth]{{{rel(figures_dir / 'distributional_fidelity_bass48.pdf')}}}",
            "  \\caption{Distributional fidelity of the exported BASS-48 subset. The subset tracks the full benchmark distributions for PSNR, SSIM, and LPIPS, complementing the mean-fidelity constraints with distributional diagnostics.}",
            "  \\label{fig:distributional_fidelity_bass48}",
            "\\end{figure}",
        ]
    if include_ranking:
        lines += [
            "",
            "\\begin{table}[t]",
            "  \\centering",
            f"  \\input{{{rel(tables_dir / 'expanded_ranking_full_vs_bass48.tex')}}}",
            "  \\caption{Cross-method ranking stability from the full shared benchmark intersection to BASS-48.}",
            "  \\label{tab:expanded_ranking_full_vs_bass48}",
            "\\end{table}",
        ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipnerf-log", type=Path, default=None)
    parser.add_argument("--descriptors", type=Path, default=None)
    parser.add_argument("--subset", type=Path, default=None)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--method-logs", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/strong_accept"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    repo_root = _repo_root()
    output_dir = _resolve_path(str(args.output_dir), repo_root)
    tables_dir = _resolve_path(str(args.tables_dir), repo_root)
    figures_dir = _resolve_path(str(args.figures_dir), repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    candidate_root = _detect_candidate_root(repo_root, args.candidate_root)
    descriptors_path = _detect_descriptors(repo_root, args.descriptors)
    zipnerf_log = _detect_zipnerf_log(repo_root, args.zipnerf_log)
    subset_path = _detect_subset(repo_root, args.subset, candidate_root)
    k6_mapping_path = _detect_k6_mapping(repo_root, candidate_root)
    baseline_config_path = (repo_root / DEFAULT_BASELINE_CONFIG).resolve()

    missing_critical: List[str] = []
    for label, path in (
        ("Zip-NeRF log", zipnerf_log),
        ("BASS subset", subset_path),
        ("k=6 cluster mapping", k6_mapping_path),
    ):
        if not _path_exists(path):
            missing_critical.append(f"{label}: {path}")

    sanity_lines: List[str] = [
        "# Strong-Accept Sanity Report",
        "",
        "## Inputs",
        f"- Candidate root: `{candidate_root}`" if candidate_root is not None else "- Candidate root: `NOT FOUND`",
        f"- Descriptors: `{descriptors_path}`" if descriptors_path is not None else "- Descriptors: `NOT FOUND`",
        f"- Zip-NeRF log: `{zipnerf_log}`" if zipnerf_log is not None else "- Zip-NeRF log: `NOT FOUND`",
        f"- BASS subset: `{subset_path}`" if subset_path is not None else "- BASS subset: `NOT FOUND`",
        f"- k=6 cluster mapping: `{k6_mapping_path}`" if k6_mapping_path is not None else "- k=6 cluster mapping: `NOT FOUND`",
        "",
    ]

    if missing_critical:
        sanity_lines += [
            "## NOT GENERATED",
            "",
            "The script stopped because required source files were missing:",
        ]
        sanity_lines += [f"- {item}" for item in missing_critical]
        (output_dir / "sanity_report.md").write_text("\n".join(sanity_lines) + "\n", encoding="utf-8")
        raise SystemExit("Missing required source files; see results/strong_accept/sanity_report.md")

    baseline_config: Dict[str, object] = {}
    if baseline_config_path.exists():
        baseline_config = _load_json(baseline_config_path)
    num_trials = int(baseline_config.get("M", 400))
    confidence_level = float(baseline_config.get("confidence_level", 0.95))
    p_min = float(baseline_config.get("p_min", DEFAULT_P_MIN))
    default_thresholds = dict(THRESHOLD_SETTINGS["default"])
    cfg_thresholds = baseline_config.get("thresholds", {})
    if isinstance(cfg_thresholds, dict):
        default_thresholds.update(
            {
                "psnr_tol": float(cfg_thresholds.get("tau_psnr", default_thresholds["psnr_tol"])),
                "ssim_tol": float(cfg_thresholds.get("tau_ssim", default_thresholds["ssim_tol"])),
                "lpips_tol": float(cfg_thresholds.get("tau_lpips", default_thresholds["lpips_tol"])),
                "ks_tol": float(cfg_thresholds.get("tau_ks", default_thresholds["ks_tol"])),
            }
        )
    threshold_settings = {
        "strict": dict(THRESHOLD_SETTINGS["strict"]),
        "default": dict(default_thresholds),
        "relaxed": dict(THRESHOLD_SETTINGS["relaxed"]),
    }

    mapping_df = _load_mapping_csv(k6_mapping_path)
    zipnerf_metrics_df = _load_zipnerf_metrics(zipnerf_log)
    full_k6_df = _merge_mapping_and_metrics(mapping_df, zipnerf_metrics_df)
    groups = _build_groups(full_k6_df)

    subset_ids = _load_subset_ids(subset_path)
    if len(subset_ids) != 48:
        sanity_lines += [
            "## Note",
            "",
            f"- The provided subset has size **{len(subset_ids)}**, not 48. Distributional and ranking outputs use the provided subset as-is.",
            "",
        ]

    method_specs, missing_method_paths, missing_method_names = _discover_method_specs(
        repo_root,
        explicit_paths=args.method_logs,
    )
    method_tables: List[MethodTable] = []
    usable_method_names: List[str] = []
    for spec in method_specs:
        try:
            table = _canonicalize_method_table(spec)
        except Exception as exc:
            LOGGER.warning("Skipping method log %s: %s", spec.path, exc)
            missing_method_paths.append(f"{spec.path} ({exc})")
            continue
        method_tables.append(table)
        usable_method_names.append(spec.name)

    balanced_trials_by_b: Dict[int, pd.DataFrame] = {}
    frontier_rows: List[Dict[str, object]] = []
    best_joint_by_b: Dict[int, Dict[str, object]] = {}
    best_mean_by_b: Dict[int, Dict[str, object]] = {}
    trial_frames: List[pd.DataFrame] = []

    for b in DEFAULT_BUDGETS_PER_CLUSTER:
        trial_df, best_joint, best_mean = _simulate_trials(
            df=full_k6_df,
            groups=groups,
            budget_per_cluster=b,
            num_trials=num_trials,
            seed=int(args.seed),
            mode="balanced",
            thresholds=default_thresholds,
        )
        balanced_trials_by_b[b] = trial_df
        best_joint_by_b[b] = best_joint
        best_mean_by_b[b] = best_mean
        trial_frames.append(trial_df)

        manifest_path = manifests_dir / f"bass_joint_k6_b{int(b)}_best_subset.csv"
        _save_subset_manifest(manifest_path, mapping_df, best_joint["subset_ids"])
        pass_mask = trial_df["joint_pass"].astype(bool)
        n_trials_cur = int(trial_df.shape[0])
        n_pass = int(pass_mask.sum())
        p_hat = float(n_pass / n_trials_cur) if n_trials_cur > 0 else float("nan")
        frontier_rows.append(
            {
                "budget_scenes": int(6 * b),
                "k": int(DEFAULT_K),
                "b": int(b),
                "n_trials": n_trials_cur,
                "n_pass": n_pass,
                "empirical_pass_rate": p_hat,
                "wilson_lcb_95": _wilson_lower_bound(p_hat, n_trials_cur, confidence_level),
                "psnr_mean_gap": float(best_joint["stats"]["abs_psnr_gap"]),
                "ssim_mean_gap": float(best_joint["stats"]["abs_ssim_gap"]),
                "lpips_mean_gap": float(best_joint["stats"]["abs_lpips_gap"]),
                "max_ks": float(best_joint["stats"]["max_ks"]),
                "best_subset_manifest": str(manifest_path),
            }
        )

    frontier_df = pd.DataFrame(frontier_rows).sort_values("budget_scenes", kind="mergesort").reset_index(drop=True)
    frontier_df.to_csv(output_dir / "reliability_frontier.csv", index=False)
    _make_reliability_frontier_table(tables_dir / "reliability_frontier.tex", frontier_df)
    _save_reliability_figure(frontier_df, figures_dir / "reliability_frontier.pdf", p_min=p_min)
    _save_reliability_figure(frontier_df, figures_dir / "fig_empirical_frontier.pdf", p_min=p_min)

    threshold_rows: List[Dict[str, object]] = []
    for setting_name, thresholds in threshold_settings.items():
        recommended_budget, lcb_at_budget, note = _recommend_budget(
            balanced_trials_by_b=balanced_trials_by_b,
            thresholds=thresholds,
            p_min=p_min,
            confidence_level=confidence_level,
        )
        threshold_rows.append(
            {
                "setting": setting_name,
                "psnr_tol": thresholds["psnr_tol"],
                "ssim_tol": thresholds["ssim_tol"],
                "lpips_tol": thresholds["lpips_tol"],
                "ks_tol": thresholds["ks_tol"],
                "recommended_budget": recommended_budget,
                "wilson_lcb_at_recommended_budget": lcb_at_budget,
                "note": note,
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    _make_threshold_sensitivity_table(tables_dir / "threshold_sensitivity.tex", threshold_df)

    uniform_trial_df, uniform_best_joint, _ = _simulate_trials(
        df=full_k6_df,
        groups=groups,
        budget_per_cluster=8,
        num_trials=num_trials,
        seed=int(args.seed),
        mode="uniform",
        thresholds=default_thresholds,
    )
    trial_frames.append(uniform_trial_df)
    random_manifest_path = manifests_dir / "random_uniform_48_best_subset.csv"
    _save_subset_manifest(random_manifest_path, mapping_df, uniform_best_joint["subset_ids"])

    mean_only_best = best_mean_by_b[8]
    mean_only_manifest_path = manifests_dir / "balanced_meanonly_48_best_subset.csv"
    _save_subset_manifest(mean_only_manifest_path, mapping_df, mean_only_best["subset_ids"])
    joint_best = best_joint_by_b[8]
    joint_manifest_path = manifests_dir / "bass_joint_48_best_subset.csv"
    _save_subset_manifest(joint_manifest_path, mapping_df, joint_best["subset_ids"])

    def _failure_row(
        label: str,
        trial_df: pd.DataFrame,
        best_record: Dict[str, object],
        manifest_path: Path,
    ) -> Dict[str, object]:
        n_trials_cur = int(trial_df.shape[0])
        mean_pass_rate = float(trial_df["mean_constraints_pass"].mean())
        ks_pass_rate = float(trial_df["ks_pass"].mean())
        joint_pass_rate = float(trial_df["joint_pass"].mean())
        ranking_preserved = _compute_ranking_preserved(method_tables, best_record["subset_ids"]) if method_tables else None
        return {
            "method": label,
            "budget_scenes": 48,
            "n_trials": n_trials_cur,
            "mean_constraints_pass_rate": mean_pass_rate,
            "ks_pass_rate": ks_pass_rate,
            "joint_pass_rate": joint_pass_rate,
            "wilson_lcb_95": _wilson_lower_bound(joint_pass_rate, n_trials_cur, confidence_level),
            "exported_or_best_subset_psnr_gap": float(best_record["stats"]["abs_psnr_gap"]),
            "exported_or_best_subset_ssim_gap": float(best_record["stats"]["abs_ssim_gap"]),
            "exported_or_best_subset_lpips_gap": float(best_record["stats"]["abs_lpips_gap"]),
            "exported_or_best_subset_max_ks": float(best_record["stats"]["max_ks"]),
            "ranking_preserved": ranking_preserved,
            "subset_manifest": str(manifest_path),
            "note": "",
        }

    failure_rows = [
        _failure_row("Random uniform", uniform_trial_df, uniform_best_joint, random_manifest_path),
        _failure_row("Balanced mean-only", balanced_trials_by_b[8], mean_only_best, mean_only_manifest_path),
        _failure_row("BASS joint criterion", balanced_trials_by_b[8], joint_best, joint_manifest_path),
    ]

    baseline_summary_path = repo_root / "sweep_cluster_k" / "baseline_comparison_lpips_ks" / "baseline_sweep_results.csv"
    if baseline_summary_path.exists():
        baseline_summary_df = pd.read_csv(baseline_summary_path)
        for baseline_name in ("facility_location", "kcenter_farthest_first"):
            row = baseline_summary_df[
                (baseline_summary_df["baseline"].astype(str) == baseline_name)
                & (pd.to_numeric(baseline_summary_df["subset_size"], errors="coerce") == 48)
            ]
            if row.empty:
                continue
            rec = row.iloc[0]
            label = "Facility location (summary only)" if baseline_name == "facility_location" else "k-center (summary only)"
            failure_rows.append(
                {
                    "method": label,
                    "budget_scenes": 48,
                    "n_trials": float("nan"),
                    "mean_constraints_pass_rate": float("nan"),
                    "ks_pass_rate": float("nan"),
                    "joint_pass_rate": float(pd.to_numeric(rec["pass_rate"], errors="coerce")),
                    "wilson_lcb_95": float(pd.to_numeric(rec["lcb"], errors="coerce")),
                    "exported_or_best_subset_psnr_gap": float("nan"),
                    "exported_or_best_subset_ssim_gap": float("nan"),
                    "exported_or_best_subset_lpips_gap": float("nan"),
                    "exported_or_best_subset_max_ks": float("nan"),
                    "ranking_preserved": float("nan"),
                    "subset_manifest": "",
                    "note": "Only summary pass-rate CSV available; subset manifest and trial-level records are missing.",
                }
            )

    failure_df = pd.DataFrame(failure_rows)
    failure_df.to_csv(output_dir / "failure_mode_comparison.csv", index=False)
    _make_failure_mode_table(tables_dir / "failure_mode_comparison.tex", failure_df)

    mean_pass_joint_fail = balanced_trials_by_b[8][
        balanced_trials_by_b[8]["mean_constraints_pass"].astype(bool)
        & ~balanced_trials_by_b[8]["joint_pass"].astype(bool)
    ].copy()
    mean_failure_example: Optional[pd.Series] = None
    if not mean_pass_joint_fail.empty:
        mean_failure_example = mean_pass_joint_fail.sort_values(
            ["joint_objective", "max_ks", "trial"],
            ascending=[True, False, True],
            kind="mergesort",
        ).iloc[0]

    dist_summary_df = _save_distribution_figure(
        full_df=full_k6_df,
        subset_ids=subset_ids,
        out_pdf=figures_dir / "distributional_fidelity_bass48.pdf",
    )
    dist_summary_df.to_csv(output_dir / "distributional_fidelity_summary.csv", index=False)

    ranking_method_df, ranking_corr_df, shared_full_ids, shared_subset_ids = _compute_expanded_ranking(
        method_tables,
        subset_ids=subset_ids,
    )
    if not ranking_method_df.empty:
        ranking_method_df.to_csv(output_dir / "expanded_ranking_full_vs_bass48.csv", index=False)
        ranking_corr_df.to_csv(output_dir / "expanded_ranking_correlation_summary.csv", index=False)
        _make_expanded_ranking_table(
            tables_dir / "expanded_ranking_full_vs_bass48.tex",
            ranking_method_df,
            ranking_corr_df,
        )

    trial_metrics_df = pd.concat(trial_frames, ignore_index=True)
    trial_metrics_df.to_csv(output_dir / "trial_metrics.csv", index=False)

    not_generated: List[str] = []
    if baseline_summary_path.exists():
        not_generated.append(
            "Facility-location and k-center subset-level gap diagnostics were not fully generated because the repo contains only summary pass-rate CSVs, not the corresponding subset manifests or trial-level records."
        )
    else:
        not_generated.append(
            "Facility-location and k-center summary CSVs were not found, so those baselines were not generated."
        )
    if missing_method_names:
        not_generated.append(
            "No additional per-scene method logs were found for: " + ", ".join(f"`{name}`" for name in missing_method_names) + "."
        )

    latex_snippets = _format_caption_snippets(
        repo_root,
        tables_dir,
        figures_dir,
        include_ranking=not ranking_method_df.empty,
    )
    (output_dir / "latex_snippets.tex").write_text(latex_snippets + "\n", encoding="utf-8")

    sanity_lines += [
        "## Result 1: Larger-Budget Reliability Frontier",
        "",
        f"- Generated: `{output_dir / 'reliability_frontier.csv'}`",
        f"- Trials per budget: **{num_trials}**",
        f"- Budgets evaluated (scenes): `{', '.join(str(v) for v in frontier_df['budget_scenes'].tolist())}`",
        "",
        "## Result 2: Failure-Mode Comparison",
        "",
        f"- Generated: `{output_dir / 'failure_mode_comparison.csv'}`",
    ]
    if mean_failure_example is not None:
        sanity_lines += [
            f"- Found **{int(mean_pass_joint_fail.shape[0])}/{int(balanced_trials_by_b[8].shape[0])}** balanced 48-scene candidates that satisfy the mean constraints but fail the joint criterion because KS exceeds the guardrail.",
            f"- Example mean-pass / joint-fail trial: `trial={int(mean_failure_example['trial'])}`, `max_ks={float(mean_failure_example['max_ks']):.3f}`, `abs_lpips_gap={float(mean_failure_example['abs_lpips_gap']):.4f}`.",
        ]
    else:
        sanity_lines.append(
            f"- No mean-pass / joint-fail balanced 48-scene candidate was found in the searched scope of **{int(balanced_trials_by_b[8].shape[0])}** trials."
        )
    sanity_lines += [
        "",
        "## Result 3: Threshold Sensitivity",
        "",
        f"- Generated: `{output_dir / 'threshold_sensitivity.csv'}`",
        f"- Default recommendation target: `p_min={p_min:.2f}`",
        "",
        "## Result 4: Distributional Fidelity",
        "",
        f"- Generated combined ECDF figure: `{figures_dir / 'distributional_fidelity_bass48.pdf'}`",
        f"- Subset evaluated: `{subset_path}`",
        f"- Full Zip-NeRF overlap with k=6 universe: **{int(full_k6_df.shape[0])}** scenes",
        "",
        "## Result 5: Expanded Cross-Method Ranking",
        "",
    ]
    if not ranking_method_df.empty:
        sanity_lines += [
            f"- Generated: `{output_dir / 'expanded_ranking_full_vs_bass48.csv'}`",
            f"- Methods included: `{', '.join(usable_method_names)}`",
            f"- Shared full-scene intersection: **{len(shared_full_ids)}** scenes",
            f"- Shared BASS-48 intersection: **{len(shared_subset_ids)}** scenes",
        ]
    else:
        sanity_lines.append("- NOT GENERATED: fewer than two valid per-scene method logs were available after parsing.")

    sanity_lines += [
        "",
        "## Search Scope",
        "",
        "- Method-log search looked for per-scene CSV/XLSX tables for Instant-NGP, Zip-NeRF, Feature-Splatting, Nerfacto, TensoRF, DVGO, Plenoxels, 3DGS, and Mip-NeRF.",
        "- Baseline search checked `sweep_cluster_k/baseline_comparison_lpips_ks/` for pass-rate summaries and looked recursively for matching subset manifests.",
        "",
        "## NOT GENERATED",
        "",
    ]
    sanity_lines += [f"- {line}" for line in not_generated]
    sanity_lines += [
        "",
        "## Notes",
        "",
        "- The larger-budget frontier, threshold sensitivity, and balanced failure-mode rows are deterministic reconstructions from the current `zipnerf.xlsx` log, the `k=6` regime mapping, `M=400`, and `seed=0`.",
        "- The saved `baseline_comparison_lpips_ks` summaries could not be reproduced exactly from repo-local code because the original LPIPS/KS baseline generator script and its parsed `latex/zipnerf.csv` source are not present.",
        "- The exported BASS-48 subset used for the distributional and ranking diagnostics is the subset at the provided `--subset` path; by default this resolves to the k=6 BASS export in `sweep_cluster_k/budget_sweep_k6_auto_v2/recommended_subset.csv`.",
        "",
        "## LaTeX Snippets",
        "",
        "```tex",
        latex_snippets,
        "```",
        "",
    ]
    (output_dir / "sanity_report.md").write_text("\n".join(sanity_lines), encoding="utf-8")

    run_config = {
        "zipnerf_log": str(zipnerf_log),
        "descriptors": None if descriptors_path is None else str(descriptors_path),
        "subset": str(subset_path),
        "candidate_root": None if candidate_root is None else str(candidate_root),
        "k6_mapping": str(k6_mapping_path),
        "default_thresholds": default_thresholds,
        "threshold_settings": threshold_settings,
        "num_trials": num_trials,
        "p_min": p_min,
        "seed": int(args.seed),
        "method_logs": [{"name": table.name, "path": str(table.path)} for table in method_tables],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(latex_snippets)


if __name__ == "__main__":
    main()
