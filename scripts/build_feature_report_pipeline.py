#!/usr/bin/env python3
"""End-to-end feature extraction + reporting pipeline.

Upgrades implemented:
1) Parallel extraction with deterministic output ordering.
2) Resume mode with cache keyed by scene path + transforms mtime.
3) Strict scene validation with failure taxonomy.
4) Quality gates for failed-scene rate and missingness.
5) Drift tracking against a baseline run.
6) Rich report artifacts (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("build_feature_report_pipeline")


# Validation / extraction taxonomy
ERR_MISSING_TRANSFORMS = "missing_transforms"
ERR_INVALID_TRANSFORMS_JSON = "invalid_transforms_json"
ERR_MISSING_SPARSE_IMAGES = "missing_sparse_images_txt"
ERR_MISSING_SPARSE_POINTS = "missing_sparse_points3d_txt"
ERR_MISSING_MASK = "missing_mask_dir"
ERR_EXTRACT_EXCEPTION = "extract_exception"

CACHE_VERSION = 2


def _format_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _progress_line(prefix: str, done: int, total: int, start_time: float, width: int = 26) -> str:
    if total <= 0:
        return f"{prefix} 0/0"
    frac = min(1.0, max(0.0, float(done) / float(total)))
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(1e-9, time.time() - start_time)
    rate = done / elapsed if done > 0 else 0.0
    remaining = (total - done) / rate if rate > 1e-9 else float("inf")
    eta = "--:--" if not np.isfinite(remaining) else _format_duration(remaining)
    return (
        f"{prefix} [{bar}] {done}/{total} "
        f"({100.0 * frac:5.1f}%) | {rate:5.1f}/s | eta {eta}"
    )


def _emit_progress(
    *,
    prefix: str,
    done: int,
    total: int,
    start_time: float,
    dynamic: bool,
    final: bool = False,
) -> None:
    line = _progress_line(prefix, done, total, start_time)
    if dynamic:
        end = "\n" if final else ""
        sys.stderr.write("\r" + line + end)
        sys.stderr.flush()
        return
    if final or done == 1 or done == total or done % 100 == 0:
        LOGGER.info("%s", line)


def _setup_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_setup_import_path()


def _preflight_feature_extractor_import() -> None:
    try:
        import feature_extractor  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name if getattr(exc, "name", None) else str(exc)
        raise ModuleNotFoundError(
            "Failed to import feature extraction dependencies. "
            f"Missing module: {missing}. "
            "Install missing packages in the active environment."
        ) from exc


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def _safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Root JSON must be an object")
    return payload


def _scene_cache_key(
    transform_path: Path,
    *,
    max_frames_per_scene: Optional[int],
    frame_stride: int,
) -> str:
    st = transform_path.stat()
    return (
        f"{transform_path.resolve()}|{st.st_mtime_ns}"
        f"|max_frames={max_frames_per_scene}|stride={frame_stride}"
    )


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = _safe_read_json(path)
    except Exception as exc:
        LOGGER.warning("Ignoring unreadable cache %s: %s", path, exc)
        return {}
    if payload.get("version") != CACHE_VERSION:
        LOGGER.warning("Ignoring cache with incompatible version: %s", path)
        return {}
    records = payload.get("records")
    if not isinstance(records, dict):
        return {}
    return {str(k): v for k, v in records.items() if isinstance(v, dict)}


def save_cache(path: Path, records: Dict[str, Dict[str, Any]]) -> None:
    payload = {"version": CACHE_VERSION, "records": records}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def discover_transforms(
    scenes_root: Path, transforms_name: str, max_scenes: Optional[int] = None
) -> List[Path]:
    if not scenes_root.exists():
        raise FileNotFoundError(f"Scenes root does not exist: {scenes_root}")
    transforms = sorted(scenes_root.rglob(transforms_name))
    if max_scenes is not None:
        transforms = transforms[: max(0, max_scenes)]
    return transforms


def _detect_mask_dir(scene_dir: Path, mask_dirs: List[str]) -> Optional[str]:
    for d in mask_dirs:
        p = scene_dir / d
        if p.exists() and p.is_dir():
            return d
    return None


def validate_scene(
    transform_path: Path, mask_dirs: List[str], require_mask: bool
) -> Optional[Dict[str, str]]:
    scene_dir = transform_path.parent

    if not transform_path.exists():
        return {
            "error_code": ERR_MISSING_TRANSFORMS,
            "error": "Transform file does not exist",
            "transform_path": str(transform_path),
        }

    try:
        payload = _safe_read_json(transform_path)
    except Exception as exc:
        return {
            "error_code": ERR_INVALID_TRANSFORMS_JSON,
            "error": str(exc),
            "transform_path": str(transform_path),
        }

    _ = payload.get("frames", [])

    sparse_images = scene_dir / "sparse" / "txt" / "images.txt"
    if not sparse_images.exists():
        return {
            "error_code": ERR_MISSING_SPARSE_IMAGES,
            "error": f"Missing {sparse_images}",
            "transform_path": str(transform_path),
        }

    sparse_points = scene_dir / "sparse" / "txt" / "points3D.txt"
    if not sparse_points.exists():
        return {
            "error_code": ERR_MISSING_SPARSE_POINTS,
            "error": f"Missing {sparse_points}",
            "transform_path": str(transform_path),
        }

    if require_mask:
        mask_dir = _detect_mask_dir(scene_dir, mask_dirs)
        if mask_dir is None:
            return {
                "error_code": ERR_MISSING_MASK,
                "error": f"None of mask dirs present: {mask_dirs}",
                "transform_path": str(transform_path),
            }

    return None


def extract_record_from_transform(
    transform_path: Path,
    *,
    max_frames_per_scene: Optional[int],
    frame_stride: int,
) -> Dict[str, Any]:
    # local import keeps this script import-light and worker-safe
    import feature_extractor as fe

    dataset_path = transform_path.parent
    scene_id = dataset_path.name
    data = _safe_read_json(transform_path)
    meta = fe.extract_features(
        str(dataset_path),
        data,
        max_frames=max_frames_per_scene,
        frame_stride=frame_stride,
    )
    record: Dict[str, Any] = {"dish_id": scene_id, "scene_path": str(dataset_path)}
    record.update(meta)
    return record


def _extract_worker(
    transform_path_str: str, max_frames_per_scene: Optional[int], frame_stride: int
) -> Dict[str, Any]:
    tpath = Path(transform_path_str)
    try:
        record = extract_record_from_transform(
            tpath,
            max_frames_per_scene=max_frames_per_scene,
            frame_stride=frame_stride,
        )
        return {"ok": True, "record": record}
    except Exception as exc:  # broad by design for resilient pipelines
        return {
            "ok": False,
            "failure": {
                "error_code": ERR_EXTRACT_EXCEPTION,
                "error": str(exc),
                "transform_path": str(tpath),
            },
        }


def _run_parallel_extraction(
    transforms: List[Path],
    num_workers: int,
    show_progress: bool,
    *,
    max_frames_per_scene: Optional[int],
    frame_stride: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    total = len(transforms)
    done = 0
    start = time.time()
    dynamic = bool(show_progress and sys.stderr.isatty())

    if num_workers <= 1:
        for tpath in transforms:
            result = _extract_worker(
                str(tpath),
                max_frames_per_scene=max_frames_per_scene,
                frame_stride=frame_stride,
            )
            if result["ok"]:
                records.append(result["record"])
            else:
                failures.append(result["failure"])
            done += 1
            if show_progress:
                _emit_progress(
                    prefix="Extracting",
                    done=done,
                    total=total,
                    start_time=start,
                    dynamic=dynamic,
                    final=done == total,
                )
        return records, failures

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        fut_map = {
            ex.submit(
                _extract_worker,
                str(t),
                max_frames_per_scene,
                frame_stride,
            ): t
            for t in transforms
        }
        for fut in as_completed(fut_map):
            res = fut.result()
            if res["ok"]:
                records.append(res["record"])
            else:
                failures.append(res["failure"])
            done += 1
            if show_progress:
                _emit_progress(
                    prefix="Extracting",
                    done=done,
                    total=total,
                    start_time=start,
                    dynamic=dynamic,
                    final=done == total,
                )
    if show_progress and (not dynamic) and total == 0:
        LOGGER.info("Extracting 0/0")
    return records, failures


def build_feature_table(
    transforms_paths: Iterable[Path],
    *,
    num_workers: int,
    show_progress: bool,
    max_frames_per_scene: Optional[int],
    frame_stride: int,
    resume: bool,
    cache_path: Path,
    mask_dirs: List[str],
    require_mask: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, str]], Dict[str, Dict[str, Any]], int]:
    transforms_list = list(transforms_paths)
    cache_records = load_cache(cache_path) if resume else {}
    cache_hits = 0

    failures: List[Dict[str, str]] = []
    records: List[Dict[str, Any]] = []

    to_extract: List[Path] = []
    key_by_transform: Dict[Path, str] = {}

    for tpath in transforms_list:
        failure = validate_scene(tpath, mask_dirs=mask_dirs, require_mask=require_mask)
        if failure is not None:
            failures.append(failure)
            continue

        key = _scene_cache_key(
            tpath,
            max_frames_per_scene=max_frames_per_scene,
            frame_stride=frame_stride,
        )
        key_by_transform[tpath] = key
        if resume and key in cache_records:
            records.append(cache_records[key])
            cache_hits += 1
        else:
            to_extract.append(tpath)

    LOGGER.info(
        "Validated scenes: %d | queued extraction: %d | cache hits: %d",
        len(transforms_list) - len([f for f in failures if f["error_code"] != ERR_EXTRACT_EXCEPTION]),
        len(to_extract),
        cache_hits,
    )

    extracted, extract_failures = _run_parallel_extraction(
        to_extract,
        num_workers=num_workers,
        show_progress=show_progress,
        max_frames_per_scene=max_frames_per_scene,
        frame_stride=frame_stride,
    )
    failures.extend(extract_failures)
    records.extend(extracted)

    # Update cache with successful records.
    if resume:
        extracted_by_scene_path = {
            str(r.get("scene_path")): r for r in extracted if isinstance(r, dict)
        }
        for tpath in to_extract:
            key = key_by_transform[tpath]
            rec = extracted_by_scene_path.get(str(tpath.parent))
            if rec is not None:
                cache_records[key] = rec
        save_cache(cache_path, cache_records)
        LOGGER.info("Updated cache: %s", cache_path)

    if not records:
        if failures:
            fdf = pd.DataFrame(failures)
            summary = (
                fdf.groupby("error_code", dropna=False)
                .size()
                .sort_values(ascending=False)
            )
            summary_str = ", ".join(f"{k}:{int(v)}" for k, v in summary.items())
            sample_rows = fdf.head(3).to_dict(orient="records")
            raise RuntimeError(
                "No scenes were successfully processed. "
                f"Failure summary: {summary_str}. "
                f"Sample failures: {sample_rows}"
            )
        raise RuntimeError("No scenes were successfully processed and no failure data is available.")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("dish_id", kind="mergesort").reset_index(drop=True)

    preferred_prefix = ["dish_id", "scene_path"]
    feature_cols = sorted(c for c in df.columns if c not in preferred_prefix)
    ordered_cols = [c for c in preferred_prefix if c in df.columns] + feature_cols
    df = df[ordered_cols]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, failures, cache_records, cache_hits


def _compute_outlier_stats(numeric_df: pd.DataFrame) -> pd.DataFrame:
    q1 = numeric_df.quantile(0.25, axis=0)
    q3 = numeric_df.quantile(0.75, axis=0)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    outlier_count: Dict[str, int] = {}
    outlier_percent: Dict[str, float] = {}
    n = max(1, int(numeric_df.shape[0]))
    for col in numeric_df.columns:
        s = numeric_df[col]
        mask = (s < low[col]) | (s > high[col])
        c = int(mask.fillna(False).sum())
        outlier_count[col] = c
        outlier_percent[col] = (100.0 * c) / n

    return pd.DataFrame(
        {"outlier_count": pd.Series(outlier_count), "outlier_percent": pd.Series(outlier_percent)}
    )


def compute_summary(
    df: pd.DataFrame, id_cols: Iterable[str]
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    id_set = set(id_cols)
    feature_cols = [c for c in df.columns if c not in id_set]
    numeric_df = df[feature_cols]
    missing_pct = numeric_df.isna().mean(axis=0) * 100.0
    outlier_df = _compute_outlier_stats(numeric_df)

    desc = pd.DataFrame(
        {
            "mean": numeric_df.mean(axis=0),
            "std": numeric_df.std(axis=0, ddof=0),
            "median": numeric_df.median(axis=0),
            "q1": numeric_df.quantile(0.25, axis=0),
            "q3": numeric_df.quantile(0.75, axis=0),
            "min": numeric_df.min(axis=0),
            "max": numeric_df.max(axis=0),
            "missing_percent": missing_pct,
        }
    ).join(outlier_df, how="left")
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc = desc[
        [
            "mean",
            "std",
            "median",
            "iqr",
            "q1",
            "q3",
            "min",
            "max",
            "missing_percent",
            "outlier_count",
            "outlier_percent",
        ]
    ].sort_index()

    corr_df = numeric_df.corr(method="pearson")

    per_feature: Dict[str, Dict[str, Any]] = {}
    for col in desc.index:
        row = desc.loc[col]
        per_feature[col] = {
            "mean": _nan_to_none(float(row["mean"])),
            "std": _nan_to_none(float(row["std"])),
            "median": _nan_to_none(float(row["median"])),
            "iqr": _nan_to_none(float(row["iqr"])),
            "q1": _nan_to_none(float(row["q1"])),
            "q3": _nan_to_none(float(row["q3"])),
            "min": _nan_to_none(float(row["min"])),
            "max": _nan_to_none(float(row["max"])),
            "missing_percent": _nan_to_none(float(row["missing_percent"])),
            "outlier_count": _nan_to_none(float(row["outlier_count"])),
            "outlier_percent": _nan_to_none(float(row["outlier_percent"])),
        }

    summary = {
        "num_scenes": int(df.shape[0]),
        "num_features": int(len(feature_cols)),
        "id_columns": list(id_cols),
        "per_feature_stats": per_feature,
    }
    return summary, desc, corr_df


def _top_correlated_pairs(corr_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr", "abs_corr"])
    corr = corr_df.copy()
    np.fill_diagonal(corr.values, np.nan)
    pairs: List[Tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iat[i, j]
            if np.isnan(v):
                continue
            pairs.append((cols[i], cols[j], float(v)))
    out = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"])
    if out.empty:
        out["abs_corr"] = []
        return out
    out["abs_corr"] = out["corr"].abs()
    return out.sort_values("abs_corr", ascending=False, kind="mergesort").head(top_k)


def _failure_summary(failures: List[Dict[str, str]]) -> pd.DataFrame:
    if not failures:
        return pd.DataFrame(columns=["error_code", "count"])
    fdf = pd.DataFrame(failures)
    summary = (
        fdf.groupby("error_code", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False, kind="mergesort")
    )
    return summary


def _compute_drift(
    current_summary: Dict[str, Any],
    current_corr: pd.DataFrame,
    baseline_summary_path: Path,
    baseline_corr_path: Optional[Path],
    *,
    mean_delta_threshold: float,
    missing_delta_threshold: float,
) -> Dict[str, Any]:
    baseline_summary = _safe_read_json(baseline_summary_path)
    baseline_per = baseline_summary.get("per_feature_stats", {})
    current_per = current_summary.get("per_feature_stats", {})

    common = sorted(set(current_per.keys()) & set(baseline_per.keys()))
    per_feature_delta: Dict[str, Dict[str, Any]] = {}
    changed_features: List[str] = []

    for feat in common:
        cur_mean = current_per[feat].get("mean")
        base_mean = baseline_per[feat].get("mean")
        cur_missing = current_per[feat].get("missing_percent")
        base_missing = baseline_per[feat].get("missing_percent")

        mean_delta = (
            None
            if cur_mean is None or base_mean is None
            else float(cur_mean) - float(base_mean)
        )
        missing_delta = (
            None
            if cur_missing is None or base_missing is None
            else float(cur_missing) - float(base_missing)
        )
        rec = {
            "mean_delta": mean_delta,
            "abs_mean_delta": None if mean_delta is None else abs(mean_delta),
            "missing_percent_delta": missing_delta,
            "abs_missing_percent_delta": None
            if missing_delta is None
            else abs(missing_delta),
        }
        per_feature_delta[feat] = rec

        trigger = False
        if rec["abs_mean_delta"] is not None and rec["abs_mean_delta"] >= mean_delta_threshold:
            trigger = True
        if (
            rec["abs_missing_percent_delta"] is not None
            and rec["abs_missing_percent_delta"] >= missing_delta_threshold
        ):
            trigger = True
        if trigger:
            changed_features.append(feat)

    corr_delta = None
    if baseline_corr_path is not None and baseline_corr_path.is_file():
        baseline_corr = pd.read_csv(baseline_corr_path, index_col=0)
        common_corr = sorted(set(current_corr.columns) & set(baseline_corr.columns))
        if common_corr:
            cur = current_corr.loc[common_corr, common_corr]
            base = baseline_corr.loc[common_corr, common_corr]
            diff = (cur - base).abs().to_numpy()
            corr_delta = {
                "mean_abs_corr_delta": float(np.nanmean(diff)),
                "max_abs_corr_delta": float(np.nanmax(diff)),
                "num_common_features": len(common_corr),
            }

    return {
        "baseline_summary_json": str(baseline_summary_path),
        "baseline_corr_csv": None if baseline_corr_path is None else str(baseline_corr_path),
        "num_common_features": len(common),
        "changed_features": sorted(changed_features),
        "num_changed_features": len(changed_features),
        "per_feature_delta": per_feature_delta,
        "correlation_drift": corr_delta,
    }


def _write_report_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_markdown_report(
    path: Path,
    *,
    scenes_discovered: int,
    scenes_processed: int,
    scenes_failed: int,
    num_features: int,
    missing_df: pd.DataFrame,
    top_corr_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    failures: List[Dict[str, str]],
    drift_report: Optional[Dict[str, Any]],
) -> None:
    top_missing = missing_df.head(15).reset_index(drop=True)
    top_corr = top_corr_df.head(15).reset_index(drop=True)
    top_failed_scenes = pd.DataFrame(failures).head(15)

    lines = [
        "# Feature Extraction Report",
        "",
        "## Overview",
        f"- Scenes discovered: **{scenes_discovered}**",
        f"- Scenes processed: **{scenes_processed}**",
        f"- Scenes failed: **{scenes_failed}**",
        f"- Features extracted: **{num_features}**",
        "",
        "## Top Missing Features",
        "",
        "| feature | missing_percent | outlier_percent |",
        "|---|---:|---:|",
    ]
    for _, row in top_missing.iterrows():
        lines.append(
            f"| {row['feature']} | {row['missing_percent']:.2f} | {row['outlier_percent']:.2f} |"
        )

    lines += [
        "",
        "## Top Correlated Feature Pairs",
        "",
        "| feature_a | feature_b | corr | abs_corr |",
        "|---|---|---:|---:|",
    ]
    for _, row in top_corr.iterrows():
        lines.append(
            f"| {row['feature_a']} | {row['feature_b']} | {row['corr']:.4f} | {row['abs_corr']:.4f} |"
        )

    lines += [
        "",
        "## Failure Taxonomy",
        "",
        "| error_code | count |",
        "|---|---:|",
    ]
    for _, row in failure_df.iterrows():
        lines.append(f"| {row['error_code']} | {int(row['count'])} |")

    lines += [
        "",
        "## Top Failed Scenes",
        "",
        "| transform_path | error_code | error |",
        "|---|---|---|",
    ]
    if top_failed_scenes.empty:
        lines.append("| - | - | - |")
    else:
        for _, row in top_failed_scenes.iterrows():
            lines.append(
                f"| {row.get('transform_path','')} | {row.get('error_code','')} | {str(row.get('error','')).replace('|','/')} |"
            )

    if drift_report is not None:
        lines += [
            "",
            "## Drift vs Baseline",
            "",
            f"- Common features: **{drift_report.get('num_common_features', 0)}**",
            f"- Changed features: **{drift_report.get('num_changed_features', 0)}**",
        ]
        corr_drift = drift_report.get("correlation_drift")
        if corr_drift:
            lines += [
                f"- Mean abs correlation delta: **{corr_drift['mean_abs_corr_delta']:.6f}**",
                f"- Max abs correlation delta: **{corr_drift['max_abs_corr_delta']:.6f}**",
            ]

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_mask_dirs(value: str) -> List[str]:
    parts = [x.strip() for x in value.split(",")]
    return [x for x in parts if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-scene features and generate reporting artifacts."
    )
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--transforms-name", type=str, default="transforms_train.json")
    parser.add_argument("--output-dir", type=Path, default=Path("."))

    parser.add_argument("--raw-csv-name", type=str, default="feats2.csv")
    parser.add_argument("--summary-json-name", type=str, default="feature_summary_stats.json")
    parser.add_argument("--missing-csv-name", type=str, default="feature_missingness.csv")
    parser.add_argument(
        "--top-corr-csv-name", type=str, default="feature_top_correlations.csv"
    )
    parser.add_argument("--corr-csv-name", type=str, default="feature_correlation_matrix.csv")
    parser.add_argument(
        "--failure-summary-csv-name", type=str, default="feature_failure_summary.csv"
    )
    parser.add_argument(
        "--failures-json-name", type=str, default="feature_extraction_failures.json"
    )
    parser.add_argument(
        "--report-json-name", type=str, default="feature_extraction_report.json"
    )
    parser.add_argument("--report-md-name", type=str, default="feature_extraction_report.md")
    parser.add_argument("--drift-json-name", type=str, default="feature_drift_report.json")

    parser.add_argument("--cache-name", type=str, default="feature_extraction_cache.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live extraction progress output.",
    )

    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=None,
        help="Limit number of frames used per scene for image-derived features "
        "(deterministic uniform sampling).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every N-th frame before optional max-frames sampling.",
    )
    parser.add_argument("--mask-dirs", type=str, default="masks_omvs,masks,rgba")
    parser.add_argument("--require-mask", action="store_true")

    # Quality gates
    parser.add_argument("--max-failed-scenes-pct", type=float, default=100.0)
    parser.add_argument("--max-feature-missing-pct", type=float, default=100.0)

    # Drift inputs
    parser.add_argument("--baseline-output-dir", type=Path, default=None)
    parser.add_argument("--baseline-summary-json", type=Path, default=None)
    parser.add_argument("--baseline-corr-csv", type=Path, default=None)
    parser.add_argument("--drift-mean-delta-threshold", type=float, default=0.25)
    parser.add_argument("--drift-missing-delta-threshold", type=float, default=5.0)

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dirs = _parse_mask_dirs(args.mask_dirs)

    _preflight_feature_extractor_import()

    LOGGER.info("Discovering scenes in: %s", args.scenes_root)
    transforms = discover_transforms(
        args.scenes_root, args.transforms_name, max_scenes=args.max_scenes
    )
    LOGGER.info("Found %d transform files", len(transforms))
    if not transforms:
        raise FileNotFoundError(
            f"No '{args.transforms_name}' files found under {args.scenes_root}"
        )

    cache_path = out_dir / args.cache_name
    df, failures, _, cache_hits = build_feature_table(
        transforms,
        num_workers=max(1, args.num_workers),
        show_progress=not args.no_progress,
        max_frames_per_scene=args.max_frames_per_scene,
        frame_stride=max(1, int(args.frame_stride)),
        resume=args.resume,
        cache_path=cache_path,
        mask_dirs=mask_dirs,
        require_mask=args.require_mask,
    )

    scenes_discovered = len(transforms)
    scenes_processed = int(df.shape[0])
    scenes_failed = len(failures)
    failed_pct = (100.0 * scenes_failed / scenes_discovered) if scenes_discovered else 0.0

    raw_csv_path = out_dir / args.raw_csv_name
    df.to_csv(raw_csv_path, index=False)
    LOGGER.info("Saved raw features CSV: %s", raw_csv_path)

    summary, stats_df, corr_df = compute_summary(df, id_cols=("dish_id", "scene_path"))

    missing_df = (
        stats_df[["missing_percent", "outlier_percent"]]
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_percent", ascending=False, kind="mergesort")
    )
    top_corr_df = _top_correlated_pairs(corr_df, top_k=50)
    failure_df = _failure_summary(failures)

    summary_path = out_dir / args.summary_json_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    LOGGER.info("Saved summary JSON: %s", summary_path)

    missing_csv_path = out_dir / args.missing_csv_name
    missing_df.to_csv(missing_csv_path, index=False)
    LOGGER.info("Saved missingness CSV: %s", missing_csv_path)

    corr_csv_path = out_dir / args.corr_csv_name
    corr_df.to_csv(corr_csv_path, index=True)
    LOGGER.info("Saved correlation CSV: %s", corr_csv_path)

    top_corr_csv_path = out_dir / args.top_corr_csv_name
    top_corr_df.to_csv(top_corr_csv_path, index=False)
    LOGGER.info("Saved top correlation pairs CSV: %s", top_corr_csv_path)

    failure_summary_csv_path = out_dir / args.failure_summary_csv_name
    failure_df.to_csv(failure_summary_csv_path, index=False)
    LOGGER.info("Saved failure summary CSV: %s", failure_summary_csv_path)

    failures_path = out_dir / args.failures_json_name
    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, sort_keys=True)
    LOGGER.info("Saved failures JSON: %s", failures_path)

    # Drift
    drift_report: Optional[Dict[str, Any]] = None
    baseline_summary_path = args.baseline_summary_json
    baseline_corr_path = args.baseline_corr_csv
    if args.baseline_output_dir is not None:
        baseline_summary_path = args.baseline_output_dir / args.summary_json_name
        baseline_corr_path = args.baseline_output_dir / args.corr_csv_name

    if baseline_summary_path is not None and baseline_summary_path.exists():
        drift_report = _compute_drift(
            summary,
            corr_df,
            baseline_summary_path=baseline_summary_path,
            baseline_corr_path=baseline_corr_path
            if baseline_corr_path is not None and baseline_corr_path.is_file()
            else None,
            mean_delta_threshold=args.drift_mean_delta_threshold,
            missing_delta_threshold=args.drift_missing_delta_threshold,
        )
        drift_path = out_dir / args.drift_json_name
        _write_report_json(drift_path, drift_report)
        LOGGER.info("Saved drift report JSON: %s", drift_path)

    report_payload = {
        "overview": {
            "scenes_discovered": scenes_discovered,
            "scenes_processed": scenes_processed,
            "scenes_failed": scenes_failed,
            "failed_scenes_percent": failed_pct,
            "num_features": summary["num_features"],
            "cache_hits": cache_hits,
            "max_frames_per_scene": args.max_frames_per_scene,
            "frame_stride": max(1, int(args.frame_stride)),
        },
        "artifacts": {
            "raw_csv": str(raw_csv_path.name),
            "summary_json": str(summary_path.name),
            "missing_csv": str(missing_csv_path.name),
            "corr_csv": str(corr_csv_path.name),
            "top_corr_csv": str(top_corr_csv_path.name),
            "failure_summary_csv": str(failure_summary_csv_path.name),
            "failures_json": str(failures_path.name),
        },
        "top_missing_features": missing_df.head(30).to_dict(orient="records"),
        "top_correlated_pairs": top_corr_df.head(30).to_dict(orient="records"),
        "failure_taxonomy": failure_df.to_dict(orient="records"),
        "top_failed_scenes": failures[:50],
        "drift_report": drift_report,
    }

    report_json_path = out_dir / args.report_json_name
    _write_report_json(report_json_path, report_payload)
    LOGGER.info("Saved report JSON: %s", report_json_path)

    report_md_path = out_dir / args.report_md_name
    write_markdown_report(
        report_md_path,
        scenes_discovered=scenes_discovered,
        scenes_processed=scenes_processed,
        scenes_failed=scenes_failed,
        num_features=summary["num_features"],
        missing_df=missing_df,
        top_corr_df=top_corr_df,
        failure_df=failure_df,
        failures=failures,
        drift_report=drift_report,
    )
    LOGGER.info("Saved markdown report: %s", report_md_path)

    LOGGER.info("Scenes loaded: %d", scenes_processed)
    LOGGER.info("Features extracted: %d", summary["num_features"])
    if scenes_failed > 0:
        LOGGER.warning("Scenes failed: %d (%.2f%%)", scenes_failed, failed_pct)

    # Quality gates
    if failed_pct > args.max_failed_scenes_pct:
        raise RuntimeError(
            f"Quality gate failed: failed scene rate {failed_pct:.2f}% > "
            f"{args.max_failed_scenes_pct:.2f}%"
        )

    worst_missing = float(missing_df["missing_percent"].max()) if not missing_df.empty else 0.0
    if worst_missing > args.max_feature_missing_pct:
        raise RuntimeError(
            f"Quality gate failed: max feature missingness {worst_missing:.2f}% > "
            f"{args.max_feature_missing_pct:.2f}%"
        )

    return report_payload


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
