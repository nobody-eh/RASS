#!/usr/bin/env python3
"""Merge per-scene Nutrition5k feature JSON files into a single table.

Expected input:
    features_raw/{scene_id}.json

Outputs:
    1) features_raw_merged.csv
    2) summary_stats.json
    3) correlation_matrix.csv

This script is deterministic:
    - files are processed in sorted filename order
    - output columns are ordered deterministically
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("merge_nutrition5k_features")


CANONICAL_COLUMNS = [
    "image_quality_sharpness",
    "image_quality_entropy",
    "mask_geometry_area",
    "mask_geometry_compactness",
    "texture_density_mean",
    "texture_density_var",
    "texture_density_num_samples",
    "colmap_sparse_num_points",
    "colmap_sparse_avg_track_len",
    "colmap_sparse_reprojection_error",
    "pose_coverage_num_views",
    "pose_coverage_angular_spread",
]


def _to_float(value: Any) -> float:
    """Convert value to float; return np.nan when conversion is not possible."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number, bool)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return np.nan


def _first_dict(payload: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    for key in keys:
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _flatten_numeric(prefix: str, value: Any, out: Dict[str, float]) -> None:
    """Flatten nested numeric dicts into out using prefix_key naming."""
    if isinstance(value, dict):
        for k in sorted(value.keys()):
            next_prefix = f"{prefix}_{k}" if prefix else str(k)
            _flatten_numeric(next_prefix, value[k], out)
        return

    numeric_value = _to_float(value)
    if not np.isnan(numeric_value):
        out[prefix] = numeric_value


def _texture_density_features(payload: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    td = _first_dict(payload, ("texture_density", "texture", "texture_stats"))

    mean_val = _to_float(td.get("mean", payload.get("texture_density_mean")))
    var_val = _to_float(
        td.get("var", td.get("variance", payload.get("texture_density_var")))
    )
    num_samples_val = _to_float(
        td.get(
            "num_samples",
            td.get("count", td.get("n", payload.get("texture_density_num_samples"))),
        )
    )

    samples = td.get("samples")
    if isinstance(samples, list) and len(samples) > 0:
        arr = pd.to_numeric(pd.Series(samples), errors="coerce").dropna().to_numpy()
        if arr.size > 0:
            if np.isnan(mean_val):
                mean_val = float(arr.mean())
            if np.isnan(var_val):
                var_val = float(arr.var(ddof=0))
            if np.isnan(num_samples_val):
                num_samples_val = float(arr.size)

    out["texture_density_mean"] = mean_val
    out["texture_density_var"] = var_val
    out["texture_density_num_samples"] = num_samples_val
    return out


def _pose_baseline_features(pose_cov: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    baseline = pose_cov.get("baseline_stats")
    if isinstance(baseline, dict):
        _flatten_numeric("pose_coverage_baseline_stats", baseline, out)
    elif isinstance(baseline, list):
        arr = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
        if arr.size > 0:
            out["pose_coverage_baseline_stats_mean"] = float(arr.mean())
            out["pose_coverage_baseline_stats_var"] = float(arr.var(ddof=0))
            out["pose_coverage_baseline_stats_min"] = float(arr.min())
            out["pose_coverage_baseline_stats_max"] = float(arr.max())
            out["pose_coverage_baseline_stats_num_samples"] = float(arr.size)
    else:
        scalar = _to_float(baseline)
        if not np.isnan(scalar):
            out["pose_coverage_baseline_stats"] = scalar
    return out


def extract_scene_record(scene_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {"scene_id": scene_id}

    # camera_intrinsics_*
    camera_intrinsics = _first_dict(payload, ("camera_intrinsics", "intrinsics"))
    _flatten_numeric("camera_intrinsics", camera_intrinsics, record)
    for key, value in payload.items():
        if key.startswith("camera_intrinsics_"):
            record[key] = _to_float(value)

    # image quality
    image_quality = _first_dict(payload, ("image_quality",))
    record["image_quality_sharpness"] = _to_float(
        image_quality.get("sharpness", payload.get("image_quality_sharpness"))
    )
    record["image_quality_entropy"] = _to_float(
        image_quality.get("entropy", payload.get("image_quality_entropy"))
    )

    # mask geometry
    mask_geometry = _first_dict(payload, ("mask_geometry", "mask_stats"))
    record["mask_geometry_area"] = _to_float(
        mask_geometry.get("area", payload.get("mask_geometry_area"))
    )
    record["mask_geometry_compactness"] = _to_float(
        mask_geometry.get("compactness", payload.get("mask_geometry_compactness"))
    )

    # texture density
    record.update(_texture_density_features(payload))

    # COLMAP sparse geometry
    colmap_sparse = _first_dict(
        payload,
        ("colmap_sparse_geometry", "colmap_sparse", "sparse_geometry", "colmap"),
    )
    record["colmap_sparse_num_points"] = _to_float(
        colmap_sparse.get("num_points", colmap_sparse.get("point_count", payload.get("colmap_sparse_num_points")))
    )
    record["colmap_sparse_avg_track_len"] = _to_float(
        colmap_sparse.get(
            "avg_track_len",
            colmap_sparse.get("mean_track_length", payload.get("colmap_sparse_avg_track_len")),
        )
    )
    record["colmap_sparse_reprojection_error"] = _to_float(
        colmap_sparse.get(
            "reprojection_error",
            colmap_sparse.get("mean_reprojection_error", payload.get("colmap_sparse_reprojection_error")),
        )
    )

    # pose coverage
    pose_cov = _first_dict(payload, ("pose_coverage",))
    record["pose_coverage_num_views"] = _to_float(
        pose_cov.get("num_views", pose_cov.get("views", payload.get("pose_coverage_num_views")))
    )
    record["pose_coverage_angular_spread"] = _to_float(
        pose_cov.get(
            "angular_spread",
            pose_cov.get("view_angle_spread", payload.get("pose_coverage_angular_spread")),
        )
    )
    record.update(_pose_baseline_features(pose_cov))

    return record


def build_dataframe(input_dir: Path) -> Tuple[pd.DataFrame, int]:
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    records: List[Dict[str, Any]] = []
    num_parse_failures = 0

    for file_path in files:
        scene_id = file_path.stem
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                LOGGER.warning("Skipping %s: root JSON is not an object", file_path.name)
                num_parse_failures += 1
                continue
            records.append(extract_scene_record(scene_id, payload))
        except (json.JSONDecodeError, OSError) as err:
            LOGGER.warning("Failed to parse %s: %s", file_path.name, err)
            num_parse_failures += 1

    if not records:
        raise RuntimeError("No valid scene records were parsed from JSON files.")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("scene_id", kind="mergesort").reset_index(drop=True)

    # Ensure canonical columns always exist, even if fully missing.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure numeric feature columns are numeric dtype.
    for col in df.columns:
        if col == "scene_id":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deterministic column order:
    # scene_id, camera_intrinsics_* (sorted), canonical columns, any extra columns (sorted).
    camera_cols = sorted(c for c in df.columns if c.startswith("camera_intrinsics_"))
    canonical_existing = [c for c in CANONICAL_COLUMNS if c in df.columns]
    used = {"scene_id"} | set(camera_cols) | set(canonical_existing)
    extra_cols = sorted(c for c in df.columns if c not in used)
    ordered_cols = ["scene_id"] + camera_cols + canonical_existing + extra_cols
    df = df[ordered_cols]

    return df, num_parse_failures


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = output_dir / "features_raw_merged.csv"
    summary_json_path = output_dir / "summary_stats.json"
    corr_csv_path = output_dir / "correlation_matrix.csv"

    df.to_csv(merged_csv_path, index=False)

    feature_cols = [c for c in df.columns if c != "scene_id"]
    numeric_df = df[feature_cols]

    mean_s = numeric_df.mean(axis=0)
    std_s = numeric_df.std(axis=0, ddof=0)
    median_s = numeric_df.median(axis=0)
    q1_s = numeric_df.quantile(0.25, axis=0)
    q3_s = numeric_df.quantile(0.75, axis=0)
    iqr_s = q3_s - q1_s
    missing_pct_s = numeric_df.isna().mean(axis=0) * 100.0

    # Correlation matrix artifact.
    corr_df = numeric_df.corr(method="pearson")
    corr_df.to_csv(corr_csv_path, index=True)

    per_feature_stats: Dict[str, Dict[str, Any]] = {}
    for col in feature_cols:
        per_feature_stats[col] = {
            "mean": _nan_to_none(float(mean_s[col])),
            "std": _nan_to_none(float(std_s[col])),
            "median": _nan_to_none(float(median_s[col])),
            "iqr": _nan_to_none(float(iqr_s[col])),
            "missing_percent": _nan_to_none(float(missing_pct_s[col])),
        }

    summary_payload = {
        "num_scenes": int(df.shape[0]),
        "num_features": int(len(feature_cols)),
        "per_feature_stats": per_feature_stats,
        "correlation_matrix_csv": str(corr_csv_path.name),
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)

    LOGGER.info("Wrote merged features CSV: %s", merged_csv_path)
    LOGGER.info("Wrote summary stats JSON: %s", summary_json_path)
    LOGGER.info("Wrote correlation matrix CSV: %s", corr_csv_path)

    # Missing-data warnings.
    missing_counts = numeric_df.isna().sum(axis=0)
    for col in feature_cols:
        missing_count = int(missing_counts[col])
        if missing_count > 0:
            pct = float(missing_pct_s[col])
            LOGGER.warning(
                "Missing data: %s -> %.2f%% (%d/%d)",
                col,
                pct,
                missing_count,
                df.shape[0],
            )

    LOGGER.info("Number of scenes loaded: %d", df.shape[0])
    LOGGER.info("Number of features extracted: %d", len(feature_cols))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Nutrition5k per-scene feature JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("features_raw"),
        help="Directory with one {scene_id}.json per scene (default: features_raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where artifacts are written (default: current directory).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Loading JSON files from: %s", args.input_dir)
    df, num_parse_failures = build_dataframe(args.input_dir)
    if num_parse_failures > 0:
        LOGGER.warning("JSON parse/validation failures: %d", num_parse_failures)

    write_outputs(df, args.output_dir)


if __name__ == "__main__":
    main()
