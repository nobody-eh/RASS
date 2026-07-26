#!/usr/bin/env python3
"""Task P6 part a / key E6.a: descriptor-group ablation.

Groups inferred from the extraction code (src/feature_extractor.py), each
descriptor's source function being unambiguous in code:
- capture_pose: camera intrinsics/distortion/resolution, NeRF capture params,
  camera-center spread, view-angle spread, capture counts (incl. the 3
  budget proxies, which are ALWAYS excluded from clustering per protocol).
- image_stats: entropy_*, sharpness_* (image-quality statistics).
- texture: edge_frac_* (Sobel edge fraction; code comment labels it
  "texture/edge density").
- mask_geometry: compute_scene_foreground_stats outputs (area_frac_*,
  bbox_aspect_ratio_*, largest_comp_frac_*, num_components_*).
- sparse_geometry: COLMAP points3D/images.txt stats (point cloud bbox,
  extents, volume, density, count, reprojection errors, track lengths,
  visible points/ratios).

Judgment calls (documented, not blocking): aabb_scale -> capture_pose (NeRF
capture config); sharpness/entropy -> image_stats rather than texture.

Renormalization note: the skew-aware pipeline (feats_norm.normalize_features)
makes per-feature-independent decisions (log1p by per-feature skew,
robust/standard scale, clip), so the column subset of feats_normalized.csv
is exactly the output of renormalizing that subset with the same pipeline.

Per configuration: recluster at k=6 seed 0 (validated recipe), balanced
audit at 48 scenes (b=8) with M=400 plus coarse sweep b in {8,12,16,20},
Zip-NeRF event, paper tolerances, audit seed 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
)
from rebuttal_audit_tasks import merge_results_json  # noqa: E402
from rebuttal_e3_sensitivity import (  # noqa: E402
    FEATS,
    PROXIES,
    ZIPNERF_LOG,
    Sweeper,
    _wilson_lower_bound,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
COARSE_B = [8, 12, 16, 20]
P_MIN = 0.08
M = 400

GROUPS = {
    "capture_pose": [
        "aabb_scale", "camera_angle_x", "camera_angle_y", "cx", "cy",
        "fl_x", "fl_y", "h", "w", "k1", "k2", "k3", "p1", "p2",
        "mean_camera_center_dist", "std_camera_center_dist",
        "mean_view_angle_var", "std_view_angle_var",
        "num_images", "num_frames_total", "num_frames_used",
    ],
    "image_stats": ["entropy_mean", "entropy_std", "sharpness_mean", "sharpness_std"],
    "texture": ["edge_frac_mean", "edge_frac_std"],
    "mask_geometry": [
        "area_frac_mean", "area_frac_std", "bbox_aspect_ratio_mean",
        "bbox_aspect_ratio_std", "largest_comp_frac_mean",
        "largest_comp_frac_std", "num_components_mean", "num_components_std",
    ],
    "sparse_geometry": [
        "bbox_max_x", "bbox_max_y", "bbox_max_z", "bbox_min_x", "bbox_min_y",
        "bbox_min_z", "extent_x", "extent_y", "extent_z", "volume",
        "point_count", "point_density", "mean_error", "median_error",
        "std_error", "mean_track_length", "median_track_length",
        "std_track_length", "mean_visible_points", "std_visible_points",
        "mean_visible_ratio", "std_visible_ratio",
    ],
}


def cluster_with_cols(feats: pd.DataFrame, cols: list[str], seed: int = 0) -> pd.DataFrame:
    use = [c for c in cols if c not in PROXIES]
    X = feats[use].fillna(feats[use].mean())
    var = X.var(axis=0)
    X = X.drop(columns=var[var == 0.0].index.tolist())
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=6, random_state=seed, n_init="auto").fit_predict(Xs)
    return pd.DataFrame({"dish_id": feats["dish_id"].astype(str), "cluster": labels}), len(use)


def main() -> None:
    feats = pd.read_csv(FEATS)
    all_cols = [c for c in feats.columns if c != "dish_id"]
    mapped = [c for g in GROUPS.values() for c in g]
    assert sorted(mapped) == sorted(all_cols), (
        f"group mapping mismatch: missing {set(all_cols)-set(mapped)}, "
        f"extra {set(mapped)-set(all_cols)}"
    )
    print("Dimension-to-group mapping (counts):",
          {g: len(v) for g, v in GROUPS.items()}, "= 57 total")

    zip_df = _load_zipnerf_metrics(ZIPNERF_LOG)

    configs = {}
    for g, cols in GROUPS.items():
        configs[f"{g}_alone"] = cols
        configs[f"minus_{g}"] = [c for c in all_cols if c not in cols]
    configs["full_57d"] = all_cols  # reference row

    table = {}
    for name, cols in configs.items():
        mapping, n_used = cluster_with_cols(feats, cols)
        merged = _merge_mapping_and_metrics(mapping, zip_df)
        sw = Sweeper(merged)
        min_regime = min(len(g) for g in sw.groups.values())
        bs = [b for b in COARSE_B if b <= min_regime]
        frontier, rec = sw.recommended_budget(bs, audit_seed=0)
        at48 = next((r for r in frontier if r["budget_scenes"] == 48), None)
        table[name] = {
            "n_clustering_dims": n_used,
            "min_regime_size": min_regime,
            "pass_rate_48": at48["empirical_pass_rate"] if at48 else None,
            "lcb_48": at48["wilson_lcb_95"] if at48 else None,
            "coarse_frontier": frontier,
            "recommended_budget_coarse_grid": (
                rec if rec is not None else "not reached up to 120 (coarse grid)"
            ),
        }
        p48 = f"{at48['empirical_pass_rate']:.4f}/{at48['wilson_lcb_95']:.4f}" if at48 else "n/a"
        print(f"  {name:24s} dims={n_used:2d} minreg={min_regime:4d} "
              f"48sc rate/LCB={p48} rec={table[name]['recommended_budget_coarse_grid']}")

    e6a = {
        "group_mapping": GROUPS,
        "group_mapping_note": (
            "Derived from src/feature_extractor.py source functions; "
            "judgment calls documented: aabb_scale->capture_pose, "
            "sharpness/entropy->image_stats (not texture). Budget proxies "
            "(num_images, num_frames_total, num_frames_used) are excluded "
            "from clustering in every configuration, matching the validated "
            "protocol."
        ),
        "renormalization_note": (
            "feats_norm.normalize_features is per-feature independent "
            "(log1p by skew, robust/standard scale, clip), so column "
            "subsets of feats_normalized.csv are identical to renormalizing "
            "each subset with the same pipeline."
        ),
        "audit": "Zip-NeRF event, paper tolerances, M=400, audit seed 0, "
                 "coarse grid b in {8,12,16,20} (48-120 scenes)",
        "per_group_table": table,
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E6": {"a": e6a}})
    print(f"Wrote E6.a into {OUT_DIR / 'rebuttal_results.json'}")


if __name__ == "__main__":
    main()
