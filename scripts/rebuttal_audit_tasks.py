#!/usr/bin/env python3
"""Rebuttal audit tasks for NeurIPS 2026 submission 675 (discussion period).

Task A (full_budget_sweep): reliability frontier over the full budget sweep
b in {4, 6, 8, 10, 12, 14, 16, 20} (subset sizes 24..120) on the Zip-NeRF
audit population, reusing the validated machinery in
generate_strong_accept_results.py (balanced generator, M=400, seed 0).

Task B (multi_method_frontier): same generator restricted to the common
cross-method intersection (Zip-NeRF, Feature-Splatting, Instant-NGP full
image; Instant-NGP object-centric diagnostic only). A candidate passes the
multi-method joint event iff every formal method passes its per-metric mean
and KS tolerances against its own reference population on the intersection.

Both tasks validate against the paper reference values before reporting:
b=8 -> 88/400 (LCB 0.1822), b=16 -> 113/400 (LCB 0.2406).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    MethodSpec,
    _build_groups,
    _canonicalize_method_table,
    _load_mapping_csv,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _metric_eval_for_ids,
    _simulate_trials,
    _wilson_lower_bound,
)

REPO = Path(__file__).resolve().parents[1]

ZIPNERF_LOG = REPO / "sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx"
FS_LOG = REPO / "sweep_cluster_k/holdout_protocol_v2/joint_selection/feature_splatting.csv"
INGP_FI_LOG = REPO / "ingp_fi.csv"
INGP_OC_LOG = REPO / "ingp_oc.csv"
K6_MAPPING = REPO / "sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv"
DESCRIPTORS = REPO / "sweep_cluster_k/k_6/feats_normalized.csv"

THRESHOLDS = {"psnr_tol": 0.5, "ssim_tol": 0.01, "lpips_tol": 0.01, "ks_tol": 0.14}
BUDGETS = [4, 6, 8, 10, 12, 14, 16, 20]
NUM_TRIALS = 400
SEED = 0
P_MIN = 0.08
CONFIDENCE = 0.95

REFERENCE = {8: 88, 16: 113}

OUT_DIR = REPO / "rebuttal"
SUMMARY_DIR = OUT_DIR / "summaries"
MANIFEST_DIR = OUT_DIR / "manifests"


def extract_dish_id(value: str) -> str:
    import re

    m = re.search(r"(dish_[0-9a-zA-Z]+)", str(value))
    return m.group(1) if m else str(value).strip()


def merge_results_json(path: Path, new_entries: dict) -> None:
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(new_entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def load_inputs():
    inventory = []

    def check(label, path):
        if not path.exists():
            raise SystemExit(f"MISSING INPUT: {label}: {path}")

    for label, path in [
        ("Zip-NeRF log", ZIPNERF_LOG),
        ("Feature-Splatting log", FS_LOG),
        ("Instant-NGP full-image log", INGP_FI_LOG),
        ("Instant-NGP object-centric log", INGP_OC_LOG),
        ("k=6 regime mapping", K6_MAPPING),
        ("Normalized descriptors", DESCRIPTORS),
    ]:
        check(label, path)

    mapping_df = _load_mapping_csv(K6_MAPPING)
    zip_df = _load_zipnerf_metrics(ZIPNERF_LOG)
    full_k6_df = _merge_mapping_and_metrics(mapping_df, zip_df)

    fs_table = _canonicalize_method_table(
        MethodSpec(name="feature-splatting", path=FS_LOG)
    ).df
    fi_table = _canonicalize_method_table(
        MethodSpec(name="instant-ngp-fi", path=INGP_FI_LOG)
    ).df
    oc_table = _canonicalize_method_table(
        MethodSpec(name="instant-ngp-oc", path=INGP_OC_LOG)
    ).df
    for tbl in (fs_table, fi_table, oc_table):
        tbl["dish_id"] = tbl["scene_id"].map(extract_dish_id)

    descriptors_rows = len(pd.read_csv(DESCRIPTORS))

    inventory.append(("descriptors (feats_normalized)", descriptors_rows))
    inventory.append(("k=6 regime mapping", len(mapping_df)))
    inventory.append(("Zip-NeRF log (unique scenes)", len(zip_df)))
    inventory.append(("Zip-NeRF x mapping audit population", len(full_k6_df)))
    inventory.append(("Feature-Splatting (unique scenes)", fs_table["dish_id"].nunique()))
    inventory.append(("Instant-NGP FI (unique scenes)", fi_table["dish_id"].nunique()))
    inventory.append(("Instant-NGP OC (unique scenes)", oc_table["dish_id"].nunique()))

    common_ids = (
        set(full_k6_df["dish_id"])
        & set(fs_table["dish_id"])
        & set(fi_table["dish_id"])
        & set(oc_table["dish_id"])
    )
    inventory.append(("common 4-way intersection", len(common_ids)))

    print("== INPUT INVENTORY ==")
    for label, count in inventory:
        print(f"  {label:42s} {count}")

    if descriptors_rows != 3522:
        raise SystemExit(f"STOP: descriptors rows {descriptors_rows} != 3522")
    if len(mapping_df) != 3522:
        raise SystemExit(f"STOP: mapping rows {len(mapping_df)} != 3522")
    if len(full_k6_df) != 3521:
        raise SystemExit(
            f"STOP: Zip-NeRF audit population {len(full_k6_df)} != 3521 "
            "(3522 descriptors minus dish_1563900172 absent from the log)"
        )
    if len(common_ids) != 3473:
        raise SystemExit(f"STOP: cross-method intersection {len(common_ids)} != 3473")

    return mapping_df, full_k6_df, fs_table, fi_table, oc_table, sorted(common_ids), inventory


# ---------------------------------------------------------------------------
# Task A: full budget sweep on the Zip-NeRF population
# ---------------------------------------------------------------------------

def run_full_budget_sweep(full_k6_df: pd.DataFrame) -> dict:
    groups = _build_groups(full_k6_df)
    rows = []
    all_trials = []
    for b in BUDGETS:
        trial_df, best_joint, _best_mean = _simulate_trials(
            full_k6_df, groups, b, NUM_TRIALS, SEED, "balanced", THRESHOLDS
        )
        n_pass = int(trial_df["joint_pass"].sum())
        p_hat = n_pass / NUM_TRIALS
        lcb = _wilson_lower_bound(p_hat, NUM_TRIALS, CONFIDENCE)
        best = best_joint["stats"]
        manifest_path = MANIFEST_DIR / f"rebuttal_full_sweep_b{b}_best_subset.csv"
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        full_k6_df[full_k6_df["dish_id"].isin(set(best_joint["subset_ids"]))][
            ["dish_id", "cluster"]
        ].sort_values(["cluster", "dish_id"], kind="mergesort").to_csv(
            manifest_path, index=False
        )
        rows.append(
            {
                "budget_scenes": 6 * b,
                "b": b,
                "n_trials": NUM_TRIALS,
                "n_pass": n_pass,
                "empirical_pass_rate": p_hat,
                "wilson_lcb_95": lcb,
                "best_joint_abs_psnr_gap": best["abs_psnr_gap"],
                "best_joint_abs_ssim_gap": best["abs_ssim_gap"],
                "best_joint_abs_lpips_gap": best["abs_lpips_gap"],
                "best_joint_max_ks": best["max_ks"],
                "best_subset_manifest": str(manifest_path.relative_to(REPO)),
                "rng_seed_effective": SEED + b,
            }
        )
        all_trials.append(trial_df)
        print(
            f"  b={b:2d} ({6*b:3d} scenes): {n_pass:3d}/400 pass "
            f"(rate {p_hat:.4f}, LCB {lcb:.4f})"
        )

    for b, expected in REFERENCE.items():
        got = next(r["n_pass"] for r in rows if r["b"] == b)
        if got != expected:
            raise SystemExit(
                f"VALIDATION FAILED: b={b} produced {got}/400, expected {expected}/400"
            )
    print("  [validation] b=8 -> 88/400 and b=16 -> 113/400 reproduced exactly")

    first_meeting = next(
        (r for r in rows if r["wilson_lcb_95"] >= P_MIN), None
    )
    pd.concat(all_trials).to_csv(OUT_DIR / "full_budget_sweep_trials.csv", index=False)
    frontier_df = pd.DataFrame(rows)
    frontier_df.to_csv(OUT_DIR / "full_budget_sweep_frontier.csv", index=False)

    return {
        "description": (
            "Reliability frontier over the full budget sweep b in {4,...,20} "
            "(subset sizes 24-120), balanced generator on k=6 regimes, M=400, "
            "seed 0, Zip-NeRF audit population of 3521 scenes (3522 descriptor "
            "scenes minus dish_1563900172, absent from the Zip-NeRF log). "
            "Joint pass event: |dPSNR|<=0.5, |dSSIM|<=0.01, |dLPIPS|<=0.01, "
            "max KS<=0.14 vs the full-population empirical CDFs."
        ),
        "population_size": 3521,
        "thresholds": THRESHOLDS,
        "num_trials": NUM_TRIALS,
        "seed": SEED,
        "per_budget_rng_seed_rule": "numpy default_rng(seed + b) for balanced mode",
        "p_min": P_MIN,
        "confidence_level": CONFIDENCE,
        "validated_against": {"b8_n_pass": 88, "b16_n_pass": 113},
        "frontier": rows,
        "first_budget_with_lcb_at_or_above_p_min": (
            {
                "budget_scenes": first_meeting["budget_scenes"],
                "wilson_lcb_95": first_meeting["wilson_lcb_95"],
            }
            if first_meeting
            else None
        ),
        "outputs": {
            "frontier_csv": "rebuttal/full_budget_sweep_frontier.csv",
            "trials_csv": "rebuttal/full_budget_sweep_trials.csv",
        },
    }


# ---------------------------------------------------------------------------
# Task B: multi-method frontier on the 3473-scene intersection
# ---------------------------------------------------------------------------

def eval_method_subset(
    pop_df: pd.DataFrame, subset_ids: set, metrics: list[str]
) -> dict:
    subset = pop_df[pop_df["dish_id"].isin(subset_ids)]
    out = {}
    all_pass = True
    for metric in metrics:
        full_vals = pop_df[metric].to_numpy(dtype=float)
        sub_vals = subset[metric].to_numpy(dtype=float)
        gap = abs(float(np.mean(sub_vals) - np.mean(full_vals)))
        ks = float(ks_2samp(full_vals, sub_vals).statistic)
        tol = THRESHOLDS[f"{metric}_tol"]
        m_pass = gap <= tol and ks <= THRESHOLDS["ks_tol"]
        out[f"abs_{metric}_gap"] = gap
        out[f"ks_{metric}"] = ks
        all_pass = all_pass and m_pass
    out["pass"] = all_pass
    return out


def run_multi_method_frontier(
    mapping_df: pd.DataFrame,
    full_k6_df: pd.DataFrame,
    fs_table: pd.DataFrame,
    fi_table: pd.DataFrame,
    oc_table: pd.DataFrame,
    common_ids: list[str],
) -> dict:
    common_set = set(common_ids)

    # Sampling population: intersection scenes with regime labels, in the same
    # row order convention as _merge_mapping_and_metrics (mapping order).
    pop_df = full_k6_df[full_k6_df["dish_id"].isin(common_set)].reset_index(drop=True)
    groups = _build_groups(pop_df)
    cluster_sizes = {c: len(v) for c, v in groups.items()}
    print(f"  intersection population: {len(pop_df)} scenes; regime sizes {cluster_sizes}")
    if min(cluster_sizes.values()) < max(BUDGETS):
        raise SystemExit("STOP: a regime has fewer scenes than the largest budget")

    method_pops = {
        "zipnerf": (
            pop_df[["dish_id", "psnr", "ssim", "lpips"]].copy(),
            ["psnr", "ssim", "lpips"],
        ),
        "feature_splatting": (
            fs_table[fs_table["dish_id"].isin(common_set)][
                ["dish_id", "psnr", "ssim", "lpips"]
            ].reset_index(drop=True),
            ["psnr", "ssim", "lpips"],
        ),
        "instant_ngp_fi": (
            fi_table[fi_table["dish_id"].isin(common_set)][
                ["dish_id", "psnr", "ssim"]
            ].reset_index(drop=True),
            ["psnr", "ssim"],
        ),
    }
    diagnostic_pops = {
        "instant_ngp_oc": (
            oc_table[oc_table["dish_id"].isin(common_set)][
                ["dish_id", "psnr", "ssim"]
            ].reset_index(drop=True),
            ["psnr", "ssim"],
        ),
    }
    for name, (df, metrics) in {**method_pops, **diagnostic_pops}.items():
        n_nan = int(df[metrics].isna().sum().sum())
        if len(df) != 3473 or n_nan:
            raise SystemExit(
                f"STOP: {name} intersection table has {len(df)} rows / {n_nan} NaNs"
            )

    # Validation of the replicated generator+evaluator: on the full 3521-scene
    # Zip-NeRF population with Zip-NeRF alone, b=8 must give 88/400.
    val_groups = _build_groups(full_k6_df)
    val_pop = full_k6_df[["dish_id", "psnr", "ssim", "lpips"]].copy()
    dish_ids_val = full_k6_df["dish_id"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED + 8)
    n_pass_val = 0
    for _trial in range(NUM_TRIALS):
        picks = [
            rng.choice(val_groups[c], size=8, replace=False) for c in sorted(val_groups)
        ]
        idx = np.sort(np.concatenate(picks))
        subset_ids = set(dish_ids_val[idx].tolist())
        if eval_method_subset(val_pop, subset_ids, ["psnr", "ssim", "lpips"])["pass"]:
            n_pass_val += 1
    if n_pass_val != 88:
        raise SystemExit(
            f"VALIDATION FAILED: replicated generator gives {n_pass_val}/400 at b=8, expected 88"
        )
    print("  [validation] replicated generator+evaluator reproduces 88/400 at b=8")

    dish_ids_pop = pop_df["dish_id"].astype(str).to_numpy()
    rows = []
    trial_rows = []
    for b in BUDGETS:
        rng = np.random.default_rng(SEED + b)
        n_joint = 0
        n_per_method = {name: 0 for name in method_pops}
        n_oc = 0
        for trial in range(NUM_TRIALS):
            picks = [
                rng.choice(groups[c], size=b, replace=False) for c in sorted(groups)
            ]
            idx = np.sort(np.concatenate(picks))
            subset_ids = set(dish_ids_pop[idx].tolist())

            trial_rec = {"b": b, "budget_scenes": 6 * b, "trial": trial}
            joint = True
            for name, (mdf, metrics) in method_pops.items():
                res = eval_method_subset(mdf, subset_ids, metrics)
                n_per_method[name] += int(res["pass"])
                joint = joint and res["pass"]
                for key, val in res.items():
                    trial_rec[f"{name}_{key}"] = val
            oc_df, oc_metrics = diagnostic_pops["instant_ngp_oc"]
            oc_res = eval_method_subset(oc_df, subset_ids, oc_metrics)
            n_oc += int(oc_res["pass"])
            for key, val in oc_res.items():
                trial_rec[f"instant_ngp_oc_{key}"] = val
            n_joint += int(joint)
            trial_rec["joint_pass"] = joint
            trial_rows.append(trial_rec)

        p_hat = n_joint / NUM_TRIALS
        lcb = _wilson_lower_bound(p_hat, NUM_TRIALS, CONFIDENCE)
        row = {
            "budget_scenes": 6 * b,
            "b": b,
            "n_trials": NUM_TRIALS,
            "n_joint_pass": n_joint,
            "joint_pass_rate": p_hat,
            "joint_wilson_lcb_95": lcb,
            "rng_seed_effective": SEED + b,
        }
        for name, count in n_per_method.items():
            row[f"{name}_n_pass"] = count
            row[f"{name}_pass_rate"] = count / NUM_TRIALS
        row["instant_ngp_oc_diagnostic_n_pass"] = n_oc
        row["instant_ngp_oc_diagnostic_pass_rate"] = n_oc / NUM_TRIALS
        rows.append(row)
        per_m = ", ".join(f"{n}={c}" for n, c in n_per_method.items())
        print(
            f"  b={b:2d} ({6*b:3d} scenes): joint {n_joint:3d}/400 "
            f"(LCB {lcb:.4f}) | per-method: {per_m}"
        )

    first_meeting = next((r for r in rows if r["joint_wilson_lcb_95"] >= P_MIN), None)
    pd.DataFrame(trial_rows).to_csv(
        OUT_DIR / "multi_method_frontier_trials.csv", index=False
    )
    pd.DataFrame(rows).to_csv(OUT_DIR / "multi_method_frontier.csv", index=False)

    return {
        "description": (
            "Multi-method reliability frontier on the 3473-scene common "
            "intersection (Zip-NeRF full image PSNR/SSIM/LPIPS, "
            "Feature-Splatting full image PSNR/SSIM/LPIPS, Instant-NGP full "
            "image PSNR/SSIM; Instant-NGP object-centric excluded from the "
            "formal event per protocol). Balanced generator on k=6 regimes "
            "restricted to the intersection, M=400, seed 0. Joint event: all "
            "formal methods pass every per-metric mean tolerance and KS<=0.14 "
            "against their own reference population on the intersection."
        ),
        "population_size": 3473,
        "formal_methods": {
            "zipnerf": ["psnr", "ssim", "lpips"],
            "feature_splatting": ["psnr", "ssim", "lpips"],
            "instant_ngp_fi": ["psnr", "ssim"],
        },
        "diagnostic_only": ["instant_ngp_oc"],
        "thresholds": THRESHOLDS,
        "num_trials": NUM_TRIALS,
        "seed": SEED,
        "per_budget_rng_seed_rule": "numpy default_rng(seed + b) for balanced mode",
        "p_min": P_MIN,
        "confidence_level": CONFIDENCE,
        "generator_validation": "replicated generator reproduces 88/400 at b=8 on the 3521-scene Zip-NeRF population",
        "frontier": rows,
        "first_budget_with_joint_lcb_at_or_above_p_min": (
            {
                "budget_scenes": first_meeting["budget_scenes"],
                "joint_wilson_lcb_95": first_meeting["joint_wilson_lcb_95"],
            }
            if first_meeting
            else None
        ),
        "outputs": {
            "frontier_csv": "rebuttal/multi_method_frontier.csv",
            "trials_csv": "rebuttal/multi_method_frontier_trials.csv",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    mapping_df, full_k6_df, fs_table, fi_table, oc_table, common_ids, inventory = load_inputs()

    print("\n== TASK A: full budget sweep (Zip-NeRF population, 3521 scenes) ==")
    task_a = run_full_budget_sweep(full_k6_df)

    print("\n== TASK B: multi-method frontier (3473-scene intersection) ==")
    task_b = run_multi_method_frontier(
        mapping_df, full_k6_df, fs_table, fi_table, oc_table, common_ids
    )

    merge_results_json(
        OUT_DIR / "rebuttal_results.json",
        {"full_budget_sweep": task_a, "multi_method_frontier": task_b},
    )
    print(f"\nWrote {OUT_DIR / 'rebuttal_results.json'}")


if __name__ == "__main__":
    main()
