#!/usr/bin/env python3
"""Task P1 / key E1: formal joint multi-method fidelity event.

Population: the 3,473-scene common intersection. All reference means and
empirical CDFs are recomputed per method ON the intersection.

E_joint(S): for each method-metric pair (Zip-NeRF PSNR/SSIM/LPIPS,
Feature-Splatting PSNR/SSIM/LPIPS, Instant-NGP full-image PSNR/SSIM):
|mean(S) - mean(pop)| <= tol AND KS(S, pop) <= 0.14. 8 mean + 8 KS
constraints. Instant-NGP object-centric excluded.

Instant-NGP PSNR uses the per-frame-mean column ('PNSR'), the convention
that reproduces the paper's cross-method table (established in P0).

Generator: balanced sampler, k=6 labels restricted to the intersection,
M=400, b in {4,6,8,10,12,14,16,20}, rng = default_rng(seed + b), seed 0.
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
    _load_subset_ids,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import (  # noqa: E402
    BUDGETS,
    NUM_TRIALS,
    SEED,
    THRESHOLDS,
    extract_dish_id,
    merge_results_json,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
P_MIN_LOW, P_MIN_HIGH = 0.08, 0.20

ZIPNERF_LOG = REPO / "sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx"
FS_LOG = REPO / "sweep_cluster_k/holdout_protocol_v2/joint_selection/feature_splatting.csv"
INGP_FI_LOG = REPO / "ingp_fi.csv"
INGP_OC_LOG = REPO / "ingp_oc.csv"
K6_MAPPING = REPO / "sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv"
RASS48 = REPO / "sweep_cluster_k/budget_sweep_k6_auto_v2/recommended_subset.csv"
RASS96 = REPO / "results/strong_accept/manifests/bass_joint_k6_b16_best_subset.csv"

METHOD_METRICS = {
    "zipnerf": ["psnr", "ssim", "lpips"],
    "feature_splatting": ["psnr", "ssim", "lpips"],
    "instant_ngp_fi": ["psnr", "ssim"],
}


def load_populations():
    mapping_df = _load_mapping_csv(K6_MAPPING)
    zip_df = _load_zipnerf_metrics(ZIPNERF_LOG)
    full_k6_df = _merge_mapping_and_metrics(mapping_df, zip_df)

    fs = _canonicalize_method_table(MethodSpec("feature-splatting", FS_LOG)).df
    fs["dish_id"] = fs["scene_id"].map(extract_dish_id)

    fi_raw = pd.read_csv(INGP_FI_LOG)
    fi_raw["dish_id"] = fi_raw["dish_id"].map(extract_dish_id)
    fi = (
        pd.DataFrame(
            {"dish_id": fi_raw["dish_id"], "psnr": fi_raw["PNSR"], "ssim": fi_raw["SSIM"]}
        )
        .groupby("dish_id", as_index=False)
        .mean()
    )

    oc_raw = pd.read_csv(INGP_OC_LOG)
    oc_raw["dish_id"] = oc_raw["dish_id"].map(extract_dish_id)

    common = (
        set(full_k6_df["dish_id"])
        & set(fs["dish_id"])
        & set(fi["dish_id"])
        & set(oc_raw["dish_id"])
    )
    if len(common) != 3473:
        raise SystemExit(f"STOP: intersection {len(common)} != 3473")

    pop_k6 = full_k6_df[full_k6_df["dish_id"].isin(common)].reset_index(drop=True)
    pops = {
        "zipnerf": pop_k6[["dish_id", "psnr", "ssim", "lpips"]].copy(),
        "feature_splatting": fs[fs["dish_id"].isin(common)][
            ["dish_id", "psnr", "ssim", "lpips"]
        ].reset_index(drop=True),
        "instant_ngp_fi": fi[fi["dish_id"].isin(common)].reset_index(drop=True),
    }
    for name, (df) in pops.items():
        metrics = METHOD_METRICS[name]
        if len(df) != 3473 or int(df[metrics].isna().sum().sum()):
            raise SystemExit(f"STOP: {name} population invalid")
    return pop_k6, pops, common


def constraint_table(pops: dict, subset_ids: set) -> tuple[list[dict], bool]:
    """Evaluate all 16 constraints for a subset; return per-constraint rows."""
    rows = []
    joint = True
    for name, metrics in METHOD_METRICS.items():
        pop = pops[name]
        sub = pop[pop["dish_id"].isin(subset_ids)]
        for metric in metrics:
            full_vals = pop[metric].to_numpy(dtype=float)
            sub_vals = sub[metric].to_numpy(dtype=float)
            gap = abs(float(np.mean(sub_vals) - np.mean(full_vals)))
            ks = float(ks_2samp(full_vals, sub_vals).statistic)
            tol = THRESHOLDS[f"{metric}_tol"]
            mean_ok = gap <= tol
            ks_ok = ks <= THRESHOLDS["ks_tol"]
            rows.append(
                {
                    "method": name,
                    "metric": metric,
                    "abs_mean_gap": gap,
                    "mean_tolerance": tol,
                    "mean_pass": mean_ok,
                    "ks": ks,
                    "ks_tolerance": THRESHOLDS["ks_tol"],
                    "ks_pass": ks_ok,
                }
            )
            joint = joint and mean_ok and ks_ok
    return rows, joint


def main() -> None:
    pop_k6, pops, common = load_populations()
    groups = _build_groups(pop_k6)
    dish_ids_pop = pop_k6["dish_id"].astype(str).to_numpy()

    # Regime support check.
    support = {int(c): int(len(v)) for c, v in groups.items()}
    max_b = max(BUDGETS)
    min_support = min(support.values())
    support_ok = min_support >= max_b
    regime_support_note = (
        f"Per-regime support on the 3,473-scene intersection: {support}. "
        f"Minimum regime size {min_support} >= largest budget b={max_b} "
        f"(and >= 20x b={max_b} = {20 * max_b} for regimes "
        f"{[c for c, n in support.items() if n >= 20 * max_b]}); "
        "balanced sampling without replacement is well-supported at every budget."
    )
    print(f"Regime support: {support} (min {min_support}, need >= {max_b})")
    if not support_ok:
        raise SystemExit("STOP: regime support insufficient for largest budget")

    # ---------------------------------------------------------------- sweep
    frontier = []
    per_method_at = {}  # (b, method) -> n_pass
    binding_counts = {}  # constraint label -> count, at b=8
    n_fail_48 = 0
    trial_records = []

    for b in BUDGETS:
        rng = np.random.default_rng(SEED + b)
        n_joint = 0
        n_method_pass = {name: 0 for name in METHOD_METRICS}
        for trial in range(NUM_TRIALS):
            picks = [
                rng.choice(groups[c], size=b, replace=False) for c in sorted(groups)
            ]
            idx = np.sort(np.concatenate(picks))
            subset_ids = set(dish_ids_pop[idx].tolist())

            rows, joint = constraint_table(pops, subset_ids)
            method_pass = {}
            for name in METHOD_METRICS:
                m_rows = [r for r in rows if r["method"] == name]
                method_pass[name] = all(r["mean_pass"] and r["ks_pass"] for r in m_rows)
                n_method_pass[name] += int(method_pass[name])
            n_joint += int(joint)

            rec = {"b": b, "budget_scenes": 6 * b, "trial": trial, "joint_pass": joint}
            for r in rows:
                key = f"{r['method']}_{r['metric']}"
                rec[f"{key}_abs_mean_gap"] = r["abs_mean_gap"]
                rec[f"{key}_ks"] = r["ks"]
                rec[f"{key}_mean_pass"] = r["mean_pass"]
                rec[f"{key}_ks_pass"] = r["ks_pass"]
            trial_records.append(rec)

            if b == 8 and not joint:
                n_fail_48 += 1
                for r in rows:
                    if not r["mean_pass"]:
                        k = f"{r['method']}.{r['metric']}.mean"
                        binding_counts[k] = binding_counts.get(k, 0) + 1
                    if not r["ks_pass"]:
                        k = f"{r['method']}.{r['metric']}.ks"
                        binding_counts[k] = binding_counts.get(k, 0) + 1

        p_hat = n_joint / NUM_TRIALS
        lcb = _wilson_lower_bound(p_hat, NUM_TRIALS, 0.95)
        frontier.append(
            {
                "budget_scenes": 6 * b,
                "b": b,
                "n_trials": NUM_TRIALS,
                "n_pass": n_joint,
                "empirical_pass_rate": p_hat,
                "wilson_lcb_95": lcb,
                "rng_seed_effective": SEED + b,
            }
        )
        for name, n in n_method_pass.items():
            per_method_at[(b, name)] = n
        pm = ", ".join(f"{n}={c}" for n, c in n_method_pass.items())
        print(f"b={b:2d} ({6*b:3d} scenes): joint {n_joint:3d}/400 (LCB {lcb:.4f}) | {pm}")

    def first_budget(p_min):
        for row in frontier:
            if row["wilson_lcb_95"] >= p_min:
                return row
        return None

    row008 = first_budget(P_MIN_LOW)
    row020 = first_budget(P_MIN_HIGH)

    # ---------------------------------------------------------------- c: RASS subsets
    rass_results = {}
    for label, path in (("rass48", RASS48), ("rass96", RASS96)):
        ids = set(_load_subset_ids(path))
        inside = ids & common
        rows, joint = constraint_table(pops, inside)
        rass_results[label] = {
            "nominal_size": len(ids),
            "scenes_in_intersection": len(inside),
            "pass": bool(joint),
            "constraints": rows,
        }
        n_viol = sum((not r["mean_pass"]) + (not r["ks_pass"]) for r in rows)
        print(f"{label}: {len(inside)}/{len(ids)} scenes in intersection, "
              f"E_joint {'PASS' if joint else 'FAIL'} ({n_viol} violated constraints)")

    # ---------------------------------------------------------------- d/f: single-method
    def sm(b, name):
        n = per_method_at[(b, name)]
        p = n / NUM_TRIALS
        return n, p, _wilson_lower_bound(p, NUM_TRIALS, 0.95)

    single_method = {}
    for key, name in (
        ("ingp", "instant_ngp_fi"),
        ("fs", "feature_splatting"),
        ("zip_on_intersection", "zipnerf"),
    ):
        n48, p48, l48 = sm(8, name)
        n96, p96, l96 = sm(16, name)
        single_method[key] = {
            "n48": n48, "p48": p48, "lcb48": l48,
            "n96": n96, "p96": p96, "lcb96": l96,
        }
        print(f"single-method {key}: 48sc {n48}/400 (LCB {l48:.4f}), "
              f"96sc {n96}/400 (LCB {l96:.4f})")

    # ---------------------------------------------------------------- e: binding
    binding_histogram = dict(
        sorted(binding_counts.items(), key=lambda kv: -kv[1])
    )
    print(f"binding constraints at 48 scenes over {n_fail_48} failing candidates:")
    for k, v in binding_histogram.items():
        print(f"  {k:38s} {v}")

    # ---------------------------------------------------------------- output
    pd.DataFrame(trial_records).to_csv(OUT_DIR / "e1_joint_event_trials.csv", index=False)
    pd.DataFrame(frontier).to_csv(OUT_DIR / "e1_joint_frontier.csv", index=False)

    e1 = {
        "description": (
            "Formal joint multi-method fidelity event E_joint on the "
            "3,473-scene common intersection: 8 mean + 8 KS constraints "
            "(Zip-NeRF PSNR/SSIM/LPIPS, Feature-Splatting PSNR/SSIM/LPIPS, "
            "Instant-NGP full-image PSNR/SSIM), all references recomputed on "
            "the intersection. Instant-NGP object-centric excluded. Balanced "
            "generator, k=6 regimes restricted to the intersection, M=400, "
            "seed 0, rng=default_rng(seed+b). Instant-NGP PSNR uses the "
            "per-frame-mean 'PNSR' column (paper convention per P0); the "
            "earlier exploratory multi_method_frontier key used the "
            "'psnr avgmse' column instead."
        ),
        "population_size": 3473,
        "thresholds": THRESHOLDS,
        "num_trials": NUM_TRIALS,
        "seed": SEED,
        "per_budget_rng_seed_rule": "numpy default_rng(seed + b), balanced mode",
        "frontier": frontier,
        "budget_p008": row008["budget_scenes"] if row008 else None,
        "rate_at_budget": row008["empirical_pass_rate"] if row008 else None,
        "lcb_at_budget": row008["wilson_lcb_95"] if row008 else None,
        "budget_p020": row020["budget_scenes"] if row020 else None,
        "rass48_pass": rass_results["rass48"]["pass"],
        "rass48_scenes_in_intersection": rass_results["rass48"]["scenes_in_intersection"],
        "rass48_constraints": rass_results["rass48"]["constraints"],
        "rass96_pass": rass_results["rass96"]["pass"],
        "rass96_scenes_in_intersection": rass_results["rass96"]["scenes_in_intersection"],
        "rass96_constraints": rass_results["rass96"]["constraints"],
        "single_method": single_method,
        "binding_histogram": binding_histogram,
        "binding_histogram_n_failing_candidates": n_fail_48,
        "regime_support_note": regime_support_note,
        "outputs": {
            "frontier_csv": "rebuttal/e1_joint_frontier.csv",
            "trials_csv": "rebuttal/e1_joint_event_trials.csv",
        },
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E1": e1})
    print(f"\nWrote E1 key into {OUT_DIR / 'rebuttal_results.json'}")


if __name__ == "__main__":
    main()
