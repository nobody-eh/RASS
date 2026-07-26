#!/usr/bin/env python3
"""Task P0: validate the audit pipeline before any new experiment.

1. Inventory of all audit inputs (paths, row counts, columns).
2. Reproduction checks using the repo's own audit code:
   a. RASS-48 export audit vs the full Zip-NeRF population.
   b. Balanced-generator sweep at 48 and 96 scenes (M=400, seed protocol).
   c. RASS-48 cross-method mean gaps on the 3,473-scene intersection.
3. Matching: exact where deterministic; 0.001 absolute otherwise.

Writes rebuttal/rebuttal_results.json key "P0" (merged) and
rebuttal/summaries/P0.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    MethodSpec,
    _build_groups,
    _canonicalize_method_table,
    _load_mapping_csv,
    _load_subset_ids,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _metric_eval_for_ids,
    _simulate_trials,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import THRESHOLDS, extract_dish_id, merge_results_json  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
TOL = 1e-3

FILES = {
    "zipnerf_log": REPO / "sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx",
    "feature_splatting_log": REPO / "sweep_cluster_k/holdout_protocol_v2/joint_selection/feature_splatting.csv",
    "ingp_fi_log": REPO / "ingp_fi.csv",
    "ingp_oc_log": REPO / "ingp_oc.csv",
    "descriptors_57d": REPO / "sweep_cluster_k/k_6/feats_normalized.csv",
    "normalization_code": REPO / "src/feats_norm.py",
    "normalization_pipeline": REPO / "scripts/run_feature_analysis_pipeline.py",
    "k6_regime_labels": REPO / "sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv",
    "k6_run_metadata": REPO / "sweep_cluster_k/k_6/clustered_scenes_k6_run_metadata.json",
    "rass48_subset": REPO / "sweep_cluster_k/budget_sweep_k6_auto_v2/recommended_subset.csv",
    "rass96_subset": REPO / "results/strong_accept/manifests/bass_joint_k6_b16_best_subset.csv",
    "facility_location_summary": REPO / "sweep_cluster_k/baseline_comparison_lpips_ks/baseline_min_size_summary.csv",
    "facility_location_config": REPO / "sweep_cluster_k/baseline_comparison_lpips_ks/baseline_eval_config.json",
    "audit_code": REPO / "scripts/generate_strong_accept_results.py",
    "cross_method_reference": REPO / "results_final/paper_extra/common_intersection_method_means_and_bass_gaps.csv",
}

# Reference values from the paper.
REF_EXPORT = {
    "psnr_gap": 0.1532,
    "ssim_gap": 0.0059,
    "lpips_gap_max": 1e-4,
    "ks_psnr": 0.0904,
    "ks_ssim": 0.0743,
    "ks_lpips": 0.1126,
}
REF_SWEEP = {
    48: {"n_pass": 88, "lcb": 0.182},
    96: {"n_pass": 113, "lcb": 0.2406},
}
# abs mean gaps (subset vs 3473-scene intersection population), per method.
REF_CROSS_METHOD = {
    "instant_ngp_fi": {"psnr": 0.0840, "ssim": 0.0020},
    "instant_ngp_oc": {"psnr": 0.1092, "ssim": 0.0029},
    "feature_splatting": {"psnr": 0.0919, "ssim": 0.0074, "lpips": 0.0211},
    "zipnerf": {"psnr": 0.1411, "ssim": 0.0056, "lpips": 0.0007},
}


def table_info(path: Path):
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        return None
    return df


def main() -> None:
    # ------------------------------------------------------------------ 1
    print("== P0.1 INVENTORY ==")
    inventory = {}
    missing = []
    for label, path in FILES.items():
        entry = {"path": str(path.relative_to(REPO)), "exists": path.exists()}
        if not path.exists():
            missing.append(label)
            inventory[label] = entry
            print(f"  {label:28s} MISSING: {path}")
            continue
        if path.suffix in (".csv", ".xlsx"):
            df = table_info(path)
            entry["rows"] = int(len(df))
            entry["columns"] = [str(c) for c in df.columns]
            print(f"  {label:28s} rows={len(df):5d}  {path.relative_to(REPO)}")
        else:
            entry["rows"] = None
            print(f"  {label:28s} (code/json)   {path.relative_to(REPO)}")
        inventory[label] = entry

    inventory["facility_location_code"] = {
        "path": None,
        "exists": False,
        "note": (
            "Selection implementation not present in the repo; only outputs "
            "(baseline_min_size_summary.csv, baseline_sweep_results.csv) and "
            "config (baseline_eval_config.json: M=400, seed 0, "
            "facility_candidate_pool=16). Audit scripts consume it as "
            "'summary only'."
        ),
    }
    print("  facility_location_code       NOT IN REPO (outputs + config only)")

    inventory["candidate_seed_protocol"] = {
        "path": "scripts/generate_strong_accept_results.py::_simulate_trials",
        "rule": (
            "numpy default_rng(seed + b) for balanced mode, "
            "default_rng(seed + 10000 + b) for uniform mode; global seed 0, "
            "M=400; k-means regimes use random_seed 0 "
            "(clustered_scenes_k6_run_metadata.json)."
        ),
    }

    if missing:
        raise SystemExit(f"STOP: missing inventory items: {missing}")

    # Load core tables.
    mapping_df = _load_mapping_csv(FILES["k6_regime_labels"])
    zip_df = _load_zipnerf_metrics(FILES["zipnerf_log"])
    full_k6_df = _merge_mapping_and_metrics(mapping_df, zip_df)
    rass48_ids = _load_subset_ids(FILES["rass48_subset"])
    rass96_ids = _load_subset_ids(FILES["rass96_subset"])

    fs = _canonicalize_method_table(MethodSpec("feature-splatting", FILES["feature_splatting_log"])).df
    fi = _canonicalize_method_table(MethodSpec("instant-ngp-fi", FILES["ingp_fi_log"])).df
    oc = _canonicalize_method_table(MethodSpec("instant-ngp-oc", FILES["ingp_oc_log"])).df
    for tbl in (fs, fi, oc):
        tbl["dish_id"] = tbl["scene_id"].map(extract_dish_id)

    common_ids = (
        set(full_k6_df["dish_id"]) & set(fs["dish_id"]) & set(fi["dish_id"]) & set(oc["dish_id"])
    )

    counts = {
        "zipnerf_log_unique_scenes": int(len(zip_df)),
        "descriptor_scenes": int(len(mapping_df)),
        "zipnerf_audit_population": int(len(full_k6_df)),
        "common_intersection": int(len(common_ids)),
        "rass48_size": len(rass48_ids),
        "rass96_size": len(rass96_ids),
    }
    inventory["derived_counts"] = counts
    print(f"  derived: audit population {counts['zipnerf_audit_population']}, "
          f"intersection {counts['common_intersection']}, "
          f"RASS-48 {counts['rass48_size']}, RASS-96 {counts['rass96_size']}")

    if counts["common_intersection"] != 3473:
        raise SystemExit(f"STOP: intersection {counts['common_intersection']} != 3473")
    if counts["rass48_size"] != 48 or counts["rass96_size"] != 96:
        raise SystemExit("STOP: RASS subset sizes wrong")
    if counts["zipnerf_audit_population"] not in (3521, 3522):
        raise SystemExit(
            f"STOP: audit population {counts['zipnerf_audit_population']} not in (3521, 3522)"
        )
    population_note = (
        "Effective Zip-NeRF audit population is 3521, not 3522: descriptor "
        "scene dish_1563900172 is absent from the Zip-NeRF log. The paper's "
        "exported reference values were computed on this 3521-scene "
        "population (full_count=3521 in the paper artifacts), so this is a "
        "documented off-by-one in the stated constant, not a data change."
    )
    inventory["population_note"] = population_note

    checks = []

    def check(name, got, expected, tol, exact=False):
        ok = (got == expected) if exact else (abs(got - expected) <= tol)
        checks.append(ok)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}: got {got}, expected {expected}")
        return ok

    # ------------------------------------------------------------------ 2a
    print("\n== P0.2a RASS-48 EXPORT AUDIT ==")
    export_stats = _metric_eval_for_ids(full_k6_df, rass48_ids, THRESHOLDS)
    export_audit = {
        "psnr_gap": export_stats["psnr_gap"],
        "ssim_gap": export_stats["ssim_gap"],
        "lpips_gap": export_stats["lpips_gap"],
        "ks_psnr": export_stats["ks_psnr"],
        "ks_ssim": export_stats["ks_ssim"],
        "ks_lpips": export_stats["ks_lpips"],
        "reference": REF_EXPORT,
    }
    check("dPSNR", round(export_stats["psnr_gap"], 4), REF_EXPORT["psnr_gap"], TOL)
    check("dSSIM", round(export_stats["ssim_gap"], 4), REF_EXPORT["ssim_gap"], TOL)
    ok_lpips = abs(export_stats["lpips_gap"]) < REF_EXPORT["lpips_gap_max"]
    checks.append(ok_lpips)
    print(f"  [{'PASS' if ok_lpips else 'FAIL'}] |dLPIPS| < 1e-4: {export_stats['lpips_gap']:.2e}")
    check("KS PSNR", round(export_stats["ks_psnr"], 4), REF_EXPORT["ks_psnr"], TOL)
    check("KS SSIM", round(export_stats["ks_ssim"], 4), REF_EXPORT["ks_ssim"], TOL)
    check("KS LPIPS", round(export_stats["ks_lpips"], 4), REF_EXPORT["ks_lpips"], TOL)
    export_audit["pass"] = all(checks)

    # ------------------------------------------------------------------ 2b
    print("\n== P0.2b BALANCED-GENERATOR SWEEP (M=400, seed 0) ==")
    groups = _build_groups(full_k6_df)
    sweep_results = {}
    for scenes, b in ((48, 8), (96, 16)):
        trial_df, _bj, _bm = _simulate_trials(
            full_k6_df, groups, b, 400, 0, "balanced", THRESHOLDS
        )
        n_pass = int(trial_df["joint_pass"].sum())
        lcb = _wilson_lower_bound(n_pass / 400, 400, 0.95)
        ref = REF_SWEEP[scenes]
        ok_n = check(f"{scenes} scenes n_pass (deterministic)", n_pass, ref["n_pass"], 0, exact=True)
        ok_l = check(f"{scenes} scenes Wilson LCB", round(lcb, 4), ref["lcb"], TOL)
        sweep_results[f"sweep_{scenes}"] = {
            "n_pass": n_pass,
            "n_trials": 400,
            "empirical_pass_rate": n_pass / 400,
            "wilson_lcb_95": lcb,
            "reference": ref,
            "rng_seed_effective": 0 + b,
            "pass": ok_n and ok_l,
        }

    # ------------------------------------------------------------------ 2c
    print("\n== P0.2c RASS-48 CROSS-METHOD GAPS (3,473-scene intersection) ==")
    # Column convention: the paper's cross-method table
    # (results_final/paper_extra/common_intersection_method_means_and_bass_gaps.csv)
    # uses the per-frame-mean PSNR columns of the Instant-NGP logs ('PNSR' in
    # ingp_fi.csv, 'PSNR' in ingp_oc.csv). The strong-accept canonicalizer
    # instead prefers the MSE-aggregated 'psnr avgmse' columns. Both are
    # computed below; the paper convention is the one checked against the
    # references.
    fi_raw = pd.read_csv(FILES["ingp_fi_log"])
    oc_raw = pd.read_csv(FILES["ingp_oc_log"])
    for raw in (fi_raw, oc_raw):
        raw["dish_id"] = raw["dish_id"].map(extract_dish_id)
    fi_paper = pd.DataFrame(
        {"dish_id": fi_raw["dish_id"], "psnr": fi_raw["PNSR"], "ssim": fi_raw["SSIM"]}
    ).groupby("dish_id", as_index=False).mean()
    oc_paper = pd.DataFrame(
        {"dish_id": oc_raw["dish_id"], "psnr": oc_raw["PSNR"], "ssim": oc_raw["SSIM"]}
    ).groupby("dish_id", as_index=False).mean()

    rass48_set = set(rass48_ids)
    method_tables = {
        "zipnerf": (full_k6_df, ["psnr", "ssim", "lpips"], None),
        "feature_splatting": (fs, ["psnr", "ssim", "lpips"], None),
        "instant_ngp_fi": (fi_paper, ["psnr", "ssim"], fi),
        "instant_ngp_oc": (oc_paper, ["psnr", "ssim"], oc),
    }
    cross_method = {}
    for name, (tbl, metrics, alt_tbl) in method_tables.items():
        pop = tbl[tbl["dish_id"].isin(common_ids)]
        sub = pop[pop["dish_id"].isin(rass48_set)]
        if len(sub) != 48:
            raise SystemExit(f"STOP: {name}: RASS-48 overlap with intersection is {len(sub)} != 48")
        entry = {"population_n": int(len(pop)), "subset_n": int(len(sub))}
        for metric in metrics:
            gap = abs(float(sub[metric].mean() - pop[metric].mean()))
            entry[f"abs_{metric}_gap"] = gap
            check(f"{name} {metric}", round(gap, 4), REF_CROSS_METHOD[name][metric], TOL)
        if alt_tbl is not None:
            apop = alt_tbl[alt_tbl["dish_id"].isin(common_ids)]
            asub = apop[apop["dish_id"].isin(rass48_set)]
            entry["abs_psnr_gap_avgmse_convention"] = abs(
                float(asub["psnr"].mean() - apop["psnr"].mean())
            )
            entry["psnr_column_note"] = (
                "checked value uses the per-frame-mean PSNR column (paper "
                "convention); avgmse-convention value included for reference"
            )
        entry["reference"] = REF_CROSS_METHOD[name]
        cross_method[name] = entry
    cross_method["column_convention_note"] = (
        "Instant-NGP PSNR reproduces the paper references only with the "
        "per-frame-mean columns (PNSR / PSNR), not the 'psnr avgmse' columns "
        "preferred by _canonicalize_method_table. Gaps under the avgmse "
        "convention: FI 0.1095 (vs 0.0840), OC 0.0344 (vs 0.1092). Any task "
        "consuming Instant-NGP PSNR must state which column it uses; the "
        "multi_method_frontier task used the avgmse convention."
    )

    # ------------------------------------------------------------------ output
    all_pass = all(checks)
    print(f"\n== P0 RESULT: {'ALL CHECKS PASS' if all_pass else 'FAILURES PRESENT'} "
          f"({sum(checks)}/{len(checks)}) ==")

    p0 = {
        "description": (
            "Pipeline validation: inventory of all audit inputs plus "
            "reproduction of the paper's exported RASS-48 audit, the "
            "balanced-generator sweep at 48/96 scenes, and the RASS-48 "
            "cross-method gaps on the 3,473-scene intersection, using the "
            "repo's own audit code (generate_strong_accept_results.py)."
        ),
        "inventory": inventory,
        "reproduction": {
            "export_audit": export_audit,
            "sweep_48": sweep_results["sweep_48"],
            "sweep_96": sweep_results["sweep_96"],
            "cross_method": cross_method,
        },
        "matching_tolerance": {"deterministic": "exact", "otherwise": TOL},
        "n_checks": len(checks),
        "n_checks_passed": int(sum(checks)),
        "all_pass": bool(all_pass),
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"P0": p0})
    print(f"Wrote P0 key into {OUT_DIR / 'rebuttal_results.json'}")

    if not all_pass:
        raise SystemExit("P0 FAILED: do not run new events on top of this audit.")


if __name__ == "__main__":
    main()
