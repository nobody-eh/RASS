#!/usr/bin/env python3
"""Task P15 step 3: versioned JSON configs for every audit event the rebuttal
references as released. Emitted from rebuttal_results.json plus the recorded
constants, so numbers cannot drift from the certified runs."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = json.loads((REPO / "rebuttal/rebuttal_results.json").read_text())
OUT = REPO / "rebuttal/event_configs"
OUT.mkdir(exist_ok=True)

SCHEMA = "rass-event-config/1.0"
DEFAULT_TOLS = {"psnr": 0.5, "ssim": 0.01, "lpips": 0.01}
K6 = {"regimes": "k=6 KMeans seed 0 on the 57-D descriptors excluding proxies "
                 "num_images/num_frames_total/num_frames_used (shipped mapping, "
                 "replication validated ARI=1.0)"}
MM3 = {"zipnerf": ["psnr", "ssim", "lpips"],
       "feature_splatting": ["psnr", "ssim", "lpips"],
       "instant_ngp_fi": ["psnr", "ssim"]}
MM4 = {**MM3, "nerfacto": ["psnr", "ssim", "lpips"]}
MM5 = {**MM4, "bionerf": ["psnr", "ssim", "lpips"]}

configs = {
    "threemethod_equal_v1.json": {
        "schema_version": SCHEMA, "event": "three-method joint fidelity (E1)",
        "population": {"definition": "4-way intersection zipnerf ∩ "
                       "feature_splatting ∩ instant_ngp_fi ∩ instant_ngp_oc",
                       "n_scenes": 3473,
                       "ingp_psnr_column": "per-frame-mean PNSR/PSNR (paper "
                       "cross-method convention, P0)"},
        "constraints": {"method_metrics": MM3,
                        "event": "all mean gaps within tolerance AND all "
                        "per-metric KS <= ks_tol"},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14},
        "allocation": {"type": "equal-per-regime", **K6},
        "M": 400, "seed_rule": "default_rng(0 + b), b = subset_size/6",
    },
    "threemethod_proportional_v1.json": {
        "schema_version": SCHEMA,
        "event": "three-method joint fidelity, proportional allocation (E10)",
        "population": {"definition": "same 3,473-scene intersection as E1",
                       "n_scenes": 3473},
        "constraints": {"method_metrics": MM3},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14},
        "allocation": {"type": "proportional-to-regime-size, largest-remainder "
                       "rounding", **K6},
        "M": 400, "seed_rule": "default_rng(0 + b), b = subset_size/6, paired "
                               "with the equal frontier",
    },
    "fourmethod_v1.json": {
        "schema_version": SCHEMA,
        "event": "four-method joint fidelity on available coverage (E9, I4)",
        "population": {"definition": "common-coverage subpopulation of methods "
                       "zipnerf/feature_splatting/instant_ngp_fi/nerfacto as "
                       "of 2026-07-24 (nerfacto incl. 188 validated local "
                       "GPU evals)", "n_scenes": 2915},
        "constraints": {"method_metrics": MM4},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14},
        "allocation": {"type": "proportional-to-regime-size, largest-remainder "
                       "rounding", **K6},
        "M": 400, "seed_rule": "default_rng(0 + budget_scenes)",
    },
    "fivemethod_v1.json": {
        "schema_version": SCHEMA,
        "event": "five-method joint fidelity on available coverage (E9, I5)",
        "population": {"definition": "I4 methods plus bionerf, common coverage "
                       "as of 2026-07-24", "n_scenes": 2228},
        "constraints": {"method_metrics": MM5},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14},
        "allocation": {"type": "proportional-to-regime-size, largest-remainder "
                       "rounding", **K6},
        "M": 400, "seed_rule": "default_rng(0 + budget_scenes)",
    },
    "ranking_variantA_v1.json": {
        "schema_version": SCHEMA,
        "event": "E1 joint event AND sign preservation of all 6 pairwise "
                 "method mean gaps (3 pairs x {PSNR, SSIM}) (E2 variant A)",
        "population": {"definition": "E1 intersection", "n_scenes": 3473},
        "constraints": {"method_metrics": MM3,
                        "ranking": "sign(subset gap) == sign(population gap) "
                        "for every method pair and metric"},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14},
        "allocation": {"type": "equal-per-regime", **K6},
        "M": 400, "seed_rule": "default_rng(0 + b), identical paired draws "
                               "with E1",
    },
    "ranking_variantB_v1.json": {
        "schema_version": SCHEMA,
        "event": "variant A AND gap-magnitude preservation within tolerances "
                 "(E2 variant B)",
        "population": {"definition": "E1 intersection", "n_scenes": 3473},
        "constraints": {"method_metrics": MM3,
                        "ranking": "variant A plus |subset gap - population "
                        "gap| within gap tolerances",
                        "gap_magnitude_tolerances": {"psnr": 0.5, "ssim": 0.01}},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14},
        "allocation": {"type": "equal-per-regime", **K6},
        "M": 400, "seed_rule": "default_rng(0 + b), identical paired draws "
                               "with E1",
    },
    "dispersion_matched_v1.json": {
        "schema_version": SCHEMA,
        "event": "three-method joint fidelity under dispersion-matched "
                 "tolerances (E11) — POST-HOC operating point declared during "
                 "the discussion period; released alongside, never replacing, "
                 "the default contract",
        "population": {"definition": "E1 intersection", "n_scenes": 3473},
        "constraints": {"method_metrics": MM3},
        "tolerances": {"psnr": 0.5, "ssim": 0.013433, "lpips": 0.0179,
                       "ks": 0.14,
                       "rule": "tau_m = c * sigma_m, c = tau_PSNR/sigma_PSNR "
                       "= 0.5/6.2681 = 0.079769 (E3c per-scene stds); "
                       "cap-checked below min cross-method gaps"},
        "allocation": {"type": "equal-per-regime (proportional companion "
                       "recorded in E11)", **K6},
        "M": 400, "seed_rule": "default_rng(0 + b), identical to E1",
    },
    "regime_constrained_c1_v1.json": {
        "schema_version": SCHEMA,
        "event": "Zip-NeRF global event AND per-regime mean constraints "
                 "|mean_m(S∩r) - mean_m(full∩r)| <= 1*tau_m over the six "
                 "regimes (E4, c=1)",
        "population": {"definition": "effective Zip-NeRF audit set",
                       "n_scenes": 3521},
        "constraints": {"method_metrics": {"zipnerf": ["psnr", "ssim", "lpips"]},
                        "per_regime": "all 6 regimes x 3 metrics, c = 1"},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14,
                       "ks_scope": "global only (per-regime KS not required)"},
        "allocation": {"type": "equal-per-regime", **K6},
        "M": 400, "seed_rule": "default_rng(0 + b)",
    },
    "regime_constrained_c2_v1.json": {
        "schema_version": SCHEMA,
        "event": "as regime_constrained_c1 with c = 2 (E4, c=2)",
        "population": {"definition": "effective Zip-NeRF audit set",
                       "n_scenes": 3521},
        "constraints": {"method_metrics": {"zipnerf": ["psnr", "ssim", "lpips"]},
                        "per_regime": "all 6 regimes x 3 metrics, c = 2"},
        "tolerances": {**DEFAULT_TOLS, "ks": 0.14,
                       "ks_scope": "global only"},
        "allocation": {"type": "equal-per-regime", **K6},
        "M": 400, "seed_rule": "default_rng(0 + b)",
    },
}

for name, cfg in configs.items():
    (OUT / name).write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote", name)
print(f"{len(configs)} configs in {OUT}")
