#!/usr/bin/env python3
"""Key E9: deadline-ready multi-method comparison on AVAILABLE data.

Population: the common-coverage intersections as of 2026-07-24 —
I4 = E1 intersection ∩ nerfacto (2,950-scene log set incl. 188 local
evals), I5 = I4 ∩ bionerf. All references computed on the corresponding
intersection; results labeled "common-coverage subpopulation".

Sections:
a. Five-method table on I5: per-method mean/std for PSNR/SSIM/LPIPS
   (Instant-NGP: PSNR/SSIM only), all pairwise mean gaps, orderings.
b. PROPORTIONAL-allocation joint audit frontier on I4 and I5 (M=400,
   rng=default_rng(seed+b) with b = budget index over the standard sizes,
   paper tolerances). Proportional allocation makes the sampler unbiased on
   regime-skewed coverage (subset expectation = population mean), fixing
   the structural failure documented in E8.diagnosis; equal-allocation E8
   numbers are reported alongside as the contrast.
c. RASS-48 / RASS-96 mean-gap tables vs I4/I5 references (with in-
   intersection truncation counts).
d. Per-scene Kendall rank correlations between all method pairs on I5.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _build_groups,
    _load_subset_ids,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import NUM_TRIALS, SEED, THRESHOLDS, merge_results_json  # noqa: E402
from rebuttal_e1_joint_event import RASS48, RASS96, load_populations  # noqa: E402
from rebuttal_e8_extended_methods import SCRATCH, load_json_dir, to_df  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
SIZES = [24, 36, 48, 60, 72, 96, 120]
P_MIN = 0.08
MM_FULL = ["psnr", "ssim", "lpips"]


def proportional_counts(sizes: dict, total: int) -> dict:
    n = sum(sizes.values())
    raw = {c: total * v / n for c, v in sizes.items()}
    base = {c: int(np.floor(r)) for c, r in raw.items()}
    rem = total - sum(base.values())
    order = sorted(raw, key=lambda c: raw[c] - base[c], reverse=True)
    for c in order[:rem]:
        base[c] += 1
    return base


def eval_joint(pops: dict, mm: dict, subset_ids: set):
    rows, joint = [], True
    for name, metrics in mm.items():
        pop = pops[name]
        sub = pop[pop["dish_id"].isin(subset_ids)]
        for m in metrics:
            fv = pop[m].to_numpy(float)
            sv = sub[m].to_numpy(float)
            gap = abs(float(sv.mean() - fv.mean()))
            ks = float(ks_2samp(fv, sv).statistic)
            ok = gap <= THRESHOLDS[f"{m}_tol"] and ks <= THRESHOLDS["ks_tol"]
            rows.append({"method": name, "metric": m, "abs_mean_gap": gap, "ks": ks,
                         "mean_pass": gap <= THRESHOLDS[f"{m}_tol"],
                         "ks_pass": ks <= THRESHOLDS["ks_tol"]})
            joint = joint and ok
    return rows, joint


def main() -> None:
    # ---------------- populations ----------------
    pop_k6, pops_e1, common = load_populations()
    nerfacto = to_df(load_json_dir(str(SCRATCH / "output_json_nerfacto/json/*.json"),
                                   "psnr", "ssim", "lpips"))
    bio = {}
    for d in ("output_json_bionerf/DISH", "output_json5/DISH", "output_json6/DISH"):
        for k, v in load_json_dir(str(SCRATCH / d / "*.json"),
                                  "fine_psnr", "fine_ssim", "fine_lpips").items():
            bio.setdefault(k, v)
    bionerf = to_df(bio)
    i4 = common & set(nerfacto["dish_id"])
    i5 = i4 & set(bionerf["dish_id"])
    print(f"I4={len(i4)}, I5={len(i5)} (nerfacto logs {len(nerfacto)}, bionerf {len(bionerf)})")

    def restrict(isect):
        mm = {"zipnerf": MM_FULL, "feature_splatting": MM_FULL,
              "instant_ngp_fi": ["psnr", "ssim"], "nerfacto": MM_FULL}
        pp = {k: v[v["dish_id"].isin(isect)].reset_index(drop=True) for k, v in pops_e1.items()}
        pp["nerfacto"] = nerfacto[nerfacto["dish_id"].isin(isect)].reset_index(drop=True)
        if isect is i5:
            pp["bionerf"] = bionerf[bionerf["dish_id"].isin(isect)].reset_index(drop=True)
            mm["bionerf"] = MM_FULL
        pk = pop_k6[pop_k6["dish_id"].isin(isect)].reset_index(drop=True)
        return pk, pp, mm

    # ---------------- a. five-method table on I5 ----------------
    pk5, pp5, mm5 = restrict(i5)
    table_a = {}
    for name, metrics in mm5.items():
        table_a[name] = {}
        for m in metrics:
            v = pp5[name][m].to_numpy(float)
            table_a[name][m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    pair_gaps, orderings = {}, {}
    for m in ("psnr", "ssim", "lpips"):
        avail = [n for n in mm5 if m in mm5[n]]
        for a, b in itertools.combinations(avail, 2):
            pair_gaps[f"{a}_minus_{b}_{m}"] = table_a[a][m]["mean"] - table_a[b][m]["mean"]
        orderings[m] = sorted(avail, key=lambda n: table_a[n][m]["mean"],
                              reverse=(m != "lpips"))
    print("orderings (best->worst):", orderings)

    # ---------------- b. proportional-allocation frontiers ----------------
    frontiers = {}
    for tag, isect in (("I4_4methods", i4), ("I5_5methods", i5)):
        pk, pp, mm = restrict(isect)
        groups = _build_groups(pk)
        sizes = {c: len(g) for c, g in groups.items()}
        ids_pop = pk["dish_id"].astype(str).to_numpy()
        rows = []
        for bi, total in enumerate(SIZES):
            alloc = proportional_counts(sizes, total)
            if min(alloc.values()) < 1 or any(alloc[c] > sizes[c] for c in alloc):
                continue
            rng = np.random.default_rng(SEED + total)  # seed rule: seed + subset size
            n_pass = 0
            for _ in range(NUM_TRIALS):
                picks = [rng.choice(groups[c], size=alloc[c], replace=False)
                         for c in sorted(groups)]
                idx = np.sort(np.concatenate(picks))
                _, joint = eval_joint(pp, mm, set(ids_pop[idx].tolist()))
                n_pass += int(joint)
            p = n_pass / NUM_TRIALS
            lcb = _wilson_lower_bound(p, NUM_TRIALS, 0.95)
            rows.append({"budget_scenes": total, "allocation": {str(k): v for k, v in alloc.items()},
                         "n_pass": n_pass, "empirical_pass_rate": p, "wilson_lcb_95": lcb,
                         "rng_seed_effective": SEED + total})
            print(f"[{tag}] {total:3d} scenes: {n_pass:3d}/400 (LCB {lcb:.4f})")
        hit = next((r for r in rows if r["wilson_lcb_95"] >= P_MIN), None)
        frontiers[tag] = {"population": len(pk), "regime_sizes": {str(k): v for k, v in sizes.items()},
                          "frontier": rows,
                          "budget_p008": hit["budget_scenes"] if hit else "not reached up to 120"}

    # ---------------- c. RASS tables ----------------
    rass = {}
    for label, path in (("rass48", RASS48), ("rass96", RASS96)):
        ids = set(_load_subset_ids(path))
        rass[label] = {}
        for tag, isect in (("I4", i4), ("I5", i5)):
            pk, pp, mm = restrict(isect)
            inside = ids & isect
            rows, joint = eval_joint(pp, mm, inside)
            rass[label][tag] = {"nominal": len(ids), "in_intersection": len(inside),
                                "pass": bool(joint),
                                "violated": [f"{r['method']}.{r['metric']}" for r in rows
                                             if not (r["mean_pass"] and r["ks_pass"])],
                                "constraints": rows}
            print(f"{label} on {tag}: {len(inside)}/{len(ids)} scenes, "
                  f"{'PASS' if joint else 'FAIL'} viol={rass[label][tag]['violated']}")

    # ---------------- d. Kendall tau on I5 ----------------
    merged = None
    for name in mm5:
        df = pp5[name][["dish_id"] + mm5[name]].rename(
            columns={m: f"{name}_{m}" for m in mm5[name]})
        merged = df if merged is None else merged.merge(df, on="dish_id")
    kendall = {}
    for m in ("psnr", "ssim", "lpips"):
        avail = [n for n in mm5 if m in mm5[n]]
        for a, b in itertools.combinations(avail, 2):
            tau = kendalltau(merged[f"{a}_{m}"], merged[f"{b}_{m}"]).statistic
            kendall[f"{a}_vs_{b}_{m}"] = float(tau)

    e9 = {
        "description": (
            "Deadline-ready multi-method comparison on AVAILABLE data "
            "(common-coverage subpopulations, 2026-07-24 log state: "
            "nerfacto 2,950 scenes incl. 188 local GPU evals, bionerf "
            "2,620). Proportional-allocation audit makes the balanced "
            "sampler unbiased on regime-skewed coverage; equal-allocation "
            "results (0/400, E8) stand as the contrast demonstrating the "
            "audit detects population shift."
        ),
        "populations": {"I4": len(i4), "I5": len(i5)},
        "caveat": (
            "I4/I5 are not random subsamples of the audit set (missingness "
            "concentrated in regimes 0/1/2/4); all claims are for the "
            "common-coverage subpopulation. dish_1551323219 evaluated at "
            "28k/30k steps (flagged)."
        ),
        "a_method_table_I5": table_a,
        "a_pairwise_gaps_I5": pair_gaps,
        "a_orderings_I5": orderings,
        "b_proportional_frontiers": frontiers,
        "b_seed_rule": "numpy default_rng(seed + budget_scenes), proportional allocation (largest-remainder rounding)",
        "b_equal_allocation_contrast": "E8 frontiers: 0/400 at every budget (see E8.diagnosis)",
        "c_rass_tables": {k: {t: {kk: vv for kk, vv in v[t].items() if kk != "constraints"}
                              for t in v} for k, v in rass.items()},
        "c_rass_constraints_full": rass,
        "d_kendall_tau_I5": kendall,
        "thresholds": THRESHOLDS,
        "num_trials": NUM_TRIALS,
        "seed": SEED,
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E9": e9})
    print("Wrote E9 key")


if __name__ == "__main__":
    main()
