#!/usr/bin/env python3
"""Task P11 / key E11: dispersion-matched operating point, declared post hoc
during the discussion period, released ALONGSIDE the default contract (never
replacing it).

Derivation: anchor fraction c = tau_PSNR/sigma_PSNR = 0.5/6.2681 = 0.079769
(E3c stds on the 3,521-scene population). tau_SSIM' = c*0.1684 = 0.013434,
tau_LPIPS' = c*0.2244 = 0.017900; tau_PSNR and KS unchanged (0.5 dB, 0.14).
Calibration cap verified: each new tolerance below the smallest cross-method
mean gap for its metric.

Runs (population/references/seeds/M as E1): three-method joint event under
the new tolerances with equal and proportional allocation; RASS-48/96
constraint tables; Zip-NeRF-only frontier on the 3,521 population; binding
histogram at 48 scenes (equal allocation).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _build_groups,
    _load_mapping_csv,
    _load_subset_ids,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import NUM_TRIALS, SEED, merge_results_json  # noqa: E402
from rebuttal_e1_joint_event import METHOD_METRICS, RASS48, RASS96, load_populations  # noqa: E402
from rebuttal_e3_sensitivity import SHIPPED_MAPPINGS, ZIPNERF_LOG, Sweeper, fast_ks  # noqa: E402
from rebuttal_e9_available_data_comparison import proportional_counts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
BS = [4, 6, 8, 10, 12, 14, 16, 20]

SIGMA = {"psnr": 6.2681, "ssim": 0.1684, "lpips": 0.2244}
C = 0.5 / SIGMA["psnr"]
NEW_TOLS = {"psnr_tol": 0.5, "ssim_tol": round(C * SIGMA["ssim"], 6),
            "lpips_tol": round(C * SIGMA["lpips"], 6), "ks_tol": 0.14}
MIN_GAPS = {"ssim": 0.0199, "lpips": 0.0952}  # E3c smallest cross-method gaps
LABEL = "dispersion-matched operating point, declared post hoc during the discussion period"


def main() -> None:
    print(f"derived tolerances: {NEW_TOLS} (c = {C:.6f})")
    cap_checks = {
        "ssim": NEW_TOLS["ssim_tol"] < MIN_GAPS["ssim"],
        "lpips": NEW_TOLS["lpips_tol"] < MIN_GAPS["lpips"],
    }
    if not all(cap_checks.values()):
        raise SystemExit(f"STOP: calibration cap violated: {cap_checks}")
    print(f"cap checks pass: ssim {NEW_TOLS['ssim_tol']} < {MIN_GAPS['ssim']}, "
          f"lpips {NEW_TOLS['lpips_tol']} < {MIN_GAPS['lpips']}")

    pop_k6, pops, common = load_populations()
    groups = _build_groups(pop_k6)
    sizes = {int(c): len(g) for c, g in groups.items()}
    ids_pop = pop_k6["dish_id"].astype(str).to_numpy()

    vals, pop_sorted, pop_mean = {}, {}, {}
    for name, metrics in METHOD_METRICS.items():
        df = pops[name].set_index("dish_id").loc[pop_k6["dish_id"]]
        for m in metrics:
            v = df[m].to_numpy(float)
            vals[(name, m)] = v
            pop_sorted[(name, m)] = np.sort(v)
            pop_mean[(name, m)] = float(v.mean())

    def trial_eval(idx, collect=None):
        joint = True
        for (name, m), v in vals.items():
            sv = v[idx]
            gap = abs(float(sv.mean()) - pop_mean[(name, m)])
            ks = fast_ks(np.sort(sv), pop_sorted[(name, m)])
            mean_ok = gap <= NEW_TOLS[f"{m}_tol"]
            ks_ok = ks <= NEW_TOLS["ks_tol"]
            if collect is not None:
                if not mean_ok:
                    collect[f"{name}.{m}.mean"] = collect.get(f"{name}.{m}.mean", 0) + 1
                if not ks_ok:
                    collect[f"{name}.{m}.ks"] = collect.get(f"{name}.{m}.ks", 0) + 1
            joint = joint and mean_ok and ks_ok
        return joint

    def run_frontier(alloc_fn, collect_at_b=None):
        rows, hist, nfail = [], {}, 0
        for b in BS:
            total = 6 * b
            alloc = alloc_fn(total)
            rng = np.random.default_rng(SEED + b)
            n_pass = 0
            for _ in range(NUM_TRIALS):
                picks = [rng.choice(groups[c], size=alloc[c], replace=False)
                         for c in sorted(groups)]
                idx = np.sort(np.concatenate(picks))
                if collect_at_b == b:
                    coll = {}
                    ok = trial_eval(idx, collect=coll)
                    if not ok:
                        nfail += 1
                        for k, v in coll.items():
                            hist[k] = hist.get(k, 0) + v
                else:
                    ok = trial_eval(idx)
                n_pass += int(ok)
            p = n_pass / NUM_TRIALS
            rows.append({"budget_scenes": total, "b": b, "n_pass": n_pass,
                         "empirical_pass_rate": p,
                         "wilson_lcb_95": _wilson_lower_bound(p, NUM_TRIALS, 0.95)})
        return rows, dict(sorted(hist.items(), key=lambda kv: -kv[1])), nfail

    def target(rows, pmin):
        hit = next((r for r in rows if r["wilson_lcb_95"] >= pmin), None)
        return hit["budget_scenes"] if hit else "not reached up to 120"

    print("== a. equal allocation under dispersion-matched tolerances ==")
    eq_rows, hist48, nfail48 = run_frontier(
        lambda total: {c: total // 6 for c in groups}, collect_at_b=8)
    for r in eq_rows:
        print(f"  {r['budget_scenes']:3d}sc: {r['n_pass']:3d}/400 (LCB {r['wilson_lcb_95']:.4f})")
    print("== b. proportional allocation ==")
    pr_rows, _, _ = run_frontier(lambda total: proportional_counts(sizes, total))
    for r in pr_rows:
        print(f"  {r['budget_scenes']:3d}sc: {r['n_pass']:3d}/400 (LCB {r['wilson_lcb_95']:.4f})")

    budget_targets = {
        "equal": {"p008": target(eq_rows, 0.08), "p020": target(eq_rows, 0.20)},
        "proportional": {"p008": target(pr_rows, 0.08), "p020": target(pr_rows, 0.20)},
    }
    print("targets:", budget_targets)

    # ---- c. RASS tables (allocation-independent) ----
    def rass_table(path):
        ids = set(_load_subset_ids(path)) & common
        rows, joint = [], True
        for (name, m), v in vals.items():
            pop = pops[name]
            sub = pop[pop["dish_id"].isin(ids)]
            gap = abs(float(sub[m].mean() - pop[m].mean()))
            import scipy.stats as st
            ks = float(st.ks_2samp(pop[m].to_numpy(float), sub[m].to_numpy(float)).statistic)
            mean_ok, ks_ok = gap <= NEW_TOLS[f"{m}_tol"], ks <= NEW_TOLS["ks_tol"]
            rows.append({"method": name, "metric": m, "abs_mean_gap": gap,
                         "tolerance": NEW_TOLS[f"{m}_tol"], "mean_pass": mean_ok,
                         "ks": ks, "ks_pass": ks_ok})
            joint = joint and mean_ok and ks_ok
        return {"scenes_in_intersection": len(ids), "pass": bool(joint), "constraints": rows,
                "violated": [f"{r['method']}.{r['metric']}" for r in rows
                             if not (r["mean_pass"] and r["ks_pass"])]}

    rass48 = rass_table(RASS48)
    rass96 = rass_table(RASS96)
    print(f"RASS-48: {'PASS' if rass48['pass'] else 'FAIL'} viol={rass48['violated']}")
    print(f"RASS-96: {'PASS' if rass96['pass'] else 'FAIL'} viol={rass96['violated']}")

    # ---- d. Zip-NeRF-only on 3,521 under new tolerances ----
    full = _merge_mapping_and_metrics(_load_mapping_csv(SHIPPED_MAPPINGS[6]),
                                      _load_zipnerf_metrics(ZIPNERF_LOG))
    sw = Sweeper(full)
    tols_sw = {"psnr": NEW_TOLS["psnr_tol"], "ssim": NEW_TOLS["ssim_tol"],
               "lpips": NEW_TOLS["lpips_tol"], "ks": NEW_TOLS["ks_tol"]}
    zip_rows = []
    for b in BS:
        n_pass, lcb, _ = sw.run_budget(b, SEED, tols=tols_sw)
        zip_rows.append({"budget_scenes": 6 * b, "n_pass": n_pass,
                         "empirical_pass_rate": n_pass / NUM_TRIALS, "wilson_lcb_95": lcb})
        print(f"  zip-only {6*b:3d}sc: {n_pass:3d}/400 (LCB {lcb:.4f})")
    zip_p008 = target(zip_rows, 0.08)

    eq48 = next(r for r in eq_rows if r["budget_scenes"] == 48)
    one_sentence = (
        f"Under the dispersion-matched operating point (tau_SSIM 0.0134, tau_LPIPS 0.0179, "
        f"same 0.5 dB and KS 0.14) the three-method joint event gives {eq48['n_pass']}/400 at 48 "
        f"scenes and reaches the 0.08 LCB target at "
        f"{budget_targets['equal']['p008']} scenes (equal allocation)."
    )
    print("one_sentence_result:", one_sentence, f"({len(one_sentence)} chars)")

    e11 = {
        "label": LABEL,
        "derivation": {
            "anchor_fraction_c": C, "sigmas": SIGMA,
            "tau_ssim_new": NEW_TOLS["ssim_tol"], "tau_lpips_new": NEW_TOLS["lpips_tol"],
            "tau_psnr_unchanged": 0.5, "ks_unchanged": 0.14,
            "rule": "every mean tolerance at the same fraction c = tau_PSNR/sigma_PSNR of its per-scene std (E3c)",
        },
        "cap_checks": {"ssim": f"{NEW_TOLS['ssim_tol']} < {MIN_GAPS['ssim']}",
                       "lpips": f"{NEW_TOLS['lpips_tol']} < {MIN_GAPS['lpips']}",
                       "all_pass": True},
        "frontier_equal": eq_rows,
        "frontier_proportional": pr_rows,
        "budget_targets": budget_targets,
        "rass48_table": rass48,
        "rass96_table": rass96,
        "zip_only_frontier": zip_rows,
        "zip_only_budget_p008": zip_p008,
        "binding_histogram_48": hist48,
        "binding_n_failures_48": nfail48,
        "one_sentence_result": one_sentence,
        "num_trials": NUM_TRIALS, "seed": SEED,
        "seed_rule": "default_rng(seed + b), b = subset_size/6, as E1",
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E11": e11})
    print("Wrote E11 key")


if __name__ == "__main__":
    main()
