#!/usr/bin/env python3
"""Task P10 / key E10: E1's three-method joint event rerun with
proportional-to-regime-size allocation; everything else identical, so the
allocation comparison is reported symmetrically.

Identical to E1: 3,473-scene intersection, same references, 16 constraints,
paper tolerances, KS<=0.14, INGP per-frame-mean PNSR, M=400,
rng = default_rng(seed + b) with b = subset_size/6 (paired with E1 as far
as the allocation change permits). Only the per-regime allocation differs:
proportional with largest-remainder rounding, min 1 scene per regime.

Validation: the equal-allocation evaluator (fast KS) must reproduce E1's
frontier exactly before the proportional run is trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import _build_groups, _wilson_lower_bound  # noqa: E402
from rebuttal_audit_tasks import NUM_TRIALS, SEED, THRESHOLDS, merge_results_json  # noqa: E402
from rebuttal_e1_joint_event import METHOD_METRICS, load_populations  # noqa: E402
from rebuttal_e3_sensitivity import fast_ks  # noqa: E402
from rebuttal_e9_available_data_comparison import proportional_counts  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
BS = [4, 6, 8, 10, 12, 14, 16, 20]


def main() -> None:
    pop_k6, pops, _ = load_populations()
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

    def trial_eval(idx, tols=THRESHOLDS, collect=None):
        joint = True
        for (name, m), v in vals.items():
            sv = v[idx]
            gap = abs(float(sv.mean()) - pop_mean[(name, m)])
            ks = fast_ks(np.sort(sv), pop_sorted[(name, m)])
            mean_ok = gap <= tols[f"{m}_tol"]
            ks_ok = ks <= tols["ks_tol"]
            if collect is not None:
                if not mean_ok:
                    collect[f"{name}.{m}.mean"] = collect.get(f"{name}.{m}.mean", 0) + 1
                if not ks_ok:
                    collect[f"{name}.{m}.ks"] = collect.get(f"{name}.{m}.ks", 0) + 1
            joint = joint and mean_ok and ks_ok
        return joint

    def run_frontier(allocation_fn, collect_at_b=None):
        rows, allocs, hist = [], {}, {}
        n_fail_48 = 0
        for b in BS:
            total = 6 * b
            alloc = allocation_fn(total)
            allocs[str(total)] = {str(k): v for k, v in alloc.items()}
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
                        n_fail_48 += 1
                        for k, v in coll.items():
                            hist[k] = hist.get(k, 0) + v
                else:
                    ok = trial_eval(idx)
                n_pass += int(ok)
            p = n_pass / NUM_TRIALS
            rows.append({"budget_scenes": total, "b": b, "n_pass": n_pass,
                         "empirical_pass_rate": p,
                         "wilson_lcb_95": _wilson_lower_bound(p, NUM_TRIALS, 0.95),
                         "rng_seed_effective": SEED + b})
        return rows, allocs, dict(sorted(hist.items(), key=lambda kv: -kv[1])), n_fail_48

    # Validation: equal allocation must reproduce E1 exactly.
    e1 = json.loads((OUT_DIR / "rebuttal_results.json").read_text())["E1"]["frontier"]
    eq_rows, _, _, _ = run_frontier(lambda total: {c: total // 6 for c in groups})
    for got, ref in zip(eq_rows, e1):
        if got["n_pass"] != ref["n_pass"]:
            raise SystemExit(f"VALIDATION FAILED at {got['budget_scenes']}sc: "
                             f"{got['n_pass']} != E1 {ref['n_pass']}")
    print("[validation] equal-allocation frontier reproduces E1 exactly "
          f"({[r['n_pass'] for r in eq_rows]})")

    prop_rows, allocs, hist48, n_fail48 = run_frontier(
        lambda total: proportional_counts(sizes, total), collect_at_b=8)
    for r in prop_rows:
        print(f"  {r['budget_scenes']:3d} scenes: {r['n_pass']:3d}/400 "
              f"(LCB {r['wilson_lcb_95']:.4f})")

    def target(rows, pmin):
        hit = next((r for r in rows if r["wilson_lcb_95"] >= pmin), None)
        return hit["budget_scenes"] if hit else "not reached up to 120"

    b008, b020 = target(prop_rows, 0.08), target(prop_rows, 0.20)
    e008, _ = target(eq_rows, 0.08), target(eq_rows, 0.20)
    print(f"proportional: LCB>=0.08 at {b008}; >=0.20 at {b020} "
          f"(equal: >=0.08 at {e008})")
    print(f"binding at 48sc over {n_fail48} failures:",
          dict(list(hist48.items())[:6]))

    same = all(abs(p["n_pass"] - q["n_pass"]) <= 12 for p, q in zip(prop_rows, eq_rows))
    if b008 == "not reached up to 120" and e008 == "not reached up to 120":
        verdict = ("Proportional allocation does not change the three-method conclusion: "
                   "the 0.08 LCB target stays unreached up to 120 scenes under both allocations.")
    elif b008 != "not reached up to 120" and e008 == "not reached up to 120":
        verdict = (f"Proportional allocation certifies at {b008} scenes where equal "
                   "allocation does not - reported for symmetry, not preference.")
    else:
        verdict = (f"Both allocations reach the 0.08 target (proportional {b008}, "
                   f"equal {e008}); the three-method conclusion is allocation-robust.")
    print("verdict:", verdict, f"({len(verdict)} chars)")

    e10 = {
        "description": (
            "E1's three-method joint event under proportional-to-regime-size "
            "allocation (largest-remainder, min 1/regime), everything else "
            "identical, for symmetric allocation reporting. Equal-allocation "
            "evaluator validated to reproduce E1's frontier exactly."
        ),
        "frontier_proportional": prop_rows,
        "frontier_equal_ref": eq_rows,
        "allocations_per_budget": allocs,
        "budget_p008": b008,
        "budget_p020": b020,
        "budget_p008_equal_ref": e008,
        "binding_histogram_48": hist48,
        "binding_n_failures_48": n_fail48,
        "verdict_sentence": verdict,
        "seed_rule": "default_rng(seed + b), b = subset_size/6, paired with E1",
        "num_trials": NUM_TRIALS, "seed": SEED, "population_size": 3473,
        "note_rass": "RASS-48/96 not rerun: allocation-independent; E1 formal.",
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E10": e10})
    print("Wrote E10 key")


if __name__ == "__main__":
    main()
