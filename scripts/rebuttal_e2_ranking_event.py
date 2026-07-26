#!/usr/bin/env python3
"""Task P2 / key E2: ranking preservation added to the formal joint event.

Prerequisite: E1 (rebuttal_e1_joint_event.py) completed. This script reuses
E1's population, references, generator, and candidate draws (identical
rng = default_rng(seed + b) streams), and joins E1's per-trial joint-pass
results so the comparison is exactly paired.

Variant A: E_joint AND, for every pair among {Zip-NeRF, Feature-Splatting,
Instant-NGP full image} and each of PSNR and SSIM, the sign of the subset
mean gap equals the sign of the full-intersection gap (6 ordering
constraints).

Variant B: Variant A AND |subset gap - full gap| <= 0.5 dB (PSNR) / 0.01
(SSIM) (gap-magnitude preservation).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import _build_groups, _wilson_lower_bound  # noqa: E402
from rebuttal_audit_tasks import BUDGETS, NUM_TRIALS, SEED, merge_results_json  # noqa: E402
from rebuttal_e1_joint_event import load_populations  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"

PAIR_METHODS = ["zipnerf", "feature_splatting", "instant_ngp_fi"]
PAIRS = list(itertools.combinations(PAIR_METHODS, 2))
ORDER_METRICS = ["psnr", "ssim"]
GAP_TOL = {"psnr": 0.5, "ssim": 0.01}


def main() -> None:
    pop_k6, pops, _common = load_populations()
    groups = _build_groups(pop_k6)
    dish_ids_pop = pop_k6["dish_id"].astype(str).to_numpy()

    # E1 per-trial joint-pass results (paired: identical draws).
    e1_trials = pd.read_csv(OUT_DIR / "e1_joint_event_trials.csv")
    e1_joint = {
        (int(r.b), int(r.trial)): bool(r.joint_pass)
        for r in e1_trials[["b", "trial", "joint_pass"]].itertuples()
    }
    e1_frontier = pd.read_csv(OUT_DIR / "e1_joint_frontier.csv")
    e1_pass_by_b = dict(zip(e1_frontier["b"].astype(int), e1_frontier["n_pass"].astype(int)))

    # Metric value lookups per method, aligned to pop_k6 row order, and
    # full-intersection reference gaps.
    method_vals = {}
    for name in PAIR_METHODS:
        df = pops[name].set_index("dish_id")
        for metric in ORDER_METRICS:
            method_vals[(name, metric)] = (
                df[metric].reindex(pop_k6["dish_id"]).to_numpy(dtype=float)
            )
    full_gaps = {}
    for a, b_ in PAIRS:
        for metric in ORDER_METRICS:
            full_gaps[(a, b_, metric)] = float(
                np.mean(method_vals[(a, metric)]) - np.mean(method_vals[(b_, metric)])
            )
    print("Full-intersection reference gaps (A - B):")
    for (a, b_, metric), g in full_gaps.items():
        print(f"  {a} - {b_} [{metric}]: {g:+.4f}")

    rows_a, rows_b = [], []
    delta_vs_e1 = []
    cond48 = None
    n_order_uncond_48 = None

    for b in BUDGETS:
        rng = np.random.default_rng(SEED + b)
        n_a = n_b = 0
        n_order_only = 0
        n_joint_and_order = 0
        n_joint = 0
        for trial in range(NUM_TRIALS):
            picks = [
                rng.choice(groups[c], size=b, replace=False) for c in sorted(groups)
            ]
            idx = np.sort(np.concatenate(picks))

            sub_means = {
                key: float(np.mean(vals[idx])) for key, vals in method_vals.items()
            }
            order_ok = True
            mag_ok = True
            for a, b_ in PAIRS:
                for metric in ORDER_METRICS:
                    sub_gap = sub_means[(a, metric)] - sub_means[(b_, metric)]
                    ref_gap = full_gaps[(a, b_, metric)]
                    if np.sign(sub_gap) != np.sign(ref_gap):
                        order_ok = False
                    if abs(sub_gap - ref_gap) > GAP_TOL[metric]:
                        mag_ok = False

            joint = e1_joint[(b, trial)]
            n_joint += int(joint)
            n_order_only += int(order_ok)
            n_joint_and_order += int(joint and order_ok)
            n_a += int(joint and order_ok)
            n_b += int(joint and order_ok and mag_ok)

        if n_joint != e1_pass_by_b[b]:
            raise SystemExit(
                f"PAIRING BROKEN at b={b}: joined joint passes {n_joint} != E1 {e1_pass_by_b[b]}"
            )

        for rows, n in ((rows_a, n_a), (rows_b, n_b)):
            p = n / NUM_TRIALS
            rows.append(
                {
                    "budget_scenes": 6 * b,
                    "b": b,
                    "n_trials": NUM_TRIALS,
                    "n_pass": n,
                    "empirical_pass_rate": p,
                    "wilson_lcb_95": _wilson_lower_bound(p, NUM_TRIALS, 0.95),
                    "rng_seed_effective": SEED + b,
                }
            )
        delta_vs_e1.append(
            {
                "budget_scenes": 6 * b,
                "e1_n_pass": e1_pass_by_b[b],
                "variantA_n_pass": n_a,
                "variantA_delta": n_a - e1_pass_by_b[b],
                "variantB_n_pass": n_b,
                "variantB_delta": n_b - e1_pass_by_b[b],
                "n_order_preserving_unconditional": n_order_only,
            }
        )
        if b == 8:
            cond48 = (n_joint_and_order / n_joint) if n_joint else None
            n_order_uncond_48 = n_order_only
        print(
            f"b={b:2d} ({6*b:3d} scenes): E1 {e1_pass_by_b[b]:3d} | "
            f"A {n_a:3d} | B {n_b:3d} | order-only {n_order_only:3d}/400"
        )

    def targets(rows):
        out = {}
        for label, pmin in (("budget_p008", 0.08), ("budget_p020", 0.20)):
            hit = next((r for r in rows if r["wilson_lcb_95"] >= pmin), None)
            out[label] = hit["budget_scenes"] if hit else None
            if label == "budget_p008":
                out["rate"] = hit["empirical_pass_rate"] if hit else None
                out["lcb"] = hit["wilson_lcb_95"] if hit else None
        return out

    print(f"\nConditional order preservation at 48 scenes "
          f"(P(order | E_joint)): {cond48 if cond48 is not None else 'n/a'} "
          f"(unconditional {n_order_uncond_48}/400)")

    pd.DataFrame(rows_a).to_csv(OUT_DIR / "e2_variantA_frontier.csv", index=False)
    pd.DataFrame(rows_b).to_csv(OUT_DIR / "e2_variantB_frontier.csv", index=False)

    e2 = {
        "description": (
            "Ranking preservation added to E1's joint event, on the same "
            "3,473-scene intersection with identical paired candidate draws "
            "(rng=default_rng(seed+b), seed 0, M=400). Variant A: E_joint AND "
            "sign preservation of all 6 pairwise method mean gaps (3 pairs x "
            "{PSNR, SSIM}). Variant B: variant A AND gap-magnitude "
            "preservation within 0.5 dB / 0.01. Instant-NGP PSNR uses the "
            "per-frame-mean 'PNSR' column as in E1."
        ),
        "population_size": 3473,
        "num_trials": NUM_TRIALS,
        "seed": SEED,
        "per_budget_rng_seed_rule": "numpy default_rng(seed + b), balanced mode",
        "full_intersection_reference_gaps": {
            f"{a}_minus_{b_}_{m}": g for (a, b_, m), g in full_gaps.items()
        },
        "gap_magnitude_tolerances": GAP_TOL,
        "variantA": {"frontier": rows_a, **targets(rows_a)},
        "variantB": {"frontier": rows_b, **targets(rows_b)},
        "conditional_order_preservation_at_48": cond48,
        "unconditional_order_preservation_at_48": n_order_uncond_48 / NUM_TRIALS,
        "delta_vs_E1": delta_vs_e1,
        "outputs": {
            "variantA_frontier_csv": "rebuttal/e2_variantA_frontier.csv",
            "variantB_frontier_csv": "rebuttal/e2_variantB_frontier.csv",
        },
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E2": e2})
    print(f"Wrote E2 key into {OUT_DIR / 'rebuttal_results.json'}")


if __name__ == "__main__":
    main()
