#!/usr/bin/env python3
"""Task P4 / key E4: budget at which regime-level fidelity becomes achievable.

Event: Zip-NeRF global event (mean gaps within paper tolerances AND global
per-metric KS <= 0.14) AND per-regime mean constraints over the six k=6
regimes: for every regime r and metric m,
|mean_m(S∩r) - mean_m(full∩r)| <= c * tau_m, for c = 1 and c = 2 as
separate events. KS stays global only.

Generator: balanced sampler on the full audit population, budgets
b in {8, 12, 16, 20, 30, 40, 60} (48..360 scenes), M = 400 throughout
(fast evaluator; runtime does not demand reduction), seed 0,
rng = default_rng(seed + b).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _load_mapping_csv,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import merge_results_json  # noqa: E402
from rebuttal_e3_sensitivity import SHIPPED_MAPPINGS, ZIPNERF_LOG, fast_ks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
TOLS = {"psnr": 0.5, "ssim": 0.01, "lpips": 0.01}
KS_TOL = 0.14
BUDGETS = [8, 12, 16, 20, 30, 40, 60]
M = 400
SEED = 0
P_MIN = 0.08
METRICS = ("psnr", "ssim", "lpips")


def main() -> None:
    mapping = _load_mapping_csv(SHIPPED_MAPPINGS[6])
    df = _merge_mapping_and_metrics(mapping, _load_zipnerf_metrics(ZIPNERF_LOG))
    df = df.reset_index(drop=True)
    print(f"Population: {len(df)} scenes, regimes "
          f"{df.groupby('cluster').size().to_dict()}")

    vals = {m: df[m].to_numpy(dtype=float) for m in METRICS}
    pop_sorted = {m: np.sort(v) for m, v in vals.items()}
    pop_mean = {m: float(v.mean()) for m, v in vals.items()}
    clusters = sorted(df["cluster"].unique())
    groups = {int(c): df.index[df["cluster"] == int(c)].to_numpy(dtype=int) for c in clusters}
    regime_means = {
        (int(c), m): float(vals[m][groups[int(c)]].mean()) for c in clusters for m in METRICS
    }

    # Validation: global event at b=8 must reproduce 88/400.
    rng = np.random.default_rng(SEED + 8)
    n_val = 0
    for _ in range(M):
        picks = [rng.choice(groups[c], size=8, replace=False) for c in sorted(groups)]
        idx = np.sort(np.concatenate(picks))
        ok = all(
            abs(float(vals[m][idx].mean()) - pop_mean[m]) <= TOLS[m]
            and fast_ks(np.sort(vals[m][idx]), pop_sorted[m]) <= KS_TOL
            for m in METRICS
        )
        n_val += int(ok)
    if n_val != 88:
        raise SystemExit(f"VALIDATION FAILED: global event at b=8 gives {n_val}/400, want 88")
    print("[validation] global event reproduces 88/400 at b=8")

    frontier = {1: [], 2: []}
    m_used = {}
    for b in BUDGETS:
        rng = np.random.default_rng(SEED + b)
        n_pass = {1: 0, 2: 0}
        for _ in range(M):
            picks = {c: rng.choice(groups[c], size=b, replace=False) for c in sorted(groups)}
            idx = np.sort(np.concatenate(list(picks.values())))

            global_ok = True
            for m in METRICS:
                sv = vals[m][idx]
                if abs(float(sv.mean()) - pop_mean[m]) > TOLS[m]:
                    global_ok = False
                    break
                if fast_ks(np.sort(sv), pop_sorted[m]) > KS_TOL:
                    global_ok = False
                    break
            if not global_ok:
                continue

            worst_ratio = 0.0
            for c in sorted(groups):
                for m in METRICS:
                    gap = abs(float(vals[m][picks[c]].mean()) - regime_means[(c, m)])
                    worst_ratio = max(worst_ratio, gap / TOLS[m])
            if worst_ratio <= 1.0:
                n_pass[1] += 1
            if worst_ratio <= 2.0:
                n_pass[2] += 1

        m_used[str(6 * b)] = M
        for c_mult in (1, 2):
            p = n_pass[c_mult] / M
            frontier[c_mult].append(
                {
                    "budget_scenes": 6 * b,
                    "b": b,
                    "n_trials": M,
                    "n_pass": n_pass[c_mult],
                    "empirical_pass_rate": p,
                    "wilson_lcb_95": _wilson_lower_bound(p, M, 0.95),
                    "rng_seed_effective": SEED + b,
                }
            )
        print(f"b={b:2d} ({6*b:3d} scenes): c=1 {n_pass[1]:3d}/400 "
              f"(LCB {frontier[1][-1]['wilson_lcb_95']:.4f}) | "
              f"c=2 {n_pass[2]:3d}/400 (LCB {frontier[2][-1]['wilson_lcb_95']:.4f})")

    result = {}
    for c_mult in (1, 2):
        hit = next((r for r in frontier[c_mult] if r["wilson_lcb_95"] >= P_MIN), None)
        result[f"c{c_mult}"] = {
            "frontier": frontier[c_mult],
            "budget_p008_or_not_reached": (
                hit["budget_scenes"] if hit else "not reached up to 360"
            ),
        }
        print(f"c={c_mult}: smallest budget with LCB>=0.08: "
              f"{result[f'c{c_mult}']['budget_p008_or_not_reached']}")

    pd.DataFrame(frontier[1]).assign(c=1).to_csv(OUT_DIR / "e4_c1_frontier.csv", index=False)
    pd.DataFrame(frontier[2]).assign(c=2).to_csv(OUT_DIR / "e4_c2_frontier.csv", index=False)

    e4 = {
        "description": (
            "Regime-level fidelity: Zip-NeRF global event (paper tolerances, "
            "global KS<=0.14) AND per-regime mean constraints "
            "|mean_m(S∩r)-mean_m(full∩r)| <= c*tau_m over the six k=6 "
            "regimes, c in {1,2}. Balanced sampler on the full audit "
            "population (3521 effective scenes), budgets 48..360, M=400 at "
            "every budget (fast evaluator; no reduction needed), seed 0."
        ),
        "population_size": int(len(df)),
        "tolerances": TOLS,
        "ks_tolerance_global_only": KS_TOL,
        "budgets_scenes": [6 * b for b in BUDGETS],
        "m_used_per_budget": m_used,
        "seed": SEED,
        "per_budget_rng_seed_rule": "numpy default_rng(seed + b), balanced mode",
        "validation": "global event reproduces 88/400 at b=8",
        **result,
        "outputs": {
            "c1_frontier_csv": "rebuttal/e4_c1_frontier.csv",
            "c2_frontier_csv": "rebuttal/e4_c2_frontier.csv",
        },
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E4": e4})
    print(f"Wrote E4 key into {OUT_DIR / 'rebuttal_results.json'}")


if __name__ == "__main__":
    main()
