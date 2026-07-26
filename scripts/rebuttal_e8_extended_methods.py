#!/usr/bin/env python3
"""Key E8: cross-method joint audit extended with nerfacto and bionerf.

New method logs recovered from cluster (student scratch, pulled 2026-07-24):
- nerfacto: per-scene PSNR/SSIM/LPIPS JSONs (output_json_nerfacto/json).
- bionerf: per-scene fine_psnr/fine_ssim/fine_lpips JSONs, deduped across
  three run dirs with priority output_json_bionerf > output_json5 >
  output_json6 (first occurrence wins).

Events (same tolerances, generator, and seed protocol as E1):
- E_joint4: E1's 8 method-metric pairs + nerfacto (PSNR, SSIM, LPIPS) =
  11 pairs / 22 constraints, on I4 = E1 intersection ∩ nerfacto.
- E_joint5: + bionerf (PSNR, SSIM, LPIPS) = 14 pairs / 28 constraints, on
  I5 = I4 ∩ bionerf.
All reference means/CDFs recomputed on the corresponding intersection.
Balanced generator on k=6 regimes restricted to each intersection, M=400,
seed 0, rng=default_rng(seed+b). Instant-NGP PSNR = per-frame-mean 'PNSR'.
"""

from __future__ import annotations

import os
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _build_groups,
    _load_subset_ids,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import (  # noqa: E402
    BUDGETS,
    NUM_TRIALS,
    SEED,
    THRESHOLDS,
    merge_results_json,
)
from rebuttal_e1_joint_event import RASS48, RASS96, load_populations  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
SCRATCH = Path(os.environ.get("RASS_SCRATCH", "/tmp/rass_scratch")) / "cluster_logs"
P_MIN = 0.08


def load_json_dir(pattern: str, psnr_key: str, ssim_key: str, lpips_key: str) -> dict:
    out = {}
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        r = d.get("results", {})
        dish = str(d.get("experiment_name", Path(f).stem))
        vals = (r.get(psnr_key), r.get(ssim_key), r.get(lpips_key))
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        out[dish] = vals
    return out


def to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"dish_id": k, "psnr": v[0], "ssim": v[1], "lpips": v[2]} for k, v in d.items()]
    )


def eval_constraints(pops: dict, subset_ids: set, method_metrics: dict):
    rows = []
    joint = True
    for name, metrics in method_metrics.items():
        pop = pops[name]
        sub = pop[pop["dish_id"].isin(subset_ids)]
        m_pass = True
        for metric in metrics:
            fv = pop[metric].to_numpy(dtype=float)
            sv = sub[metric].to_numpy(dtype=float)
            gap = abs(float(np.mean(sv) - np.mean(fv)))
            ks = float(ks_2samp(fv, sv).statistic)
            ok = gap <= THRESHOLDS[f"{metric}_tol"] and ks <= THRESHOLDS["ks_tol"]
            rows.append({"method": name, "metric": metric, "abs_mean_gap": gap,
                         "ks": ks, "mean_pass": gap <= THRESHOLDS[f"{metric}_tol"],
                         "ks_pass": ks <= THRESHOLDS["ks_tol"]})
            m_pass = m_pass and ok
        joint = joint and m_pass
    return rows, joint


def run_event(tag: str, pop_k6: pd.DataFrame, pops: dict, method_metrics: dict):
    groups = _build_groups(pop_k6)
    support = {int(c): int(len(v)) for c, v in groups.items()}
    bs = [b for b in BUDGETS if b <= min(support.values())]
    ids_pop = pop_k6["dish_id"].astype(str).to_numpy()
    print(f"[{tag}] population {len(pop_k6)}, regime support {support}, budgets {bs}")

    frontier = []
    per_method = {}
    for b in bs:
        rng = np.random.default_rng(SEED + b)
        n_joint = 0
        n_m = {m: 0 for m in method_metrics}
        for _ in range(NUM_TRIALS):
            picks = [rng.choice(groups[c], size=b, replace=False) for c in sorted(groups)]
            idx = np.sort(np.concatenate(picks))
            sids = set(ids_pop[idx].tolist())
            rows, joint = eval_constraints(pops, sids, method_metrics)
            for m in method_metrics:
                ok = all(r["mean_pass"] and r["ks_pass"] for r in rows if r["method"] == m)
                n_m[m] += int(ok)
            n_joint += int(joint)
        p = n_joint / NUM_TRIALS
        lcb = _wilson_lower_bound(p, NUM_TRIALS, 0.95)
        frontier.append({"budget_scenes": 6 * b, "b": b, "n_pass": n_joint,
                         "empirical_pass_rate": p, "wilson_lcb_95": lcb,
                         "rng_seed_effective": SEED + b})
        per_method[6 * b] = dict(n_m)
        print(f"  b={b:2d} ({6*b:3d}sc): joint {n_joint:3d}/400 (LCB {lcb:.4f}) | "
              + ", ".join(f"{m}={c}" for m, c in n_m.items()))

    hit = next((r for r in frontier if r["wilson_lcb_95"] >= P_MIN), None)
    rass = {}
    for label, path in (("rass48", RASS48), ("rass96", RASS96)):
        ids = set(_load_subset_ids(path))
        inside = ids & set(ids_pop.tolist())
        rows, joint = eval_constraints(pops, inside, method_metrics)
        viol = [f"{r['method']}.{r['metric']}" for r in rows
                if not (r["mean_pass"] and r["ks_pass"])]
        rass[label] = {"nominal_size": len(ids), "scenes_in_intersection": len(inside),
                       "pass": bool(joint), "violated": viol, "constraints": rows}
        print(f"  {label}: {len(inside)}/{len(ids)} in intersection, "
              f"{'PASS' if joint else 'FAIL'} violated={viol}")
    return {
        "population_size": int(len(pop_k6)),
        "regime_support": support,
        "frontier": frontier,
        "per_method_n_pass_by_budget": per_method,
        "budget_p008": hit["budget_scenes"] if hit else "not reached",
        "rass48": {k: v for k, v in rass["rass48"].items() if k != "constraints"},
        "rass48_constraints": rass["rass48"]["constraints"],
        "rass96": {k: v for k, v in rass["rass96"].items() if k != "constraints"},
        "rass96_constraints": rass["rass96"]["constraints"],
    }


def main() -> None:
    # New method tables.
    nerfacto = load_json_dir(str(SCRATCH / "output_json_nerfacto/json/*.json"),
                             "psnr", "ssim", "lpips")
    bionerf_runs = [
        ("output_json_bionerf", str(SCRATCH / "output_json_bionerf/DISH/*.json")),
        ("output_json5", str(SCRATCH / "output_json5/DISH/*.json")),
        ("output_json6", str(SCRATCH / "output_json6/DISH/*.json")),
    ]
    bionerf = {}
    per_run_counts = {}
    for run_name, pat in bionerf_runs:
        d = load_json_dir(pat, "fine_psnr", "fine_ssim", "fine_lpips")
        per_run_counts[run_name] = len(d)
        for k, v in d.items():
            bionerf.setdefault(k, v)  # priority order: first occurrence wins
    nerfacto_df, bionerf_df = to_df(nerfacto), to_df(bionerf)
    print(f"nerfacto scenes: {len(nerfacto_df)}; bionerf scenes (deduped): "
          f"{len(bionerf_df)} from {per_run_counts}")

    # E1 populations (3,473 intersection) + regime labels.
    pop_k6, pops, common = load_populations()
    i4 = common & set(nerfacto_df["dish_id"])
    i5 = i4 & set(bionerf_df["dish_id"])
    print(f"I4 (E1 ∩ nerfacto): {len(i4)}; I5 (I4 ∩ bionerf): {len(i5)}")

    results = {"inventory": {
        "nerfacto_scenes": len(nerfacto_df),
        "bionerf_scenes_deduped": len(bionerf_df),
        "bionerf_per_run_counts": per_run_counts,
        "bionerf_metric_convention": "fine_psnr/fine_ssim/fine_lpips (fine network)",
        "I4": len(i4), "I5": len(i5),
        "source": "group cluster scratch, pulled 2026-07-24",
    }}

    for tag, isect, extra in (
        ("E_joint4", i4, {"nerfacto": nerfacto_df}),
        ("E_joint5", i5, {"nerfacto": nerfacto_df, "bionerf": bionerf_df}),
    ):
        pk = pop_k6[pop_k6["dish_id"].isin(isect)].reset_index(drop=True)
        mm = {"zipnerf": ["psnr", "ssim", "lpips"],
              "feature_splatting": ["psnr", "ssim", "lpips"],
              "instant_ngp_fi": ["psnr", "ssim"]}
        pp = {}
        for name, df in pops.items():
            pp[name] = df[df["dish_id"].isin(isect)].reset_index(drop=True)
        for name, df in extra.items():
            pp[name] = df[df["dish_id"].isin(isect)].reset_index(drop=True)
            mm[name] = ["psnr", "ssim", "lpips"]
        results[tag] = run_event(tag, pk, pp, mm)

    results["description"] = (
        "E1's formal joint event extended with nerfacto (E_joint4, 11 "
        "method-metric pairs) and nerfacto+bionerf (E_joint5, 14 pairs) on "
        "the correspondingly reduced intersections; references recomputed "
        "per intersection; same tolerances, balanced k=6 generator, M=400, "
        "seed 0 as E1."
    )
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E8": results})
    print("Wrote E8 key")


if __name__ == "__main__":
    main()
