#!/usr/bin/env python3
"""Task P17 audit stage: two-method DL3DV joint event (nerfacto + splatfacto).

CAMERA-READY MATERIAL, reported separately from the discussion-period E13
result. The E13 contract is extended method-wise and otherwise unchanged:
k=4 regimes on the same descriptors, E3c dispersion-rule tolerances computed
per method from that method's own per-scene stds, the same size-dependent KS
guardrail, balanced generator, M=400, seed 0. A candidate subset passes only
if EVERY method's mean gaps and KS distances are within tolerance
simultaneously.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuttal_audit_tasks import merge_results_json  # noqa: E402
from rebuttal_e9_available_data_comparison import proportional_counts  # noqa: E402
from rebuttal_e13_dl3dv_audit import (  # noqa: E402
    AUDIT_SEED, BUDGETS_B, C_ANCHOR, K, KS_CONST, M, META, cluster_dl3dv,
)
from rebuttal_e3_sensitivity import (  # noqa: E402
    SHIPPED_MAPPINGS, ZIPNERF_LOG, Sweeper, _load_mapping_csv,
    _load_zipnerf_metrics, _merge_mapping_and_metrics, _wilson_lower_bound,
    fast_ks,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "rebuttal/rebuttal_results.json"
SCRATCH = Path(os.environ.get("RASS_SCRATCH", "/tmp/rass_scratch"))
NERFACTO_DIR = REPO / "rebuttal/method_logs/dl3dv_nerfacto"
# read from the committed release copies: scratch is /tmp and does not
# survive a reboot (it already ate the splatfacto/tensorf staging copies once)
SPLAT_DIR = REPO / "rebuttal/method_logs/dl3dv_splatfacto"
TENSORF_DIR = REPO / "rebuttal/method_logs/dl3dv_tensorf"
INGP_DIR = REPO / "rebuttal/method_logs/dl3dv_instant_ngp"
METRICS = ("psnr", "ssim", "lpips")
P_MIN, P_STRICT = 0.08, 0.20


def load_method(d: Path, hashes: list[str]) -> pd.DataFrame:
    rows = []
    for h in hashes:
        p = d / f"{h}.json"
        if not p.exists():
            raise SystemExit(f"STOP: missing log {p}")
        r = json.load(open(p))["results"]
        rows.append({"dish_id": h, **{m: r[m] for m in METRICS}})
    return pd.DataFrame(rows)


def main() -> None:
    # validation gate first, exactly as E13
    ref = Sweeper(_merge_mapping_and_metrics(
        _load_mapping_csv(SHIPPED_MAPPINGS[6]), _load_zipnerf_metrics(ZIPNERF_LOG)
    )[["dish_id", "psnr", "ssim", "lpips", "cluster"]])
    n_ref, lcb_ref, _ = ref.run_budget(8, AUDIT_SEED)
    print(f"validation b=8 k=6: {n_ref}/400, LCB {lcb_ref:.4f}")
    assert n_ref == 88 and abs(lcb_ref - 0.182) < 5e-4, "STOP: validation failed"

    hashes = pd.read_csv(META)["hash"].astype(str).tolist()
    methods = {"nerfacto": load_method(NERFACTO_DIR, hashes),
               "splatfacto": load_method(SPLAT_DIR, hashes)}
    if TENSORF_DIR.exists() and len(list(TENSORF_DIR.glob("*.json"))) == len(hashes):
        methods["tensorf"] = load_method(TENSORF_DIR, hashes)
    # instant-ngp trained to 16k steps, NOT the 30k default used by the other
    # three (NVlabs run.py falls back to 35k). Within-method constraints keep
    # this audit valid, but no cross-method quality claim may be drawn.
    if INGP_DIR.exists() and len(list(INGP_DIR.glob("*.json"))) == len(hashes):
        methods["instant-ngp"] = load_method(INGP_DIR, hashes)
    labels, ndims = cluster_dl3dv(hashes)
    N = len(hashes)
    groups = {int(c): np.where(labels == c)[0] for c in sorted(set(labels))}
    regime_sizes = {c: len(g) for c, g in groups.items()}
    print(f"population {N}, {ndims} dims, regimes {regime_sizes}")

    stats, vals, pop_sorted, pop_mean, taus = {}, {}, {}, {}, {}
    for name, df in methods.items():
        df = df.set_index("dish_id").loc[hashes]
        vals[name] = {m: df[m].to_numpy(float) for m in METRICS}
        pop_sorted[name] = {m: np.sort(v) for m, v in vals[name].items()}
        pop_mean[name] = {m: float(v.mean()) for m, v in vals[name].items()}
        sd = {m: float(df[m].std(ddof=1)) for m in METRICS}
        taus[name] = {m: C_ANCHOR * sd[m] for m in METRICS}
        stats[name] = {"means": pop_mean[name], "stds_ddof1": sd,
                       "tolerances": taus[name]}
        print(f"  {name}: " + " ".join(
            f"{m} {pop_mean[name][m]:.3f}+-{sd[m]:.3f} (tau {taus[name][m]:.4f})"
            for m in METRICS))

    def evaluate(idx, ks_tol):
        for name in methods:
            for m in METRICS:
                sv = vals[name][m][idx]
                if abs(float(sv.mean()) - pop_mean[name][m]) > taus[name][m]:
                    return False
                if fast_ks(np.sort(sv), pop_sorted[name][m]) > ks_tol:
                    return False
        return True

    def frontier(alloc):
        rows = []
        for b in BUDGETS_B:
            size = K * b
            counts = alloc(b, size)
            if counts is None:
                continue
            ks_tol = KS_CONST * np.sqrt((size + N) / (size * N))
            rng = np.random.default_rng(AUDIT_SEED + b)
            n_pass = 0
            for _ in range(M):
                idx = np.sort(np.concatenate(
                    [rng.choice(groups[c], size=counts[c], replace=False)
                     for c in sorted(groups)]))
                n_pass += int(evaluate(idx, ks_tol))
            lcb = _wilson_lower_bound(n_pass / M, M, 0.95)
            rows.append({"b": b, "size": size, "n_pass": n_pass,
                         "empirical_pass_rate": n_pass / M,
                         "wilson_lcb_95": lcb, "ks_tol": float(ks_tol)})
            print(f"    size={size:>3}: {n_pass}/400, LCB {lcb:.4f}")
        return rows

    print("  balanced (equal):")
    eq = frontier(lambda b, size: ({c: b for c in groups}
                                   if b <= min(regime_sizes.values()) else None))
    print("  proportional:")
    pr = frontier(lambda b, size: (lambda cs: cs if all(
        cs[c] <= regime_sizes[c] for c in cs) else None)(
        proportional_counts(regime_sizes, size)))

    first = lambda rows, t: next(
        (r["size"] for r in rows if r["wilson_lcb_95"] >= t), None)
    targets = {"equal": {"p008": first(eq, P_MIN), "p020": first(eq, P_STRICT)},
               "proportional": {"p008": first(pr, P_MIN),
                                "p020": first(pr, P_STRICT)}}
    print("targets:", targets)

    e17 = json.loads(RESULTS.read_text())["E17"]
    e17["progress"]["status"] = "splatfacto COMPLETE 140/140, zero failures"
    e17["progress"]["scenes_completed"] = 140
    e17["audit"] = {
        "label": f"CAMERA-READY MATERIAL - {len(methods)}-method DL3DV joint event; "
                 "NOT for the discussion-period rebuttal",
        "date": "2026-07-26",
        "methods": list(methods),
        "contract": "E13 contract extended method-wise: k=4 regimes, per-method "
                    "E3c dispersion tolerances (c=0.079769 x that method's own "
                    "per-scene std), KS 1.358*sqrt((n+140)/(140n)), M=400, "
                    "seed 0, rng default_rng(0+b); a subset passes only if ALL "
                    "methods satisfy every mean and KS constraint at once",
        "validation": {"reference": "b=8 k=6 Nutrition5k", "n_pass": n_ref,
                       "wilson_lcb_95": lcb_ref},
        "population": {"n_scenes": N, "regime_sizes": {str(k): v for k, v in regime_sizes.items()}},
        "per_method_stats": stats,
        "n_constraints": len(methods) * len(METRICS) * 2,
        "frontier_equal": eq,
        "frontier_proportional": pr,
        "budgets_at_targets": targets,
    }
    merge_results_json(RESULTS, {"E17": e17})
    print("merged E17.audit")


if __name__ == "__main__":
    main()
