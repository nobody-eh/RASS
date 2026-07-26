#!/usr/bin/env python3
"""Task P13 / key E13, audit stage: RASS audit on locally generated DL3DV-140
nerfacto logs.

Declared design (recorded in rebuttal_results.json["E13"] before any result):
- population: all 140 DL3DV-Benchmark scenes, nerfacto per-scene PSNR/SSIM/
  LPIPS generated locally (seed 0, downscale 4);
- k = 4 KMeans regimes on the extracted descriptors, same recipe as the
  validated Nutrition5k replication (exclude proxy dims, fillna mean, drop
  zero variance, StandardScaler, KMeans(random_state=0, n_init='auto'));
- mean tolerances by the E3c dispersion rule: tau_m = c * sigma_m with
  c = 0.5 / 6.2681 = 0.079769 (the Nutrition5k PSNR anchor), sigma_m the
  DL3DV per-scene std (ddof=1);
- KS guardrail from the two-sample critical-distance formula at each target
  subset size n against the N=140 population: D_crit = 1.358*sqrt((n+N)/(n*N));
- balanced generator, M = 400 trials, audit seed 0 (rng = default_rng(0+b)),
  uniform baseline rng = default_rng(10000+b);
- machinery revalidated against the paper reference (b=8, k=6 Nutrition5k ->
  88/400, LCB 0.182) before the DL3DV run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuttal_audit_tasks import merge_results_json  # noqa: E402
from rebuttal_e9_available_data_comparison import proportional_counts  # noqa: E402
from rebuttal_e3_sensitivity import (  # noqa: E402
    PROXIES,
    SHIPPED_MAPPINGS,
    ZIPNERF_LOG,
    Sweeper,
    _load_mapping_csv,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _wilson_lower_bound,
    fast_ks,
)

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("RASS_SCRATCH", "/tmp/rass_scratch"))
JSON_DIR = SCRATCH / "dl3dv_logs/json"
DESC_CSV = SCRATCH / "dl3dv_logs/dl3dv_descriptors.csv"
META = REPO / "benchmark-meta.csv"
RESULTS = REPO / "rebuttal/rebuttal_results.json"

M = 400
AUDIT_SEED = 0
K = 4
C_ANCHOR = 0.5 / 6.2681  # E3c rule: tau_PSNR / sigma_PSNR on Nutrition5k
KS_CONST = 1.358  # alpha = 0.05 two-sample critical constant
BUDGETS_B = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]  # subset sizes 8..80 of 140
P_MIN, P_STRICT = 0.08, 0.20


def load_dl3dv() -> pd.DataFrame:
    hashes = pd.read_csv(META)["hash"].astype(str).tolist()
    rows = []
    for h in hashes:
        p = JSON_DIR / f"{h}.json"
        if not p.exists():
            continue
        r = json.load(open(p))["results"]
        rows.append({"dish_id": h, "psnr": r["psnr"], "ssim": r["ssim"],
                     "lpips": r["lpips"]})
    df = pd.DataFrame(rows)
    missing = sorted(set(hashes) - set(df["dish_id"]))
    if missing:
        raise SystemExit(f"STOP: {len(missing)} scene logs missing: "
                         f"{[m[:12] for m in missing]}")
    return df


def cluster_dl3dv(scene_ids: list[str]) -> pd.Series:
    feats = pd.read_csv(DESC_CSV).drop_duplicates("dish_id", keep="first")
    feats = feats.set_index("dish_id").loc[scene_ids].reset_index()
    cols = [c for c in feats.columns if c != "dish_id" and c not in PROXIES]
    X = feats[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())
    var = X.var(axis=0)
    X = X.drop(columns=var[var == 0.0].index.tolist())
    X = X.dropna(axis=1, how="all")
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=K, random_state=AUDIT_SEED,
                    n_init="auto").fit_predict(Xs)
    return pd.Series(labels, index=feats["dish_id"]).loc[scene_ids].values, X.shape[1]


def uniform_frontier(df: pd.DataFrame, sizes_tols: list[tuple[int, dict]]):
    vals = {m: df[m].to_numpy(dtype=float) for m in ("psnr", "ssim", "lpips")}
    pop_sorted = {m: np.sort(v) for m, v in vals.items()}
    pop_mean = {m: float(v.mean()) for m, v in vals.items()}
    n_total = len(df)
    out = []
    for b, (size, tols) in zip(BUDGETS_B, sizes_tols):
        rng = np.random.default_rng(10000 + b)
        n_pass = 0
        for _ in range(M):
            idx = rng.choice(n_total, size=size, replace=False)
            ok = True
            for m in ("psnr", "ssim", "lpips"):
                sv = vals[m][idx]
                if abs(float(sv.mean()) - pop_mean[m]) > tols[m]:
                    ok = False
                    break
                if fast_ks(np.sort(sv), pop_sorted[m]) > tols["ks"]:
                    ok = False
                    break
            n_pass += int(ok)
        out.append({"size": size, "n_pass": n_pass,
                    "wilson_lcb_95": _wilson_lower_bound(n_pass / M, M, 0.95)})
    return out


def main() -> None:
    # --- 0. machinery revalidation on the paper reference -------------------
    # 3,521-scene Zip-NeRF population with the shipped k6 mapping (the frame
    # the 88/400 reference is defined on; E1's load_populations returns the
    # 3,473 cross-method intersection instead).
    ref_frame = _merge_mapping_and_metrics(
        _load_mapping_csv(SHIPPED_MAPPINGS[6]), _load_zipnerf_metrics(ZIPNERF_LOG)
    )
    ref = Sweeper(ref_frame[["dish_id", "psnr", "ssim", "lpips", "cluster"]])
    n_pass_ref, lcb_ref, _ = ref.run_budget(8, AUDIT_SEED)
    print(f"validation b=8 k=6: {n_pass_ref}/400, LCB {lcb_ref:.4f}")
    assert n_pass_ref == 88 and abs(lcb_ref - 0.182) < 5e-4, "STOP: validation failed"

    # --- 1. DL3DV population, regimes, contract -----------------------------
    df = load_dl3dv()
    labels, n_feat_dims = cluster_dl3dv(df["dish_id"].tolist())
    df["cluster"] = labels
    regime_sizes = df["cluster"].value_counts().sort_index().to_dict()
    print(f"population {len(df)}, {n_feat_dims} feature dims, "
          f"regimes {regime_sizes}")

    stds = {m: float(df[m].std(ddof=1)) for m in ("psnr", "ssim", "lpips")}
    taus = {m: C_ANCHOR * stds[m] for m in stds}
    N = len(df)
    sizes_tols = []
    for b in BUDGETS_B:
        n = K * b
        ks_tol = KS_CONST * np.sqrt((n + N) / (n * N))
        sizes_tols.append((n, {**taus, "ks": float(ks_tol)}))

    # --- 2. balanced frontier ----------------------------------------------
    sw = Sweeper(df)
    frontier = []
    for b, (size, tols) in zip(BUDGETS_B, sizes_tols):
        if b > min(regime_sizes.values()):
            print(f"skip b={b}: exceeds smallest regime ({min(regime_sizes.values())})")
            continue
        n_pass, lcb, _ = sw.run_budget(b, AUDIT_SEED, tols=tols)
        frontier.append({"b": b, "size": size, "n_pass": n_pass,
                         "empirical_pass_rate": n_pass / M,
                         "wilson_lcb_95": lcb, "ks_tol": tols["ks"]})
        print(f"b={b:>2} size={size:>3}: {n_pass}/400, LCB {lcb:.4f}, "
              f"KS_tol {tols['ks']:.4f}")

    # --- 2b. proportional-allocation companion (E10 conventions) ------------
    vals = {m: df[m].to_numpy(dtype=float) for m in ("psnr", "ssim", "lpips")}
    pop_sorted = {m: np.sort(v) for m, v in vals.items()}
    pop_mean = {m: float(v.mean()) for m, v in vals.items()}
    groups = {int(c): df.index[df["cluster"] == int(c)].to_numpy(dtype=int)
              for c in sorted(df["cluster"].unique())}
    reg_sizes = {c: len(groups[c]) for c in sorted(groups)}
    prop_frontier = []
    for b, (size, tols) in zip(BUDGETS_B, sizes_tols):
        counts = proportional_counts(reg_sizes, size)
        if any(counts[c] > reg_sizes[c] for c in reg_sizes):
            continue
        rng = np.random.default_rng(AUDIT_SEED + b)
        n_pass = 0
        for _ in range(M):
            idx = np.sort(np.concatenate(
                [rng.choice(groups[c], size=counts[c], replace=False)
                 for c in sorted(groups)]))
            ok = True
            for m in ("psnr", "ssim", "lpips"):
                sv = vals[m][idx]
                if abs(float(sv.mean()) - pop_mean[m]) > tols[m]:
                    ok = False
                    break
                if fast_ks(np.sort(sv), pop_sorted[m]) > tols["ks"]:
                    ok = False
                    break
            n_pass += int(ok)
        lcb = _wilson_lower_bound(n_pass / M, M, 0.95)
        prop_frontier.append({"b": b, "size": size, "counts": counts,
                              "n_pass": n_pass,
                              "empirical_pass_rate": n_pass / M,
                              "wilson_lcb_95": lcb})
        print(f"prop b={b:>2} size={size:>3}: {n_pass}/400, LCB {lcb:.4f}")

    rec = next((f for f in frontier if f["wilson_lcb_95"] >= P_MIN), None)
    rec_strict = next((f for f in frontier if f["wilson_lcb_95"] >= P_STRICT), None)
    rec_prop = next((f for f in prop_frontier if f["wilson_lcb_95"] >= P_MIN), None)
    rec_prop_strict = next(
        (f for f in prop_frontier if f["wilson_lcb_95"] >= P_STRICT), None)

    # --- 3. uniform baseline ------------------------------------------------
    uni = uniform_frontier(df, sizes_tols)
    for u in uni:
        print(f"uniform size={u['size']:>3}: {u['n_pass']}/400, "
              f"LCB {u['wilson_lcb_95']:.4f}")

    # --- 4. persist ---------------------------------------------------------
    audit = {
        "date": "2026-07-26",
        "population": {"n_scenes": N, "method": "nerfacto (local RTX 3090, "
                       "ns-train defaults, seed 0, downscale 4)",
                       "means": {m: float(df[m].mean()) for m in stds},
                       "stds_ddof1": stds},
        "validation": {"reference": "b=8 k=6 Nutrition5k",
                       "n_pass": n_pass_ref, "wilson_lcb_95": lcb_ref},
        "contract": {"k": K, "c_anchor": C_ANCHOR, "mean_tolerances": taus,
                     "ks_rule": "1.358*sqrt((n+140)/(140n)) per subset size n",
                     "M": M, "audit_seed": AUDIT_SEED,
                     "generator_rng": "default_rng(audit_seed + b) balanced, "
                                      "default_rng(10000 + b) uniform"},
        "regime_sizes": {str(k): int(v) for k, v in regime_sizes.items()},
        "feature_dims_used": int(n_feat_dims),
        "balanced_frontier": frontier,
        "proportional_frontier": prop_frontier,
        "uniform_frontier": uni,
        "recommended_p008_balanced": rec,
        "recommended_p020_balanced": rec_strict,
        "recommended_p008_proportional": rec_prop,
        "recommended_p020_proportional": rec_prop_strict,
    }
    e13 = json.loads(RESULTS.read_text())["E13"]
    e13["audit"] = audit
    e13["progress"]["status"] = "COMPLETE 2026-07-26 - all 140 scene logs generated"
    e13["progress"]["scenes_completed"] = N
    merge_results_json(RESULTS, {"E13": e13})
    print("merged E13.audit into rebuttal_results.json")


if __name__ == "__main__":
    main()
