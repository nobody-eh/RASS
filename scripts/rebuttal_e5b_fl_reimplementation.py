#!/usr/bin/env python3
"""Key E5 option (b): LABELED REIMPLEMENTATION of the facility-location
baseline and full audit of a newly certified FL-36' subset.

PROVENANCE: the paper's original FL selection script was not preserved
(E5.recovery_exhausted). This is a documented reimplementation of the
recorded protocol (baseline_eval_config.json: M=400, random_seed=0,
facility_candidate_pool=16, paper tolerances, 3521-scene overlap); its
outputs are presented as a NEW run (FL-36'), never as the paper's original
subset. Sanity check: the reimplemented sweep curve is compared to the
recorded baseline_sweep_results.csv FL row qualitatively, not expected to
match run-for-run.

Protocol (declared):
- Space: 54 normalized descriptors (57 minus budget proxies), restricted to
  the 3521-scene Zip-NeRF audit population, StandardScaler-standardized
  (same recipe as the validated clustering replication).
- Objective: facility location - minimize sum over all scenes of the
  Euclidean distance to the nearest selected scene.
- Greedy with stochastic candidate pools: at each step, draw 16 unselected
  candidates uniformly (rng per run), add the one that most reduces the
  objective. This is the natural reading of 'facility_candidate_pool: 16'
  and reproduces the recorded per-run runtimes (~0.01-0.04 s).
- Perturbation protocol: run r in {0..M-1} uses rng = default_rng([0, r]).
- Certified FL-36': among the 400 size-36 runs, the run passing the
  Zip-NeRF joint event with the best (lowest) joint objective.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _load_mapping_csv,
    _load_subset_ids,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import THRESHOLDS, merge_results_json  # noqa: E402
from rebuttal_e1_joint_event import RASS48, load_populations  # noqa: E402
from rebuttal_e3_sensitivity import FEATS, PROXIES, SHIPPED_MAPPINGS, ZIPNERF_LOG, fast_ks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
SUBSET_DIR = REPO / "subsets" / "fl36"
M = 400
POOL = 16
SIZES = [12, 24, 36, 48, 60, 72, 96]
TARGET_SIZE = 36
P_MIN = 0.08
STABILITY_SEEDS = 20
RANDOM_N = 1000
RANDOM_SEED = 42
METRICS = ("psnr", "ssim", "lpips")


def greedy_facility(D: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    n = D.shape[0]
    selected: list[int] = []
    min_dist = np.full(n, np.inf)
    unsel = np.ones(n, dtype=bool)
    for _ in range(size):
        pool_idx = rng.choice(np.flatnonzero(unsel), size=POOL, replace=False)
        best_c, best_obj = -1, np.inf
        for c in pool_idx:
            obj = np.minimum(min_dist, D[c]).sum()
            if obj < best_obj:
                best_obj, best_c = obj, c
        selected.append(best_c)
        min_dist = np.minimum(min_dist, D[best_c])
        unsel[best_c] = False
    return np.array(sorted(selected), dtype=int)


def main() -> None:
    t0 = time.time()
    # ---------------- population and features ----------------
    mapping = _load_mapping_csv(SHIPPED_MAPPINGS[6])
    zdf = _merge_mapping_and_metrics(mapping, _load_zipnerf_metrics(ZIPNERF_LOG)).reset_index(drop=True)
    assert len(zdf) == 3521
    feats = pd.read_csv(FEATS)
    cols = [c for c in feats.columns if c != "dish_id" and c not in PROXIES]
    feats = feats.set_index("dish_id").loc[zdf["dish_id"]][cols]
    X = feats.fillna(feats.mean())
    var = X.var(axis=0)
    X = X.drop(columns=var[var == 0.0].index.tolist())
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    from scipy.spatial.distance import cdist
    tD = time.time()
    D = cdist(Xs, Xs).astype(np.float32)
    print(f"pairwise precompute: {time.time()-tD:.2f}s (recorded: 1.18s)")

    vals = {m: zdf[m].to_numpy(dtype=float) for m in METRICS}
    pop_sorted = {m: np.sort(v) for m, v in vals.items()}
    pop_mean = {m: float(v.mean()) for m, v in vals.items()}
    ids = zdf["dish_id"].astype(str).to_numpy()

    def joint_eval(idx: np.ndarray):
        gaps, kss = {}, {}
        for m in METRICS:
            sv = vals[m][idx]
            gaps[m] = abs(float(sv.mean()) - pop_mean[m])
            kss[m] = fast_ks(np.sort(sv), pop_sorted[m])
        mean_obj = max(gaps[m] / THRESHOLDS[f"{m}_tol"] for m in METRICS)
        joint_obj = max(mean_obj, max(kss.values()) / THRESHOLDS["ks_tol"])
        ok = mean_obj <= 1.0 and max(kss.values()) <= THRESHOLDS["ks_tol"]
        return ok, joint_obj, gaps, kss

    # ---------------- sweep: reimplemented FL curve ----------------
    recorded = pd.read_csv(REPO / "sweep_cluster_k/baseline_comparison_lpips_ks/baseline_sweep_results.csv")
    recorded_fl = recorded[recorded["baseline"] == "facility_location"].set_index("subset_size")
    sweep_rows = []
    best36 = None
    runs36 = []
    for size in SIZES:
        n_pass = 0
        t1 = time.time()
        for r in range(M):
            rng = np.random.default_rng([0, r])
            idx = greedy_facility(D, size, rng)
            ok, jobj, _, _ = joint_eval(idx)
            n_pass += int(ok)
            if size == TARGET_SIZE:
                runs36.append((r, ok, jobj, idx))
                if ok and (best36 is None or jobj < best36[2]):
                    best36 = (r, ok, jobj, idx)
        p = n_pass / M
        lcb = _wilson_lower_bound(p, M, 0.95)
        rec_p = float(recorded_fl.loc[size, "pass_rate"]) if size in recorded_fl.index else None
        sweep_rows.append({"subset_size": size, "n_pass": n_pass, "pass_rate": p,
                           "wilson_lcb_95": lcb, "recorded_pass_rate": rec_p,
                           "runtime_sec_per_run": (time.time()-t1)/M})
        print(f"size {size:3d}: {n_pass:3d}/400 pass (LCB {lcb:.4f}) | recorded {rec_p}")

    if best36 is None:
        raise SystemExit("No size-36 FL run passed the joint event; report as-is and stop.")
    fl_run, _, fl_jobj, fl_idx = best36
    fl_ids = sorted(ids[fl_idx].tolist())
    print(f"certified FL-36': run {fl_run}, joint objective {fl_jobj:.4f}")

    # ---------------- step 2: export validation table ----------------
    ok, jobj, gaps, kss = joint_eval(fl_idx)
    export_audit = {"pass": bool(ok), "joint_objective": fl_jobj, "constraints": []}
    for m in METRICS:
        export_audit["constraints"].append({
            "metric": m, "abs_mean_gap": gaps[m], "tolerance": THRESHOLDS[f"{m}_tol"],
            "normalized_gap": gaps[m] / THRESHOLDS[f"{m}_tol"],
            "ks": kss[m], "ks_tolerance": THRESHOLDS["ks_tol"],
            "mean_pass": gaps[m] <= THRESHOLDS[f"{m}_tol"],
            "ks_pass": kss[m] <= THRESHOLDS["ks_tol"],
        })

    # ---------------- steps 3-4: cross-method on 3473 intersection ----------------
    pop_k6, pops, common = load_populations()
    oc_raw = pd.read_csv(REPO / "ingp_oc.csv")
    import re
    oc_raw["dish_id"] = oc_raw["dish_id"].map(lambda s: (re.search(r"(dish_\w+)", str(s)) or [s, str(s)])[1] if re.search(r"(dish_\w+)", str(s)) else str(s).strip())
    oc = pd.DataFrame({"dish_id": oc_raw["dish_id"], "psnr": oc_raw["PSNR"], "ssim": oc_raw["SSIM"]}).groupby("dish_id", as_index=False).mean()
    method_tables = {**{k: (v, [c for c in v.columns if c != "dish_id"]) for k, v in pops.items()},
                     "instant_ngp_oc": (oc[oc["dish_id"].isin(common)].reset_index(drop=True), ["psnr", "ssim"])}
    fl_set = set(fl_ids)
    fl_in_common = fl_set & common
    cross = {}
    for name, (tbl, mets) in method_tables.items():
        pop = tbl[tbl["dish_id"].isin(common)]
        sub = pop[pop["dish_id"].isin(fl_in_common)]
        cross[name] = {"subset_n": int(len(sub)),
                       **{f"abs_{m}_gap": abs(float(sub[m].mean() - pop[m].mean())) for m in mets}}
    rank = {}
    for m in ("psnr", "ssim"):
        okm = True
        for a, b in itertools.combinations(["zipnerf", "feature_splatting", "instant_ngp_fi"], 2):
            fa, fb = method_tables[a][0], method_tables[b][0]
            fga = fa[fa["dish_id"].isin(common)][m].mean() - fb[fb["dish_id"].isin(common)][m].mean()
            sga = fa[fa["dish_id"].isin(fl_in_common)][m].mean() - fb[fb["dish_id"].isin(fl_in_common)][m].mean()
            okm = okm and (np.sign(fga) == np.sign(sga))
        rank[m] = bool(okm)

    # ---------------- step 5: regime coverage ----------------
    reg = zdf[zdf["dish_id"].isin(fl_set)].groupby("cluster").size().to_dict()
    reg = {int(k): int(v) for k, v in reg.items()}
    for c in sorted(zdf["cluster"].unique()):
        reg.setdefault(int(c), 0)
    thin = [c for c, n in reg.items() if n < 3]

    # ---------------- step 6: percentiles vs 1000 random 36-subsets ----------------
    rngp = np.random.default_rng(RANDOM_SEED)
    common_ids = np.array(sorted(common))
    rand_gaps = {(n, m): [] for n, (t, ms) in method_tables.items() for m in ms}
    for _ in range(RANDOM_N):
        ridx = set(rngp.choice(common_ids, size=TARGET_SIZE, replace=False).tolist())
        for name, (tbl, mets) in method_tables.items():
            pop = tbl[tbl["dish_id"].isin(common)]
            sub = pop[pop["dish_id"].isin(ridx)]
            for m in mets:
                rand_gaps[(name, m)].append(abs(float(sub[m].mean() - pop[m].mean())))
    percentiles = {}
    for (name, m), arr in rand_gaps.items():
        arr = np.array(arr)
        fl_gap = cross[name][f"abs_{m}_gap"]
        percentiles[f"{name}.{m}"] = float((arr < fl_gap).mean() * 100)
    avg_pct = float(np.mean(list(percentiles.values())))

    # ---------------- step 7: stability over 20 seeds ----------------
    stab_sets, stab_pass = [], 0
    for s in range(STABILITY_SEEDS):
        rng = np.random.default_rng([1000, s])
        idx = greedy_facility(D, TARGET_SIZE, rng)
        oks, _, _, _ = joint_eval(idx)
        stab_pass += int(oks)
        stab_sets.append(set(ids[idx].tolist()))
    overlaps = [len(a & b) / TARGET_SIZE for a, b in itertools.combinations(stab_sets, 2)]
    mean_overlap = float(np.mean(overlaps))

    # ---------------- step 8: package ----------------
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"dish_id": fl_ids}).merge(
        zdf[["dish_id", "cluster"]], on="dish_id"
    ).to_csv(SUBSET_DIR / "fl36_scene_list.csv", index=False)
    card = {
        "name": "FL-36-prime",
        "provenance": (
            "LABELED REIMPLEMENTATION (2026-07-24). The paper's original FL "
            "selection script was not preserved; this subset was produced "
            "by scripts/rebuttal_e5b_fl_reimplementation.py per the recorded "
            "protocol (baseline_eval_config.json) and is a NEW run, not the "
            "paper's original FL-36."
        ),
        "protocol": {
            "objective": "greedy facility location (sum of Euclidean distances to nearest selected), stochastic candidate pool of 16 per step",
            "feature_space": "54 normalized descriptors (57 minus budget proxies), StandardScaler, 3521-scene audit population",
            "M": M, "pool": POOL, "run_seed_rule": "default_rng([0, run_index])",
            "certified_run_index": int(fl_run),
            "thresholds": THRESHOLDS,
        },
        "scene_ids": fl_ids,
        "export_audit": export_audit,
        "cross_method_gaps_3473": cross,
        "rank_preserved": rank,
        "regime_counts_k6": reg,
        "stability": {"seeds": STABILITY_SEEDS, "mean_pairwise_overlap": mean_overlap,
                      "pass_fraction": stab_pass / STABILITY_SEEDS},
    }
    (SUBSET_DIR / "fl36_audit_card.json").write_text(json.dumps(card, indent=2) + "\n")
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "e5b_fl_sweep.csv", index=False)

    # ---------------- results JSON ----------------
    existing = json.loads((OUT_DIR / "rebuttal_results.json").read_text())
    e5 = existing["E5"]
    e5["status"] = "reimplemented (option b) - original still unrecoverable"
    e5["reimplementation"] = {
        "sweep": sweep_rows,
        "export_audit": export_audit,
        "cross_method_gaps": cross,
        "rank_preserved": rank,
        "regime_counts": reg,
        "thin_regimes": thin,
        "random_percentiles": {"per_pair": percentiles, "average": avg_pct,
                               "n_random": RANDOM_N, "seed": RANDOM_SEED,
                               "definition": "percent of 1000 uniform random 36-scene subsets (common 3473 pool) with SMALLER abs mean gap than FL-36' (lower = FL better)"},
        "stability": card["stability"],
        "artifact_paths": {"scene_list": "subsets/fl36/fl36_scene_list.csv",
                           "audit_card": "subsets/fl36/fl36_audit_card.json",
                           "sweep_csv": "rebuttal/e5b_fl_sweep.csv"},
        "runtime_sec_total": time.time() - t0,
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E5": e5})
    print(f"\nFL-36' in intersection: {len(fl_in_common)}/36; regime coverage {reg} thin={thin}")
    print(f"rank preserved: {rank}; percentile avg {avg_pct:.1f}; "
          f"stability overlap {mean_overlap:.3f}, pass {stab_pass}/{STABILITY_SEEDS}")
    print("Wrote E5.reimplementation + subsets/fl36/")


if __name__ == "__main__":
    main()
