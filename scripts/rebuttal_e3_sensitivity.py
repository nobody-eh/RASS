#!/usr/bin/env python3
"""Task P3 / key E3: sensitivity and calibration numbers.

c) Calibration: KS critical distances, per-metric stds and tolerance
   fractions, minimum cross-method mean gaps on the intersection.
a) 4x4 tolerance sweep (tau_PSNR x tau_KS) reusing the stored per-trial
   statistics of the full budget sweep (identical candidate draws).
b) Regime-count grid k in {4,5,6,8,10} and a 5x5 (K-Means seed x audit seed)
   grid at k=6, plus Jaccard overlap of exported 48-scene subsets.

Clustering replication validated: shipped k=6 labels are reproduced exactly
(ARI=1.0) with features = 57 normalized descriptors minus the 3 budget
proxies (num_images, num_frames_total, num_frames_used), fillna(mean),
drop zero-variance, StandardScaler, KMeans(random_state=seed, n_init='auto')
under scikit-learn 1.4.2.

Audit event: Zip-NeRF joint event (paper tolerances unless swept), fast KS
evaluator validated to reproduce 88/400 at b=8 and 113/400 at b=16.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_strong_accept_results import (  # noqa: E402
    _load_mapping_csv,
    _load_zipnerf_metrics,
    _merge_mapping_and_metrics,
    _wilson_lower_bound,
)
from rebuttal_audit_tasks import merge_results_json  # noqa: E402
from rebuttal_e1_joint_event import load_populations  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "rebuttal"
ZIPNERF_LOG = REPO / "sweep_cluster_k/holdout_protocol_v2/joint_selection/zipnerf.xlsx"
FEATS = REPO / "sweep_cluster_k/k_6/feats_normalized.csv"
PROXIES = ["num_images", "num_frames_total", "num_frames_used"]
DEFAULT_TOLS = {"psnr": 0.5, "ssim": 0.01, "lpips": 0.01, "ks": 0.14}
P_MIN = 0.08
M = 400
STANDARD_SIZES = [24, 36, 48, 60, 72, 84, 96, 120]

SHIPPED_MAPPINGS = {
    4: REPO / "sweep_cluster_k/k_4/clustered_scenes_k4_dish_cluster_mapping.csv",
    6: REPO / "sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv",
    8: REPO / "sweep_cluster_k/k_8/clustered_scenes_k8_dish_cluster_mapping.csv",
    10: REPO / "sweep_cluster_k/k_10/clustered_scenes_k10_dish_cluster_mapping.csv",
}


def fast_ks(sub_sorted: np.ndarray, pop_sorted: np.ndarray) -> float:
    """Two-sample KS sup-distance, identical to scipy ks_2samp statistic."""
    allv = np.concatenate([sub_sorted, pop_sorted])
    cdf_s = np.searchsorted(sub_sorted, allv, side="right") / sub_sorted.size
    cdf_p = np.searchsorted(pop_sorted, allv, side="right") / pop_sorted.size
    return float(np.abs(cdf_s - cdf_p).max())


def cluster_labels(k: int, seed: int) -> pd.DataFrame:
    feats = pd.read_csv(FEATS)
    cols = [c for c in feats.columns if c != "dish_id" and c not in PROXIES]
    X = feats[cols].fillna(feats[cols].mean())
    var = X.var(axis=0)
    X = X.drop(columns=var[var == 0.0].index.tolist())
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(Xs)
    return pd.DataFrame({"dish_id": feats["dish_id"].astype(str), "cluster": labels})


class Sweeper:
    """Balanced-generator Zip-NeRF audit with the fast KS evaluator."""

    def __init__(self, merged_df: pd.DataFrame):
        self.df = merged_df.reset_index(drop=True)
        self.vals = {
            m: self.df[m].to_numpy(dtype=float) for m in ("psnr", "ssim", "lpips")
        }
        self.pop_sorted = {m: np.sort(v) for m, v in self.vals.items()}
        self.pop_mean = {m: float(v.mean()) for m, v in self.vals.items()}
        self.groups = {
            int(c): self.df.index[self.df["cluster"] == int(c)].to_numpy(dtype=int)
            for c in sorted(self.df["cluster"].unique())
        }
        self.ids = self.df["dish_id"].astype(str).to_numpy()

    def run_budget(self, b: int, audit_seed: int, tols=DEFAULT_TOLS, keep_best=False):
        rng = np.random.default_rng(audit_seed + b)
        n_pass = 0
        best = None
        for trial in range(M):
            picks = [rng.choice(g, size=b, replace=False) for _, g in sorted(self.groups.items())]
            idx = np.sort(np.concatenate(picks))
            gaps, kss = {}, {}
            for m in ("psnr", "ssim", "lpips"):
                sv = self.vals[m][idx]
                gaps[m] = abs(float(sv.mean()) - self.pop_mean[m])
                kss[m] = fast_ks(np.sort(sv), self.pop_sorted[m])
            mean_obj = max(gaps[m] / tols[m] for m in ("psnr", "ssim", "lpips"))
            joint_obj = max(mean_obj, max(kss.values()) / tols["ks"])
            ok = mean_obj <= 1.0 and max(kss.values()) <= tols["ks"]
            n_pass += int(ok)
            if keep_best:
                key = (joint_obj, mean_obj, trial)
                if best is None or key < best[0]:
                    best = (key, set(self.ids[idx].tolist()))
        lcb = _wilson_lower_bound(n_pass / M, M, 0.95)
        return n_pass, lcb, (best[1] if keep_best else None)

    def recommended_budget(self, budgets_b: list[int], audit_seed: int):
        k = len(self.groups)
        frontier = []
        rec = None
        for b in budgets_b:
            n_pass, lcb, _ = self.run_budget(b, audit_seed)
            frontier.append(
                {"b": b, "budget_scenes": k * b, "n_pass": n_pass,
                 "empirical_pass_rate": n_pass / M, "wilson_lcb_95": lcb}
            )
            if rec is None and lcb >= P_MIN:
                rec = k * b
        return frontier, rec


def budgets_for_k(k: int) -> list[int]:
    bs = []
    for size in STANDARD_SIZES:
        b = int(np.floor(size / k + 0.5))  # round half up
        if b >= 1 and b not in bs:
            bs.append(b)
    return bs


def main() -> None:
    # Population.
    zip_df = _load_zipnerf_metrics(ZIPNERF_LOG)

    # ================================================================ c
    print("== P3c: CALIBRATION ==")
    N = 3522
    ks_critical = {
        str(n): round(1.358 * np.sqrt((n + N) / (n * N)), 4) for n in (36, 48, 96, 120)
    }
    print("  KS critical (alpha=0.05):", ks_critical)

    k6_map = _load_mapping_csv(SHIPPED_MAPPINGS[6])
    full_k6 = _merge_mapping_and_metrics(k6_map, zip_df)
    metric_stds = {m: float(full_k6[m].std(ddof=1)) for m in ("psnr", "ssim", "lpips")}
    tolerance_fractions = {
        m: DEFAULT_TOLS[m] / metric_stds[m] for m in ("psnr", "ssim", "lpips")
    }
    print("  stds:", {k: round(v, 4) for k, v in metric_stds.items()})
    print("  tolerance/std:", {k: round(v, 4) for k, v in tolerance_fractions.items()})

    _, pops, _ = load_populations()  # E1 populations on 3473 intersection
    means = {
        name: {m: float(df[m].mean()) for m in df.columns if m != "dish_id"}
        for name, df in pops.items()
    }
    gaps_by_metric = {"psnr": {}, "ssim": {}, "lpips": {}}
    for a, b_ in itertools.combinations(pops.keys(), 2):
        for m in ("psnr", "ssim", "lpips"):
            if m in means[a] and m in means[b_]:
                gaps_by_metric[m][f"{a}_vs_{b_}"] = abs(means[a][m] - means[b_][m])
    min_method_gaps = {m: min(v.values()) for m, v in gaps_by_metric.items()}
    tolerances_below = {
        m: DEFAULT_TOLS[m] < min_method_gaps[m] for m in ("psnr", "ssim", "lpips")
    }
    all_below = all(tolerances_below.values())
    for m in ("psnr", "ssim", "lpips"):
        flag = "OK" if tolerances_below[m] else "** TOLERANCE NOT BELOW GAP **"
        print(f"  min |gap| {m}: {min_method_gaps[m]:.4f} vs tol {DEFAULT_TOLS[m]} -> {flag}")

    part_c = {
        "ks_critical": ks_critical,
        "ks_critical_formula": "1.358*sqrt((n+N)/(n*N)), N=3522, alpha=0.05",
        "metric_stds": metric_stds,
        "metric_stds_population_n": int(len(full_k6)),
        "tolerance_fractions": tolerance_fractions,
        "cross_method_gaps_all": gaps_by_metric,
        "min_method_gaps": min_method_gaps,
        "tolerances_below_gaps": bool(all_below),
        "tolerances_below_gaps_per_metric": tolerances_below,
    }

    # ================================================================ a
    print("\n== P3a: 4x4 TOLERANCE SWEEP (reusing stored draws) ==")
    trials = pd.read_csv(OUT_DIR / "full_budget_sweep_trials.csv")
    tau_psnr_grid = [0.25, 0.375, 0.5, 0.75]
    tau_ks_grid = [0.10, 0.12, 0.14, 0.18]
    table = {}
    for tp in tau_psnr_grid:
        ts = 0.02 * tp
        row = {}
        for tk in tau_ks_grid:
            ok = (
                (trials["abs_psnr_gap"] <= tp)
                & (trials["abs_ssim_gap"] <= ts)
                & (trials["abs_lpips_gap"] <= ts)
                & (trials["ks_psnr"] <= tk)
                & (trials["ks_ssim"] <= tk)
                & (trials["ks_lpips"] <= tk)
            )
            rec = None
            for size in sorted(trials["budget_scenes"].unique()):
                sub = ok[trials["budget_scenes"] == size]
                lcb = _wilson_lower_bound(sub.mean(), len(sub), 0.95)
                if lcb >= P_MIN:
                    rec = int(size)
                    break
            row[str(tk)] = rec if rec is not None else "not reached up to 120"
        table[str(tp)] = row
        print(f"  tau_psnr={tp}: {row}")
    if table["0.5"]["0.14"] != 36:
        raise SystemExit("VALIDATION FAILED: default cell should recommend 36")
    print("  [validation] default cell (0.5, 0.14) -> 36 as in full_budget_sweep")

    part_a = {
        "tau_psnr_grid": tau_psnr_grid,
        "tau_ks_grid": tau_ks_grid,
        "coupling": "tau_ssim = tau_lpips = 0.02 * tau_psnr",
        "budget_table_4x4": table,
        "note": (
            "Candidate draws identical across cells (reused per-trial stats "
            "of full_budget_sweep, seed 0, M=400, b in {4..20}); only the "
            "pass indicator is recomputed."
        ),
    }

    # ================================================================ b
    print("\n== P3b: REGIME COUNT AND SEEDS ==")
    # Validate fast evaluator against reference values.
    sweeper6 = Sweeper(full_k6)
    n48, _, _ = sweeper6.run_budget(8, 0)
    n96, _, _ = sweeper6.run_budget(16, 0)
    if (n48, n96) != (88, 113):
        raise SystemExit(f"VALIDATION FAILED: fast evaluator gives {n48}/{n96}, want 88/113")
    print("  [validation] fast evaluator reproduces 88/400 (b=8) and 113/400 (b=16)")

    # k grid.
    budget_per_k = {}
    mapping_ari = {}
    for k in (4, 5, 6, 8, 10):
        if k in SHIPPED_MAPPINGS:
            mapping = _load_mapping_csv(SHIPPED_MAPPINGS[k])
            source = "shipped"
            rep = cluster_labels(k, 0)
            joined = mapping.merge(rep, on="dish_id", suffixes=("_ship", "_rep"))
            mapping_ari[k] = float(
                adjusted_rand_score(joined["cluster_ship"], joined["cluster_rep"])
            )
        else:
            mapping = cluster_labels(k, 0)
            source = "replicated (seed 0)"
            mapping_ari[k] = None
        merged = _merge_mapping_and_metrics(mapping, zip_df)
        sw = Sweeper(merged)
        min_regime = min(len(g) for g in sw.groups.values())
        bs = [b for b in budgets_for_k(k) if b <= min_regime]
        frontier, rec = sw.recommended_budget(bs, audit_seed=0)
        budget_per_k[str(k)] = {
            "mapping_source": source,
            "ari_replication_vs_shipped": mapping_ari.get(k),
            "budgets_scenes": [k * b for b in bs],
            "min_regime_size": min_regime,
            "frontier": frontier,
            "recommended_budget_scenes": rec if rec is not None else "not reached",
        }
        print(f"  k={k:2d} ({source}, ARI={mapping_ari.get(k)}): sizes {[k*b for b in bs]}"
              f" -> recommended {rec}")

    # 5x5 seed grid at k=6.
    print("  seed grid 5x5 at k=6 ...")
    seed_grid = {}
    rec_list = []
    for km_seed in range(5):
        if km_seed == 0:
            mapping = _load_mapping_csv(SHIPPED_MAPPINGS[6])
        else:
            mapping = cluster_labels(6, km_seed)
        sw = Sweeper(_merge_mapping_and_metrics(mapping, zip_df))
        for audit_seed in range(5):
            _, rec = sw.recommended_budget(budgets_for_k(6), audit_seed)
            key = f"kmeans{km_seed}_audit{audit_seed}"
            seed_grid[key] = rec if rec is not None else "not reached"
            rec_list.append(seed_grid[key])
            print(f"    {key}: recommended {seed_grid[key]}")
    dist = {}
    for r in rec_list:
        dist[str(r)] = dist.get(str(r), 0) + 1

    # Jaccard of exported 48-scene subsets across audit seeds at kmeans seed 0.
    sw0 = Sweeper(full_k6)
    best_subsets = []
    for audit_seed in range(5):
        _, _, best = sw0.run_budget(8, audit_seed, keep_best=True)
        best_subsets.append(best)
    jac = [
        len(a & b) / len(a | b)
        for a, b in itertools.combinations(best_subsets, 2)
    ]
    jaccard_mean = float(np.mean(jac))
    print(f"  mean pairwise Jaccard of best 48-scene subsets (audit seeds 0-4): "
          f"{jaccard_mean:.4f} (pairs: {[round(j,3) for j in jac]})")

    part_b = {
        "clustering_replication": (
            "features = 57 normalized descriptors minus budget proxies "
            f"{PROXIES}, fillna(mean), drop zero-variance, StandardScaler, "
            "KMeans(random_state=seed, n_init='auto'), scikit-learn 1.4.2; "
            "reproduces the shipped k=6 seed-0 labels exactly (ARI=1.0)."
        ),
        "budget_per_k": budget_per_k,
        "seed_grid_recommended_budgets": seed_grid,
        "seed_budget_distribution": dist,
        "jaccard_overlap": {
            "mean_pairwise": jaccard_mean,
            "pairs": jac,
            "definition": (
                "Best joint-objective 48-scene candidate at b=8 per audit "
                "seed (0-4), K-Means seed 0; 10 pairwise Jaccard values."
            ),
        },
        "audit_seed_rule": "rng = default_rng(audit_seed + b) per budget",
    }

    e3 = {
        "description": (
            "Sensitivity and calibration: (c) KS critical distances, "
            "tolerance/std fractions, min cross-method gaps; (a) 4x4 "
            "tau_PSNR x tau_KS tolerance sweep with shared candidate draws; "
            "(b) regime-count grid k in {4,5,6,8,10} and 5x5 K-Means x audit "
            "seed grid at k=6 with Jaccard overlap of exported subsets. "
            "Zip-NeRF event on the full audit population (3521 scenes)."
        ),
        "c": part_c,
        "a": part_a,
        "b": part_b,
    }
    merge_results_json(OUT_DIR / "rebuttal_results.json", {"E3": e3})
    print(f"\nWrote E3 key into {OUT_DIR / 'rebuttal_results.json'}")


if __name__ == "__main__":
    main()
