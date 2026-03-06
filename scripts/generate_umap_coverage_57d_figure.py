#!/usr/bin/env python3
"""Generate UMAP coverage figure for 57-D descriptors."""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate figures/fig_umap_coverage_57d.pdf from descriptor, cluster, "
            "and subset-manifest CSV files."
        )
    )
    parser.add_argument("--feats_csv", type=Path, default=Path("feats_normalized.csv"))
    parser.add_argument(
        "--clusters_csv",
        type=Path,
        default=Path("sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv"),
    )
    parser.add_argument(
        "--subset_csv",
        type=Path,
        default=Path("sweep_cluster_k/budget_sweep_k6_auto_v2/recommended_subset.csv"),
    )
    parser.add_argument(
        "--out_pdf", type=Path, default=Path("figures/fig_umap_coverage_57d.pdf")
    )
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=0)
    return parser.parse_args()


def _find_id_col(df: pd.DataFrame) -> str:
    for c in ("dish_id", "scene_id", "id"):
        if c in df.columns:
            return c
    raise ValueError("Could not find scene id column (expected one of: dish_id, scene_id, id).")


def _find_cluster_col(df: pd.DataFrame) -> str:
    for c in ("cluster", "label", "cluster_id"):
        if c in df.columns:
            return c
    raise ValueError("Could not find cluster label column (expected cluster/label/cluster_id).")


def _compute_embedding_umap_or_pca(
    x: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[np.ndarray, str]:
    umap_mod = None
    try:
        import umap  # type: ignore

        umap_mod = umap
    except Exception:
        # Try installing umap-learn; if this fails, fallback to PCA.
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "umap-learn"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import umap  # type: ignore

            umap_mod = umap
            warnings.warn("Installed umap-learn at runtime.", stacklevel=2)
        except Exception:
            warnings.warn(
                "UMAP is unavailable and could not be installed; falling back to PCA.",
                stacklevel=2,
            )

    if umap_mod is not None:
        reducer = umap_mod.UMAP(
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            random_state=int(random_state),
        )
        emb = reducer.fit_transform(x)
        return np.asarray(emb, dtype=float), "UMAP"

    # Deterministic PCA fallback via SVD.
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    emb = u[:, :2] * s[:2]
    return np.asarray(emb, dtype=float), "PCA"


def main() -> None:
    args = parse_args()

    feats = pd.read_csv(args.feats_csv)
    clusters = pd.read_csv(args.clusters_csv)
    subset = pd.read_csv(args.subset_csv)

    feats_id = _find_id_col(feats)
    clusters_id = _find_id_col(clusters)
    subset_id = _find_id_col(subset)
    cluster_col = _find_cluster_col(clusters)

    feats = feats.copy()
    clusters = clusters.copy()
    subset = subset.copy()
    feats[feats_id] = feats[feats_id].astype(str)
    clusters[clusters_id] = clusters[clusters_id].astype(str)
    subset[subset_id] = subset[subset_id].astype(str)

    merged = feats.merge(
        clusters[[clusters_id, cluster_col]],
        left_on=feats_id,
        right_on=clusters_id,
        how="inner",
    )
    if clusters_id != feats_id:
        merged = merged.drop(columns=[clusters_id])

    merged = merged.drop_duplicates(subset=[feats_id], keep="first").reset_index(drop=True)
    merged[cluster_col] = pd.to_numeric(merged[cluster_col], errors="coerce")
    merged = merged.dropna(subset=[cluster_col]).copy()
    merged[cluster_col] = merged[cluster_col].astype(int)

    subset_ids = set(subset[subset_id].astype(str))
    merged["is_subset"] = merged[feats_id].isin(subset_ids)

    numeric_cols = [
        c
        for c in merged.columns
        if c not in {feats_id, cluster_col, "is_subset"} and np.issubdtype(merged[c].dtype, np.number)
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found after merge.")

    x = merged[numeric_cols].to_numpy(dtype=float)
    emb, method = _compute_embedding_umap_or_pca(
        x=x,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )

    merged["emb_x"] = emb[:, 0]
    merged["emb_y"] = emb[:, 1]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for c in sorted(merged[cluster_col].unique().tolist()):
        m = merged[cluster_col] == c
        ax.scatter(
            merged.loc[m, "emb_x"],
            merged.loc[m, "emb_y"],
            s=14,
            alpha=0.75,
            label=f"Cluster {c}",
        )

    m_sub = merged["is_subset"]
    ax.scatter(
        merged.loc[m_sub, "emb_x"],
        merged.loc[m_sub, "emb_y"],
        s=72,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="Selected subset",
        zorder=4,
    )

    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.set_title("57-D Descriptor Coverage")
    ax.grid(alpha=0.25)
    ax.legend()

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_pdf, format="pdf")
    plt.close(fig)

    print(
        "Saved: "
        f"{args.out_pdf} | method={method} | "
        f"scenes={len(merged)} | subset_overlap={int(m_sub.sum())}"
    )


if __name__ == "__main__":
    main()
