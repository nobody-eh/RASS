#!/usr/bin/env python3
"""Create stratified holdout split (selection/tune/test) from cluster mapping."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("create_holdout_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified holdout split CSV.")
    parser.add_argument("--cluster-mapping-csv", type=Path, required=True)
    parser.add_argument("--selection-frac", type=float, default=0.6)
    parser.add_argument("--tune-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("subset_holdout_split.csv"))
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    total_frac = float(args.selection_frac + args.tune_frac + args.test_frac)
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError("selection_frac + tune_frac + test_frac must sum to 1.0")

    df = pd.read_csv(args.cluster_mapping_csv)
    if "dish_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError("cluster mapping CSV must contain dish_id and cluster")
    df = df[["dish_id", "cluster"]].copy()
    df["dish_id"] = df["dish_id"].astype(str)
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df = df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    rng = np.random.default_rng(int(args.random_seed))
    rows = []
    for cluster, sub in df.groupby("cluster", dropna=False):
        ids = sub["dish_id"].astype(str).to_numpy()
        perm = rng.permutation(ids.shape[0])
        ids = ids[perm]

        n = ids.shape[0]
        n_sel = int(round(float(args.selection_frac) * n))
        n_tune = int(round(float(args.tune_frac) * n))
        if n_sel + n_tune > n:
            n_tune = max(0, n - n_sel)
        n_test = n - n_sel - n_tune

        split_labels = (
            ["selection"] * n_sel + ["tune"] * n_tune + ["test"] * n_test
        )
        for dish_id, split in zip(ids.tolist(), split_labels):
            rows.append({"dish_id": dish_id, "cluster": int(cluster), "split": split})

    out = pd.DataFrame(rows).sort_values(["cluster", "dish_id"], kind="mergesort").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    LOGGER.info("Saved split CSV: %s", args.output_csv)
    LOGGER.info("Rows: %d", out.shape[0])
    LOGGER.info("Split counts: %s", out["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()

