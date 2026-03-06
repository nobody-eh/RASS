#!/usr/bin/env python3
"""Joint sweep over clustering k and subset budget."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sweep_subset_budgets as sb


LOGGER = logging.getLogger("sweep_k_budget_selection")


def _find_mapping_csv(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("*_dish_cluster_mapping.csv"))
    if not matches:
        raise FileNotFoundError(f"No dish_cluster_mapping CSV found in {run_dir}")
    return matches[0]


def _to_jsonable(v: object) -> object:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        return float(v)
    return v


def _compute_pareto_mask(df: pd.DataFrame) -> np.ndarray:
    # maximize joint_pass_rate_lcb, minimize total_subset_size
    vals = df[["joint_pass_rate_lcb", "total_subset_size"]].to_numpy(dtype=float)
    out = np.ones(vals.shape[0], dtype=bool)
    for i in range(vals.shape[0]):
        if not out[i]:
            continue
        p_i = vals[i]
        dominates_i = (
            (vals[:, 0] >= p_i[0])
            & (vals[:, 1] <= p_i[1])
            & ((vals[:, 0] > p_i[0]) | (vals[:, 1] < p_i[1]))
        )
        if np.any(dominates_i):
            out[i] = False
    return out


def _select_joint(
    df: pd.DataFrame,
    target_joint_pass_rate: float,
    use_lcb: bool,
    max_total_subset: Optional[int],
) -> Dict[str, object]:
    metric_col = "joint_pass_rate_lcb" if use_lcb else "joint_pass_rate"
    cand = df[(df["feasible"] == True)].copy()  # noqa: E712
    cand = cand[(cand["best_global_pass"] == True) & (cand["best_per_cluster_pass"] == True)]  # noqa: E712
    if max_total_subset is not None:
        c2 = cand[cand["total_subset_size"] <= int(max_total_subset)]
        if not c2.empty:
            cand = c2
    if cand.empty:
        cand = df[df["feasible"] == True].copy()  # noqa: E712

    meets = cand[cand[metric_col] >= float(target_joint_pass_rate)]
    if not meets.empty:
        selected = (
            meets.sort_values(
                ["total_subset_size", metric_col, "silhouette", "cluster_balance_min_over_max"],
                ascending=[True, False, False, False],
                kind="mergesort",
            )
            .iloc[0]
            .to_dict()
        )
        mode = "min_size_meeting_target"
        target_met = True
    else:
        selected = (
            cand.sort_values(
                [metric_col, "total_subset_size", "silhouette", "cluster_balance_min_over_max"],
                ascending=[False, True, False, False],
                kind="mergesort",
            )
            .iloc[0]
            .to_dict()
        )
        mode = "max_metric_fallback"
        target_met = False

    rec = {
        "selection_mode": mode,
        "selection_metric": metric_col,
        "target_joint_pass_rate": float(target_joint_pass_rate),
        "target_met": bool(target_met),
        "recommended_k": int(selected["k"]),
        "recommended_budget_per_cluster": int(selected["budget_per_cluster"]),
        "recommended_total_subset_size": int(selected["total_subset_size"]),
        "recommended_joint_pass_rate": float(selected["joint_pass_rate"]),
        "recommended_joint_pass_rate_lcb": float(selected["joint_pass_rate_lcb"]),
        "recommended_manifest": str(selected["best_manifest"]),
    }
    return rec


def _write_markdown(path: Path, rows: List[Dict[str, object]], rec: Dict[str, object]) -> None:
    lines = [
        "# Joint k x Budget Sweep",
        "",
        f"- Recommended k: **{rec['recommended_k']}**",
        f"- Recommended budget/cluster: **{rec['recommended_budget_per_cluster']}**",
        f"- Recommended total subset: **{rec['recommended_total_subset_size']}**",
        f"- Recommended joint pass rate: **{rec['recommended_joint_pass_rate']:.4f}**",
        f"- Recommended joint pass LCB: **{rec['recommended_joint_pass_rate_lcb']:.4f}**",
        f"- Selection mode: **{rec['selection_mode']}**",
        "",
        "| rank | pareto | k | budget | total_subset | joint_pass | joint_pass_lcb | silhouette | balance | best_manifest |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {rank} | {pareto_front} | {k} | {budget_per_cluster} | {total_subset_size} | "
            "{joint_pass_rate:.4f} | {joint_pass_rate_lcb:.4f} | {silhouette:.4f} | "
            "{cluster_balance_min_over_max:.4f} | {best_manifest} |".format(**r)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint k x budget sweep selector.")
    parser.add_argument(
        "--k-sweep-summary-csv",
        type=Path,
        default=Path("sweep_cluster_k/k_sweep_summary.csv"),
    )
    parser.add_argument("--oc-csv", type=Path, default=Path("ingp_oc.csv"))
    parser.add_argument("--fi-csv", type=Path, default=Path("ingp_fi.csv"))
    parser.add_argument("--runtime-csv", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["all", "selection", "tune", "test"],
        default="all",
    )
    parser.add_argument("--budgets", type=str, default="2,4,6,8,10")
    parser.add_argument("--num-simulations", type=int, default=1500)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-abs-psnr-gap", type=float, default=0.5)
    parser.add_argument("--max-abs-ssim-gap", type=float, default=0.01)
    parser.add_argument("--max-per-cluster-psnr-gap", type=float, default=None)
    parser.add_argument("--max-per-cluster-ssim-gap", type=float, default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--target-joint-pass-rate", type=float, default=0.10)
    parser.add_argument("--selection-use-lcb", action="store_true", default=True)
    parser.add_argument(
        "--selection-use-mean",
        dest="selection_use_lcb",
        action="store_false",
    )
    parser.add_argument("--max-total-subset", type=int, default=None)
    parser.add_argument("--weight-runtime", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("sweep_cluster_k/k_budget_sweep"))
    parser.add_argument("--summary-csv-name", type=str, default="k_budget_sweep_summary.csv")
    parser.add_argument("--summary-json-name", type=str, default="k_budget_sweep_summary.json")
    parser.add_argument("--summary-md-name", type=str, default="k_budget_sweep_summary.md")
    parser.add_argument("--recommendation-json-name", type=str, default="k_budget_recommendation.json")
    parser.add_argument("--recommended-manifest-name", type=str, default="recommended_subset.csv")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    budgets = sb._parse_budgets(args.budgets)  # intentional reuse
    ks = pd.read_csv(args.k_sweep_summary_csv)
    if "k" not in ks.columns or "run_dir" not in ks.columns:
        raise ValueError("k sweep summary CSV must contain k and run_dir columns")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for i, r in ks.sort_values("k", kind="mergesort").iterrows():
        k = int(r["k"])
        run_dir = Path(str(r["run_dir"]))
        mapping_csv = _find_mapping_csv(run_dir)

        df, groups, _, runtime_available = sb._prepare_combined(
            mapping_csv=mapping_csv,
            oc_csv=args.oc_csv,
            fi_csv=args.fi_csv,
            runtime_csv=args.runtime_csv,
            split_csv=args.split_csv,
            eval_split=args.eval_split,
        )

        for j, b in enumerate(budgets):
            res = sb._simulate_budget(
                df=df,
                groups=groups,
                budget_per_cluster=int(b),
                num_simulations=int(args.num_simulations),
                seed=int(args.random_seed) + (i * 1009) + (j * 97),
                max_abs_psnr_gap=float(args.max_abs_psnr_gap),
                max_abs_ssim_gap=float(args.max_abs_ssim_gap),
                confidence_level=float(args.confidence_level),
                max_per_cluster_psnr_gap=args.max_per_cluster_psnr_gap,
                max_per_cluster_ssim_gap=args.max_per_cluster_ssim_gap,
                weight_runtime=float(args.weight_runtime),
            )
            if not bool(res["feasible"]):
                row = {
                    "k": int(k),
                    "budget_per_cluster": int(b),
                    "total_subset_size": int(b * len(groups)),
                    "feasible": False,
                    "reason": str(res.get("reason", "")),
                    "joint_pass_rate": 0.0,
                    "joint_pass_rate_lcb": 0.0,
                    "best_global_pass": False,
                    "best_per_cluster_pass": False,
                    "exp_abs_oc_psnr": np.nan,
                    "exp_abs_oc_ssim": np.nan,
                    "exp_abs_fi_psnr": np.nan,
                    "exp_abs_fi_ssim": np.nan,
                    "best_manifest": "",
                    "runtime_available": runtime_available,
                    "silhouette": float(r.get("silhouette", np.nan)),
                    "cluster_balance_min_over_max": float(r.get("cluster_balance_min_over_max", np.nan)),
                }
                rows.append(row)
                continue

            manifest_path = manifests_dir / f"subset_k{k}_budget_{int(b)}.csv"
            best_subset_df = res.pop("best_subset_df")
            assert isinstance(best_subset_df, pd.DataFrame)
            best_subset_df.to_csv(manifest_path, index=False)

            row = {k1: v for k1, v in res.items() if k1 != "best_subset_df"}
            row["k"] = int(k)
            row["runtime_available"] = bool(runtime_available)
            row["best_manifest"] = str(manifest_path)
            row["silhouette"] = float(r.get("silhouette", np.nan))
            row["cluster_balance_min_over_max"] = float(r.get("cluster_balance_min_over_max", np.nan))
            rows.append(row)

    if not rows:
        raise RuntimeError("No joint sweep rows produced")

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["joint_pass_rate_lcb", "total_subset_size", "silhouette"],
        ascending=[False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["pareto_front"] = _compute_pareto_mask(summary)
    summary["rank"] = np.arange(1, summary.shape[0] + 1)

    rec = _select_joint(
        summary,
        target_joint_pass_rate=float(args.target_joint_pass_rate),
        use_lcb=bool(args.selection_use_lcb),
        max_total_subset=args.max_total_subset,
    )

    # Mark recommended row.
    summary["recommended"] = (
        (pd.to_numeric(summary["k"], errors="coerce").astype(int) == int(rec["recommended_k"]))
        & (
            pd.to_numeric(summary["budget_per_cluster"], errors="coerce").astype(int)
            == int(rec["recommended_budget_per_cluster"])
        )
    )

    recommended_src = Path(str(rec["recommended_manifest"]))
    recommended_dst = out_dir / args.recommended_manifest_name
    recommended_dst.write_text(recommended_src.read_text(encoding="utf-8"), encoding="utf-8")
    rec["exported_manifest"] = str(recommended_dst)

    summary_csv = out_dir / args.summary_csv_name
    summary_json = out_dir / args.summary_json_name
    summary_md = out_dir / args.summary_md_name
    rec_json = out_dir / args.recommendation_json_name

    summary.to_csv(summary_csv, index=False)
    payload = {
        "config": {
            "k_sweep_summary_csv": str(args.k_sweep_summary_csv),
            "budgets": budgets,
            "num_simulations": int(args.num_simulations),
            "random_seed": int(args.random_seed),
            "max_abs_psnr_gap": float(args.max_abs_psnr_gap),
            "max_abs_ssim_gap": float(args.max_abs_ssim_gap),
            "max_per_cluster_psnr_gap": args.max_per_cluster_psnr_gap,
            "max_per_cluster_ssim_gap": args.max_per_cluster_ssim_gap,
            "confidence_level": float(args.confidence_level),
            "target_joint_pass_rate": float(args.target_joint_pass_rate),
            "selection_use_lcb": bool(args.selection_use_lcb),
            "max_total_subset": args.max_total_subset,
            "weight_runtime": float(args.weight_runtime),
            "split_csv": None if args.split_csv is None else str(args.split_csv),
            "eval_split": str(args.eval_split),
        },
        "recommendation": {k: _to_jsonable(v) for k, v in rec.items()},
        "rows": [
            {k: _to_jsonable(v) for k, v in row.items()}
            for row in summary.to_dict(orient="records")
        ],
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    with rec_json.open("w", encoding="utf-8") as f:
        json.dump({k: _to_jsonable(v) for k, v in rec.items()}, f, indent=2, sort_keys=False)

    _write_markdown(summary_md, summary.to_dict(orient="records"), rec)

    LOGGER.info("Saved summary CSV: %s", summary_csv)
    LOGGER.info("Saved summary JSON: %s", summary_json)
    LOGGER.info("Saved recommendation JSON: %s", rec_json)
    LOGGER.info("Saved summary Markdown: %s", summary_md)
    LOGGER.info(
        "Recommended k=%d | budget=%d | total=%d | joint_lcb=%.4f",
        rec["recommended_k"],
        rec["recommended_budget_per_cluster"],
        rec["recommended_total_subset_size"],
        rec["recommended_joint_pass_rate_lcb"],
    )


if __name__ == "__main__":
    main()
