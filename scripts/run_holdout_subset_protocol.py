#!/usr/bin/env python3
"""Run holdout protocol for subset selection and validation.

Steps:
1) Create stratified split (if not provided)
2) Joint k x budget sweep on selection split
3) Validate recommended manifest on tune split
4) Validate recommended manifest on test split
5) Write consolidated holdout report
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


LOGGER = logging.getLogger("run_holdout_subset_protocol")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: List[str], cwd: Path) -> None:
    LOGGER.info("Running: %s", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run holdout subset selection protocol.")
    parser.add_argument(
        "--cluster-mapping-csv",
        type=Path,
        required=True,
        help="Dish->cluster mapping CSV for selected base k run.",
    )
    parser.add_argument(
        "--k-sweep-summary-csv",
        type=Path,
        default=Path("sweep_cluster_k/k_sweep_summary.csv"),
    )
    parser.add_argument("--oc-csv", type=Path, default=Path("ingp_oc.csv"))
    parser.add_argument("--fi-csv", type=Path, default=Path("ingp_fi.csv"))
    parser.add_argument("--runtime-csv", type=Path, default=None)

    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--selection-frac", type=float, default=0.6)
    parser.add_argument("--tune-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)

    parser.add_argument("--budgets", type=str, default="2,4,6,8,10,12")
    parser.add_argument("--num-simulations", type=int, default=1500)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-abs-psnr-gap", type=float, default=0.5)
    parser.add_argument("--max-abs-ssim-gap", type=float, default=0.01)
    parser.add_argument("--max-per-cluster-psnr-gap", type=float, default=None)
    parser.add_argument("--max-per-cluster-ssim-gap", type=float, default=None)
    parser.add_argument("--target-joint-pass-rate", type=float, default=0.10)
    parser.add_argument("--max-total-subset", type=int, default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--weight-runtime", type=float, default=0.0)
    parser.add_argument("--selection-use-mean", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sweep_cluster_k/holdout_protocol"),
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root = _repo_root()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_csv = args.split_csv or (out_dir / "holdout_split.csv")
    if not split_csv.exists():
        _run(
            [
                sys.executable,
                str(root / "scripts" / "create_holdout_split.py"),
                "--cluster-mapping-csv",
                str(args.cluster_mapping_csv),
                "--selection-frac",
                str(args.selection_frac),
                "--tune-frac",
                str(args.tune_frac),
                "--test-frac",
                str(args.test_frac),
                "--random-seed",
                str(args.random_seed),
                "--output-csv",
                str(split_csv),
                "--log-level",
                args.log_level,
            ],
            cwd=root,
        )
    else:
        LOGGER.info("Using existing split CSV: %s", split_csv)

    joint_out = out_dir / "joint_selection"
    sweep_cmd = [
        sys.executable,
        str(root / "scripts" / "sweep_k_budget_selection.py"),
        "--k-sweep-summary-csv",
        str(args.k_sweep_summary_csv),
        "--oc-csv",
        str(args.oc_csv),
        "--fi-csv",
        str(args.fi_csv),
        "--budgets",
        args.budgets,
        "--num-simulations",
        str(args.num_simulations),
        "--random-seed",
        str(args.random_seed),
        "--max-abs-psnr-gap",
        str(args.max_abs_psnr_gap),
        "--max-abs-ssim-gap",
        str(args.max_abs_ssim_gap),
        "--target-joint-pass-rate",
        str(args.target_joint_pass_rate),
        "--confidence-level",
        str(args.confidence_level),
        "--split-csv",
        str(split_csv),
        "--eval-split",
        "selection",
        "--weight-runtime",
        str(args.weight_runtime),
        "--output-dir",
        str(joint_out),
        "--log-level",
        args.log_level,
    ]
    if args.runtime_csv is not None:
        sweep_cmd += ["--runtime-csv", str(args.runtime_csv)]
    if args.max_total_subset is not None:
        sweep_cmd += ["--max-total-subset", str(args.max_total_subset)]
    if args.max_per_cluster_psnr_gap is not None:
        sweep_cmd += ["--max-per-cluster-psnr-gap", str(args.max_per_cluster_psnr_gap)]
    if args.max_per_cluster_ssim_gap is not None:
        sweep_cmd += ["--max-per-cluster-ssim-gap", str(args.max_per_cluster_ssim_gap)]
    if args.selection_use_mean:
        sweep_cmd += ["--selection-use-mean"]
    _run(sweep_cmd, cwd=root)

    recommendation = _load_json(joint_out / "k_budget_recommendation.json")
    rec_k = int(recommendation["recommended_k"])
    rec_budget = int(recommendation["recommended_budget_per_cluster"])
    rec_run_dir = Path(f"sweep_cluster_k/k_{rec_k}")
    mapping_csv = sorted(rec_run_dir.glob("*_dish_cluster_mapping.csv"))
    if not mapping_csv:
        raise FileNotFoundError(f"Could not find mapping CSV for recommended k={rec_k} in {rec_run_dir}")
    mapping_csv = mapping_csv[0]

    validations: Dict[str, Dict[str, object]] = {}
    split_recommendations: Dict[str, Dict[str, object]] = {}
    for split_name in ("tune", "test"):
        split_budget_dir = out_dir / f"{split_name}_budget_eval"
        cmd_sweep = [
            sys.executable,
            str(root / "scripts" / "sweep_subset_budgets.py"),
            "--cluster-mapping-csv",
            str(mapping_csv),
            "--oc-csv",
            str(args.oc_csv),
            "--fi-csv",
            str(args.fi_csv),
            "--budgets",
            str(rec_budget),
            "--num-simulations",
            str(args.num_simulations),
            "--random-seed",
            str(args.random_seed),
            "--max-abs-psnr-gap",
            str(args.max_abs_psnr_gap),
            "--max-abs-ssim-gap",
            str(args.max_abs_ssim_gap),
            "--target-joint-pass-rate",
            str(args.target_joint_pass_rate),
            "--confidence-level",
            str(args.confidence_level),
            "--split-csv",
            str(split_csv),
            "--eval-split",
            split_name,
            "--output-dir",
            str(split_budget_dir),
            "--log-level",
            args.log_level,
        ]
        if args.runtime_csv is not None:
            cmd_sweep += ["--runtime-csv", str(args.runtime_csv)]
        if args.max_per_cluster_psnr_gap is not None:
            cmd_sweep += ["--max-per-cluster-psnr-gap", str(args.max_per_cluster_psnr_gap)]
        if args.max_per_cluster_ssim_gap is not None:
            cmd_sweep += ["--max-per-cluster-ssim-gap", str(args.max_per_cluster_ssim_gap)]
        if args.max_total_subset is not None:
            cmd_sweep += ["--max-total-subset", str(args.max_total_subset)]
        if args.selection_use_mean:
            cmd_sweep += ["--selection-use-mean"]
        _run(cmd_sweep, cwd=root)

        split_rec = _load_json(split_budget_dir / "budget_recommendation.json")
        split_recommendations[split_name] = split_rec
        split_manifest = Path(str(split_rec["exported_manifest"]))

        split_out_prefix = out_dir / f"recommended_{split_name}_validation"
        cmd = [
            sys.executable,
            str(root / "scripts" / "validate_subset_metric_matching.py"),
            "--subset-manifest",
            str(split_manifest),
            "--cluster-mapping-csv",
            str(mapping_csv),
            "--oc-csv",
            str(args.oc_csv),
            "--fi-csv",
            str(args.fi_csv),
            "--num-simulations",
            str(args.num_simulations),
            "--max-abs-psnr-gap",
            str(args.max_abs_psnr_gap),
            "--max-abs-ssim-gap",
            str(args.max_abs_ssim_gap),
            "--split-csv",
            str(split_csv),
            "--eval-split",
            split_name,
            "--output-json",
            str(split_out_prefix.with_suffix(".json")),
            "--output-markdown",
            str(split_out_prefix.with_suffix(".md")),
            "--output-gap-csv",
            str(split_out_prefix.with_name(split_out_prefix.name + "_gaps.csv")),
            "--output-cluster-gap-csv",
            str(split_out_prefix.with_name(split_out_prefix.name + "_cluster_gaps.csv")),
            "--log-level",
            args.log_level,
        ]
        if args.max_per_cluster_psnr_gap is not None:
            cmd += ["--max-per-cluster-psnr-gap", str(args.max_per_cluster_psnr_gap)]
        if args.max_per_cluster_ssim_gap is not None:
            cmd += ["--max-per-cluster-ssim-gap", str(args.max_per_cluster_ssim_gap)]
        _run(cmd, cwd=root)
        validations[split_name] = _load_json(split_out_prefix.with_suffix(".json"))

    report = {
        "split_csv": str(split_csv),
        "joint_selection_recommendation": recommendation,
        "split_budget_recommendations": split_recommendations,
        "tune_validation": validations["tune"],
        "test_validation": validations["test"],
        "protocol_pass": bool(
            validations["tune"].get("overall_pass", False)
            and validations["test"].get("overall_pass", False)
        ),
    }

    report_json = out_dir / "holdout_protocol_report.json"
    report_md = out_dir / "holdout_protocol_report.md"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)

    lines = [
        "# Holdout Protocol Report",
        "",
        f"- Split CSV: `{split_csv}`",
        f"- Protocol pass: **{report['protocol_pass']}**",
        f"- Recommended k: **{recommendation['recommended_k']}**",
        f"- Recommended budget per cluster: **{recommendation['recommended_budget_per_cluster']}**",
        f"- Recommended subset size: **{recommendation['recommended_total_subset_size']}**",
        f"- Selection-split manifest: `{recommendation['exported_manifest']}`",
        "",
        "## Validation",
        f"- Tune split manifest: `{split_recommendations['tune']['exported_manifest']}`",
        f"- Test split manifest: `{split_recommendations['test']['exported_manifest']}`",
        f"- Tune pass: **{validations['tune'].get('overall_pass', False)}**",
        f"- Test pass: **{validations['test'].get('overall_pass', False)}**",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("Saved holdout report JSON: %s", report_json)
    LOGGER.info("Saved holdout report Markdown: %s", report_md)
    LOGGER.info("Protocol pass: %s", report["protocol_pass"])


if __name__ == "__main__":
    main()
