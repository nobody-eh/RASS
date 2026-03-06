#!/usr/bin/env python3
"""Build a one-click final report pack for subset selection."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


LOGGER = logging.getLogger("build_final_report_pack")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: List[str], cwd: Path) -> None:
    LOGGER.info("Running: %s", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final subset-selection report pack.")
    parser.add_argument("--sweep-dir", type=Path, default=Path("sweep_cluster_k"))
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--cluster-mapping-csv", type=Path, required=True)
    parser.add_argument("--oc-csv", type=Path, default=Path("ingp_oc.csv"))
    parser.add_argument("--fi-csv", type=Path, default=Path("ingp_fi.csv"))
    parser.add_argument("--selected-k", type=int, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["all", "selection", "tune", "test"],
        default="all",
    )
    parser.add_argument("--max-abs-psnr-gap", type=float, default=0.5)
    parser.add_argument("--max-abs-ssim-gap", type=float, default=0.01)
    parser.add_argument("--max-per-cluster-psnr-gap", type=float, default=None)
    parser.add_argument("--max-per-cluster-ssim-gap", type=float, default=None)
    parser.add_argument("--num-simulations", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("final_report_pack"))
    parser.add_argument("--skip-visuals", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
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

    visual_out = out_dir / "visuals"
    if not args.skip_visuals:
        cmd = [
            sys.executable,
            str(root / "scripts" / "visualize_cluster_sweep.py"),
            "--sweep-dir",
            str(args.sweep_dir),
            "--subset-manifest",
            str(args.subset_manifest),
            "--output-dir",
            str(visual_out),
            "--log-level",
            args.log_level,
        ]
        if args.selected_k is not None:
            cmd += ["--selected-k", str(args.selected_k)]
        _run(cmd, cwd=root)

    validation_json = out_dir / "subset_validation.json"
    validation_md = out_dir / "subset_validation.md"
    if not args.skip_validation:
        cmd = [
            sys.executable,
            str(root / "scripts" / "validate_subset_metric_matching.py"),
            "--subset-manifest",
            str(args.subset_manifest),
            "--cluster-mapping-csv",
            str(args.cluster_mapping_csv),
            "--oc-csv",
            str(args.oc_csv),
            "--fi-csv",
            str(args.fi_csv),
            "--num-simulations",
            str(args.num_simulations),
            "--random-seed",
            str(args.random_seed),
            "--max-abs-psnr-gap",
            str(args.max_abs_psnr_gap),
            "--max-abs-ssim-gap",
            str(args.max_abs_ssim_gap),
            "--output-json",
            str(validation_json),
            "--output-markdown",
            str(validation_md),
            "--output-gap-csv",
            str(out_dir / "subset_validation_gaps.csv"),
            "--output-cluster-gap-csv",
            str(out_dir / "subset_validation_cluster_gaps.csv"),
            "--log-level",
            args.log_level,
        ]
        if args.split_csv is not None:
            cmd += ["--split-csv", str(args.split_csv), "--eval-split", args.eval_split]
        if args.max_per_cluster_psnr_gap is not None:
            cmd += ["--max-per-cluster-psnr-gap", str(args.max_per_cluster_psnr_gap)]
        if args.max_per_cluster_ssim_gap is not None:
            cmd += ["--max-per-cluster-ssim-gap", str(args.max_per_cluster_ssim_gap)]
        _run(cmd, cwd=root)

    joint_rec = _read_json(args.sweep_dir / "k_budget_sweep" / "k_budget_recommendation.json")
    val = _read_json(validation_json)

    pack = {
        "subset_manifest": str(args.subset_manifest),
        "cluster_mapping_csv": str(args.cluster_mapping_csv),
        "visual_report_md": str(visual_out / "visual_report.md"),
        "validation_json": str(validation_json) if validation_json.exists() else None,
        "joint_recommendation": joint_rec,
        "validation_summary": val,
    }
    pack_json = out_dir / "final_report_pack.json"
    with pack_json.open("w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, sort_keys=False)

    lines = [
        "# Final Report Pack",
        "",
        f"- Subset manifest: `{args.subset_manifest}`",
        f"- Cluster mapping: `{args.cluster_mapping_csv}`",
        f"- Visual report: `{visual_out / 'visual_report.md'}`",
        f"- Validation report: `{validation_md}`",
    ]
    if joint_rec is not None:
        lines += [
            "",
            "## Joint Recommendation",
            f"- Recommended k: **{joint_rec.get('recommended_k')}**",
            f"- Recommended budget per cluster: **{joint_rec.get('recommended_budget_per_cluster')}**",
            f"- Recommended total subset size: **{joint_rec.get('recommended_total_subset_size')}**",
            f"- Recommended manifest: `{joint_rec.get('exported_manifest')}`",
        ]
    if val is not None:
        lines += [
            "",
            "## Validation",
            f"- Overall pass: **{val.get('overall_pass')}**",
            f"- Eval split: **{val.get('eval_split')}**",
        ]
    lines.append("")
    pack_md = out_dir / "final_report_pack.md"
    pack_md.write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("Saved final report JSON: %s", pack_json)
    LOGGER.info("Saved final report Markdown: %s", pack_md)


if __name__ == "__main__":
    main()

