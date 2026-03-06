#!/usr/bin/env python3
"""Run full Nutrition5k feature pipeline with a JSON config.

This wrapper orchestrates:
1) scripts/build_feature_report_pipeline.py
2) scripts/run_feature_analysis_pipeline.py

Usage patterns:
- Write starter config:
  python scripts/run_full_feature_pipeline.py --write-default-config pipeline_config.json
- Execute pipeline:
  python scripts/run_full_feature_pipeline.py --config pipeline_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


LOGGER = logging.getLogger("run_full_feature_pipeline")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config() -> Dict[str, Any]:
    return {
        "extract": {
            "enabled": True,
            "args": {
                "scenes_root": "n5k360p",
                "transforms_name": "transforms_train.json",
                "output_dir": ".",
                "raw_csv_name": "feats2.csv",
                "summary_json_name": "feature_summary_stats.json",
                "missing_csv_name": "feature_missingness.csv",
                "top_corr_csv_name": "feature_top_correlations.csv",
                "corr_csv_name": "feature_correlation_matrix.csv",
                "failure_summary_csv_name": "feature_failure_summary.csv",
                "failures_json_name": "feature_extraction_failures.json",
                "report_json_name": "feature_extraction_report.json",
                "report_md_name": "feature_extraction_report.md",
                "drift_json_name": "feature_drift_report.json",
                "cache_name": "feature_extraction_cache.json",
                "resume": True,
                "num_workers": 8,
                "max_frames_per_scene": None,
                "frame_stride": 1,
                "mask_dirs": "masks_omvs,masks,rgba",
                "require_mask": False,
                "max_failed_scenes_pct": 100.0,
                "max_feature_missing_pct": 100.0,
                "log_level": "INFO",
            },
        },
        "analysis": {
            "enabled": True,
            "args": {
                "input_csv": "feats2.csv",
                "normalized_csv": "feats_normalized.csv",
                "output_prefix": "clustered_scenes",
                "clustered_features_csv": "clustered_feats2.csv",
                "cluster_hist_csv": "clustered_scenes_cluster_size_histogram.csv",
                "analysis_report_json": "clustered_scenes_analysis_report.json",
                "analysis_report_md": "clustered_scenes_analysis_report.md",
                "method": "kmeans",
                "k_min": 2,
                "k_max": 10,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "rep_method": "centroid",
                "n_per_cluster": 4,
                "oc_csv": "ingp_oc.csv",
                "fi_csv": "ingp_fi.csv",
                "random_seed": 0,
                "log_level": "INFO",
            },
        },
    }


def _normalize_cli_key(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def _flatten_args(args_map: Dict[str, Any]) -> List[str]:
    cli: List[str] = []
    for key, value in args_map.items():
        flag = _normalize_cli_key(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            cli.append(flag)
            cli.extend(str(x) for x in value)
            continue
        cli.extend([flag, str(value)])
    return cli


def _run_command(cmd: List[str], *, cwd: Path, dry_run: bool) -> None:
    LOGGER.info("Running: %s", shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object")
    return payload


def _write_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(default_config(), f, indent=2, sort_keys=False)


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    sec = config.get(name, {})
    if not isinstance(sec, dict):
        raise ValueError(f"Config section '{name}' must be an object")
    return sec


def _section_enabled(section: Dict[str, Any]) -> bool:
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("Section 'enabled' must be boolean")
    return enabled


def _section_args(section: Dict[str, Any]) -> Dict[str, Any]:
    args = section.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Section 'args' must be an object")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction + analysis pipelines via JSON config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline_config.json"),
        help="Path to pipeline JSON config.",
    )
    parser.add_argument(
        "--write-default-config",
        type=Path,
        default=None,
        help="Write a default config JSON to the provided path and exit.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run only extraction stage even if analysis is enabled in config.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run only analysis stage even if extraction is enabled in config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def run_pipeline(config: Dict[str, Any], *, dry_run: bool, extract_only: bool, analysis_only: bool) -> None:
    if extract_only and analysis_only:
        raise ValueError("--extract-only and --analysis-only cannot be used together")

    root = _repo_root()
    extract = _section(config, "extract")
    analysis = _section(config, "analysis")

    run_extract = _section_enabled(extract)
    run_analysis = _section_enabled(analysis)

    if extract_only:
        run_analysis = False
    if analysis_only:
        run_extract = False

    if not run_extract and not run_analysis:
        raise ValueError("Nothing to run: both extract and analysis are disabled")

    python_exe = sys.executable

    if run_extract:
        extract_script = root / "scripts" / "build_feature_report_pipeline.py"
        extract_args = _section_args(extract)
        cmd = [python_exe, str(extract_script)] + _flatten_args(extract_args)
        _run_command(cmd, cwd=root, dry_run=dry_run)

    if run_analysis:
        analysis_script = root / "scripts" / "run_feature_analysis_pipeline.py"
        analysis_args = _section_args(analysis)
        cmd = [python_exe, str(analysis_script)] + _flatten_args(analysis_args)
        _run_command(cmd, cwd=root, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.write_default_config is not None:
        _write_default_config(args.write_default_config)
        LOGGER.info("Wrote default config: %s", args.write_default_config)
        return

    if not args.config.exists():
        raise FileNotFoundError(
            f"Config not found: {args.config}. "
            "Run with --write-default-config to generate one."
        )

    config = _load_config(args.config)
    run_pipeline(
        config,
        dry_run=args.dry_run,
        extract_only=args.extract_only,
        analysis_only=args.analysis_only,
    )


if __name__ == "__main__":
    main()
