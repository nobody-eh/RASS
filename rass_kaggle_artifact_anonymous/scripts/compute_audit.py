#!/usr/bin/env python3
"""Audit computation scaffold for full RASS audit reproduction.

TODO_REQUIRED: Implement or port the exact paper audit loop. Required inputs
are candidate manifests, full metric tables, threshold settings, Wilson-LCB
confidence, trial count, and output paths.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Candidate manifest file or directory.")
    parser.add_argument("--metrics", required=True, help="Full-population metrics CSV/XLSX.")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml", help="Threshold YAML.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Wilson confidence level.")
    parser.add_argument("--output", required=True, help="Output audit frontier CSV.")
    parser.parse_args()
    raise SystemExit(
        "TODO_REQUIRED: Audit computation is a scaffold. Port the final paper audit loop before use."
    )


if __name__ == "__main__":
    raise SystemExit(main())
