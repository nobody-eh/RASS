#!/usr/bin/env python3
"""Candidate generation scaffold for full RASS audit reproduction.

TODO_REQUIRED: Implement or port the exact candidate-generation routine used
for the final paper audit. Required inputs are descriptors, regime labels,
target subset size, trial count, seed, and an output path.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", required=True, help="Scene descriptor CSV.")
    parser.add_argument("--regime-labels", required=True, help="Regime label CSV.")
    parser.add_argument("--subset-size", type=int, required=True, help="Target number of scenes.")
    parser.add_argument("--trials", type=int, default=400, help="Number of candidate trials.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", required=True, help="Output candidate manifest path.")
    parser.parse_args()
    raise SystemExit(
        "TODO_REQUIRED: Candidate generation is a scaffold. Port the exact paper-generation routine before use."
    )


if __name__ == "__main__":
    raise SystemExit(main())
