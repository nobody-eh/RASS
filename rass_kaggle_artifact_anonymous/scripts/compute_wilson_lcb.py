#!/usr/bin/env python3
"""Compute the Wilson lower confidence bound for a binomial pass rate."""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist


def wilson_lcb(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")

    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = phat + z2 / (2.0 * trials)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * trials)) / trials)
    return max(0.0, (center - margin) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successes", type=int, required=True, help="Number of passing trials.")
    parser.add_argument("--trials", type=int, required=True, help="Total number of trials.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Two-sided confidence level.")
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of a bare value.")
    args = parser.parse_args()

    value = wilson_lcb(args.successes, args.trials, args.confidence)
    if args.json:
        print(json.dumps({
            "successes": args.successes,
            "trials": args.trials,
            "confidence": args.confidence,
            "wilson_lcb": value,
        }, indent=2))
    else:
        print(f"{value:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
