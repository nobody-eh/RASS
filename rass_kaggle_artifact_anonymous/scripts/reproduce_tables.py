#!/usr/bin/env python3
"""Summarize packaged audit tables and flag external inputs needed for full reproduction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Artifact root.")
    args = parser.parse_args()
    root = args.root
    audit = root / "results" / "audit_frontier.csv"
    cross = root / "results" / "cross_method_diagnostics.csv"
    for path in (audit, cross):
        if not path.is_file():
            raise SystemExit(f"missing packaged result: {path}")
    print(f"audit_frontier.csv rows: {count_rows(audit)}")
    print(f"cross_method_diagnostics.csv rows: {count_rows(cross)}")
    print(
        "TODO_REQUIRED: Full table recomputation requires external per-scene metric tables "
        "and regenerated candidates; see external_data/README_how_to_obtain_or_regenerate_inputs.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
