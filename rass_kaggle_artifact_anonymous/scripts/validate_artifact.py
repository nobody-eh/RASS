#!/usr/bin/env python3
"""Validate the RASS Kaggle artifact structure and parseable files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "dataset-metadata.json",
    "metadata/croissant.json",
    "metadata/dataset_card.md",
    "metadata/artifact_manifest.json",
    "scene_lists/rass48_scene_ids.txt",
    "scene_lists/rass96_scene_ids.txt",
    "scene_lists/full_zipnerf_audit_ids.txt",
    "scene_lists/cross_method_common_ids.txt",
    "descriptors/scene_descriptors.csv",
    "descriptors/regime_labels.csv",
    "descriptors/descriptor_schema.json",
    "configs/thresholds.yaml",
    "configs/seeds.yaml",
    "configs/audit_settings.yaml",
    "scripts/generate_candidates.py",
    "scripts/compute_fidelity_event.py",
    "scripts/compute_wilson_lcb.py",
    "scripts/compute_audit.py",
    "scripts/reproduce_tables.py",
    "scripts/validate_artifact.py",
    "results/audit_frontier.csv",
    "results/cross_method_diagnostics.csv",
    "external_data/README_how_to_obtain_or_regenerate_inputs.md",
]


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def parse_json(path: Path) -> None:
    with path.open(encoding="utf-8") as f:
        json.load(f)


def parse_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    if not rows[0]:
        raise ValueError("CSV header is empty")
    return max(0, len(rows) - 1)


def parse_yaml(path: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse YAML files")
    with path.open(encoding="utf-8") as f:
        yaml.safe_load(f)


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def find_todos(root: Path) -> list[tuple[str, int, str]]:
    marker = "TODO" + "_REQUIRED"
    hits: list[tuple[str, int, str]] = []
    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if marker in line:
                hits.append((path.relative_to(root).as_posix(), lineno, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Artifact root directory.")
    args = parser.parse_args()
    root = args.root.resolve()

    errors: list[str] = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if not path.is_file():
            errors.append(f"missing required file: {rel}")

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            if path.suffix == ".json":
                parse_json(path)
            elif path.suffix == ".csv":
                parse_csv(path)
            elif path.suffix in {".yaml", ".yml"}:
                parse_yaml(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"parse failure in {rel}: {exc}")

    expected_counts = {
        "scene_lists/rass48_scene_ids.txt": 48,
        "scene_lists/rass96_scene_ids.txt": 96,
    }
    for rel, expected in expected_counts.items():
        path = root / rel
        if path.exists():
            observed = count_nonempty_lines(path)
            if observed != expected:
                errors.append(f"{rel} has {observed} nonempty lines; expected {expected}")

    todos = find_todos(root)

    if errors:
        print("Artifact validation failed:")
        for error in errors:
            print(f"- {error}")
        if todos:
            print(f"{'TODO' + '_REQUIRED'} placeholders found: {len(todos)}")
        return 1

    print("Artifact validation passed.")
    print(f"Files checked: {sum(1 for _ in iter_files(root))}")
    if todos:
        print(f"{'TODO' + '_REQUIRED'} placeholders found: {len(todos)}")
        for rel, lineno, text in todos[:25]:
            print(f"- {rel}:{lineno}: {text}")
        if len(todos) > 25:
            print(f"- ... {len(todos) - 25} more")
    else:
        print(f"{'TODO' + '_REQUIRED'} placeholders found: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
