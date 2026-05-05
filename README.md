# RASS: Risk-Audited Scene Subsets for NeRF Benchmarking

This repository contains the anonymous RASS artifact package for compact,
risk-audited NeRF benchmark scene lists and reproduction metadata.

Canonical repository location:

```text
https://github.com/nobody-eh/RASS
```

Current private Kaggle artifact:

```text
https://www.kaggle.com/datasets/nobodyeh/rass-nerf-benchmark-artifact
```

The Kaggle account slug is `nobodyeh`; the GitHub organization slug is
`nobody-eh`.

## What To Use

- `rass_kaggle_artifact_anonymous/`: Kaggle-ready anonymous artifact package.
- `rass_kaggle_artifact_anonymous/scene_lists/rass48_scene_ids.txt`: RASS-48,
  intended for rapid screening and smoke-test style comparisons.
- `rass_kaggle_artifact_anonymous/scene_lists/rass96_scene_ids.txt`: RASS-96,
  the stronger compact reporting option when a full run is too expensive.
- `rass_kaggle_artifact_anonymous/scene_lists/full_zipnerf_audit_ids.txt`: the
  full ZipNeRF audit population; use this for leaderboard-quality claims.
- `rass_kaggle_artifact_anonymous/metadata/artifact_manifest.json`: exact file
  inventory, source or placeholder status, required flags, and checksums.
- `REPRODUCIBILITY.md`: command checklist for validating and rebuilding the
  packaged artifact.

Raw Nutrition5k-derived assets and complete NeRF outputs are not redistributed
here unless permitted by their upstream terms. See
`rass_kaggle_artifact_anonymous/external_data/README_how_to_obtain_or_regenerate_inputs.md`.

## Scope

RASS-48 is for rapid screening. RASS-96 is the stronger compact reporting
option. The full audit population is still needed for leaderboard-quality
claims.

RASS does not certify arbitrary same-size subsets, regime-level fidelity,
cross-method ranking preservation, or nutrition/clinical conclusions. The
compact lists are benchmark engineering aids, not replacements for full
population evaluation or domain-specific validation.

## Quick Validation

From the repository root:

```bash
python -m pip install pyyaml
python rass_kaggle_artifact_anonymous/scripts/validate_artifact.py --root rass_kaggle_artifact_anonymous
```

Expected high-level checks:

- the artifact structure is complete
- JSON, CSV, and YAML files parse
- RASS-48 has 48 nonempty scene IDs
- RASS-96 has 96 nonempty scene IDs
- required `TODO_REQUIRED` placeholders are reported explicitly

## Packaged Reproduction

These commands use only the files distributed in the artifact package:

```bash
python rass_kaggle_artifact_anonymous/scripts/compute_wilson_lcb.py \
  --successes 113 \
  --trials 400 \
  --confidence 0.95

python rass_kaggle_artifact_anonymous/scripts/reproduce_tables.py \
  --root rass_kaggle_artifact_anonymous
```

To rebuild the local upload archive:

```bash
rm -f rass_kaggle_artifact_anonymous_v1.zip
zip -r rass_kaggle_artifact_anonymous_v1.zip rass_kaggle_artifact_anonymous
unzip -t rass_kaggle_artifact_anonymous_v1.zip
```

## Full Audit Reproduction

Full audit recomputation requires external inputs that are intentionally not
bundled into this repository:

- the permitted Nutrition5k-derived scene assets
- per-scene ZipNeRF full-image metrics for the full audit population
- cross-method NeRF result tables for matched-method diagnostics
- regenerated candidate manifests if auditing a new operating point

The package scripts are conservative: implemented utilities run directly, while
scripts that require unavailable external inputs state the required files rather
than inventing metrics, scene IDs, or scientific results.

## Repository Layout

- `rass_kaggle_artifact_anonymous/`: anonymous Kaggle package and validation
  scripts.
- `scripts/`: research and analysis utilities used by the broader local study.
- `src/`: feature extraction, normalization, clustering, and geometry helpers.
- `latex/`: manuscript source, if included in this checkout.
- `tests/`: regression and unit tests for source pipelines, if included in this
  checkout.

The artifact package is the review-safe entry point. Historical source scripts
may preserve older internal names or assumptions from earlier experiments.

## Anonymity Hygiene

The working-tree artifact has been scrubbed of direct author, institution,
personal profile, personal Kaggle owner, and local absolute-path identifiers.
For strict anonymous review, publish from a clean repository copy or verify that
git history, issue trackers, releases, and repository settings do not expose
personal metadata.
