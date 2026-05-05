# RASS / NeRF Benchmark Artifact

This package is a Kaggle-ready artifact for the RASS scene-subset audit used in the NeRF benchmark paper. The manifest records the exact source path or placeholder status for each copied or generated file.

## What This Artifact Contains

- `scene_lists/rass48_scene_ids.txt`: 48 scenes for rapid screening and smoke-test style comparisons.
- `scene_lists/rass96_scene_ids.txt`: 96 scenes for the stronger compact reporting option.
- `scene_lists/full_zipnerf_audit_ids.txt`: the full ZipNeRF audit population available in the repository. Leaderboard-quality claims should be checked against the full audit population, not only a compact list.
- `scene_lists/cross_method_common_ids.txt`: scenes shared by the available cross-method tables.
- `descriptors/`: normalized scene descriptors and k=6 regime labels copied from the source repository.
- `results/`: saved audit frontier and cross-method diagnostic CSVs copied from existing outputs.

## Scope And Limits

RASS-48 is intended for rapid screening. RASS-96 is the stronger compact reporting option when a full run is too expensive. The full audit population is still needed for leaderboard-quality claims.

RASS does not certify arbitrary same-size subsets. It also does not certify regime-level fidelity, cross-method ranking preservation, or nutrition/clinical conclusions. The compact lists are benchmark engineering aids, not a replacement for full-population evaluation or domain-specific validation.

Raw Nutrition5k-derived assets and complete NeRF outputs are not redistributed here unless permission is confirmed. See `external_data/README_how_to_obtain_or_regenerate_inputs.md`.

## Validate

From the repository root:

```bash
python rass_kaggle_artifact_anonymous/scripts/validate_artifact.py --root rass_kaggle_artifact_anonymous
```

From inside this folder:

```bash
python scripts/validate_artifact.py --root .
```

Validation checks required files, parses CSV/YAML/JSON files, verifies scene-list counts for the compact lists, and reports required placeholder markers.

## Reproduce The Audit

The packaged CSVs are copied from existing repository results. Full audit recomputation requires the external metric tables and any raw or regenerated NeRF outputs described under `external_data/`.

Typical flow:

```bash
python scripts/generate_candidates.py --help
python scripts/compute_fidelity_event.py --help
python scripts/compute_audit.py --help
python scripts/reproduce_tables.py --root .
```

`compute_wilson_lcb.py` is fully implemented and can be used independently:

```bash
python scripts/compute_wilson_lcb.py --successes 113 --trials 400 --confidence 0.95
```

## Anonymous Review Note

This copy removes direct author, institution, personal profile, personal Kaggle owner, and local absolute-path identifiers. The Kaggle owner slug is set to the neutral account `nobodyeh`.
