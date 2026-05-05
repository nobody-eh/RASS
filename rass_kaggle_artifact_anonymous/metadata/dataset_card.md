# Dataset Card: RASS / NeRF Benchmark Artifact

## Summary

This anonymized review artifact packages compact scene lists, descriptors, regime labels, audit settings, copied diagnostic results, and minimal reproduction scripts for RASS-style scene-subset auditing in a NeRF benchmark paper.

This package keeps copied-file provenance and placeholder status in `metadata/artifact_manifest.json`.

## Intended Use

- Rapid screening with RASS-48.
- Stronger compact reporting with RASS-96.
- Structural validation and audit-script scaffolding on Kaggle.
- Reproduction of copied audit tables when the required external metric tables and regenerated NeRF outputs are available.

## Not Intended For

RASS does not certify arbitrary same-size subsets, regime-level fidelity, cross-method ranking preservation, or nutrition/clinical conclusions. The compact lists are not a substitute for full-population leaderboard evaluation.

## Data Composition

- Scene ID lists for compact and full audit populations.
- Normalized scene descriptors copied from the k=6 descriptor table.
- k=6 regime labels copied from the source repository.
- Existing audit frontier and cross-method diagnostic CSVs.
- Metadata, configs, validation code, Wilson LCB code, and reproduction stubs.

## External Data

Raw Nutrition5k-derived assets and full NeRF outputs are not redistributed here unless permitted. Users must obtain or regenerate those inputs separately. See `external_data/README_how_to_obtain_or_regenerate_inputs.md`.

## Known TODOs

Required placeholder markers identify fields that need paper-specific or hosting-specific information before archival release, including final citation details, DOI, public URL, and confirmed upstream licensing details.

## License

The copied repository license is GPL-3.0. Upstream raw-data and external model-output licensing may impose separate requirements.
