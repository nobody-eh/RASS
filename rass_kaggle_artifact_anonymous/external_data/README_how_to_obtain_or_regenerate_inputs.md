# External Data And Regeneration Notes

This artifact does not redistribute raw Nutrition5k-derived scene assets or full NeRF outputs unless redistribution is confirmed as permitted. The packaged files are lightweight scene lists, descriptors, labels, configs, metadata, copied CSV diagnostics, and scripts.

To reproduce the full audit, users need external inputs matching the paper run:

- Raw or prepared Nutrition5k-derived scene assets for the listed `dish_id` scenes.
- Per-scene ZipNeRF full-image metrics for the full audit population.
- Per-scene metric tables for every additional NeRF method included in cross-method claims.
- Any method-specific training/rendering outputs needed to regenerate those metric tables.

Suggested regeneration path from the source repository:

1. Obtain the raw dataset under its upstream terms.
2. Regenerate scene descriptors with the repository feature pipeline.
3. Run or mount the NeRF method outputs and per-scene metric tables.
4. Recreate candidate subsets and audit diagnostics with the scripts in this package or the source repository.
5. Validate the final package with `scripts/validate_artifact.py`.

Do not upload raw assets or full generated outputs to Kaggle unless the applicable dataset, model-output, and institutional permissions allow redistribution.
