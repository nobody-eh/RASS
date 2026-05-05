# Reproducibility Checklist

This checklist validates the anonymous RASS artifact and rebuilds the Kaggle
upload archive from repository files.

## 1. Environment

The packaged validation path only needs Python plus PyYAML:

```bash
python -m pip install pyyaml
```

The broader research scripts may require additional scientific Python packages
depending on which full-audit workflow is rerun.

## 2. Validate The Artifact

```bash
python rass_kaggle_artifact_anonymous/scripts/validate_artifact.py \
  --root rass_kaggle_artifact_anonymous
```

Expected summary for the current package:

- `Artifact validation passed.`
- `Files checked: 27`
- RASS-48 scene list count: 48
- RASS-96 scene list count: 96
- `TODO_REQUIRED` placeholders are listed for missing external or semantic
  fields that must be filled before making stronger claims

Additional count checks:

```bash
wc -l \
  rass_kaggle_artifact_anonymous/scene_lists/rass48_scene_ids.txt \
  rass_kaggle_artifact_anonymous/scene_lists/rass96_scene_ids.txt \
  rass_kaggle_artifact_anonymous/scene_lists/full_zipnerf_audit_ids.txt \
  rass_kaggle_artifact_anonymous/scene_lists/cross_method_common_ids.txt
```

Expected counts:

- `48` RASS-48 IDs
- `96` RASS-96 IDs
- `3625` full ZipNeRF audit IDs
- `3473` cross-method common IDs

## 3. Reproduce Packaged Summaries

```bash
python rass_kaggle_artifact_anonymous/scripts/reproduce_tables.py \
  --root rass_kaggle_artifact_anonymous
```

This summarizes the packaged audit CSVs and explicitly reports that full table
recomputation needs external per-scene metric tables.

Wilson lower confidence bound utility:

```bash
python rass_kaggle_artifact_anonymous/scripts/compute_wilson_lcb.py \
  --successes 113 \
  --trials 400 \
  --confidence 0.95
```

Expected value:

```text
0.240610272351
```

## 4. Rebuild The Upload Zip

```bash
rm -f rass_kaggle_artifact_anonymous_v1.zip
zip -r rass_kaggle_artifact_anonymous_v1.zip rass_kaggle_artifact_anonymous
unzip -t rass_kaggle_artifact_anonymous_v1.zip
```

The zip is a generated upload artifact and does not need to be committed when
the package folder is version-controlled.

## 5. Kaggle Metadata

Kaggle metadata lives in:

```text
rass_kaggle_artifact_anonymous/dataset-metadata.json
```

The current dataset ID is:

```text
nobodyeh/rass-nerf-benchmark-artifact
```

The package is configured as private through `isPrivate: true`.

## 6. Anonymity Scan

Before publishing an anonymous review copy, scan the committed files and inspect
git metadata:

```bash
rg -n -i "personal-name|institution-name|personal-kaggle-slug|personal-profile-url|local-absolute-path" \
  README.md REPRODUCIBILITY.md rass_kaggle_artifact_anonymous

git log --format='%an <%ae>' | sort -u
```

If git history contains identifying metadata, publish from a clean anonymized
repository or rewrite history before sharing the review URL.
