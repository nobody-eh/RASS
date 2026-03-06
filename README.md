# BASS: Risk-Aware Scene Subset Selection for Reliable Large-Scale NeRF Benchmarking

This repository contains the code, analysis pipeline, and paper assets for the BASS study on budget-aware, risk-aware scene subset selection for large-scale NeRF benchmarking on Nutrition5k.

It also keeps legacy Pre-NeRF 360 reconstruction/preprocessing scripts used in earlier experiments.

## Related Papers
- **BASS**: *Risk-Aware Scene Subset Selection for Reliable Large-Scale NeRF Benchmarking* (draft in [latex/main.tex](latex/main.tex))
- **Pre-NeRF 360**: [Pre-NeRF: Enriching Unbounded Appearances for Neural Radiance Fields](https://amughrabi.github.io/prenerf/)

## Repository Layout
- `scripts/`: end-to-end BASS pipelines, sweeps, validation, report/figure exporters
- `src/`: feature extraction, normalization, clustering, geometry utilities
- `latex/`: paper source and publication figures
- `docker/`: Dockerfiles and container build notes
- `tests/`: pipeline regression/unit tests
- `LLFF/`: LLFF submodule used by legacy workflows

## Environment
Recommended:
- Python 3.9+
- Linux + CUDA GPU (for NeRF runtime generation; not required for CSV-only analysis)

Typical Python dependencies for the BASS pipeline:
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `umap-learn`, `opencv-python`

## BASS Pipeline (Current)
The current workflow is centered on these scripts:
- `scripts/build_feature_report_pipeline.py`
- `scripts/run_feature_analysis_pipeline.py`
- `scripts/sweep_feature_clusters.py`
- `scripts/sweep_subset_budgets.py`
- `scripts/sweep_k_budget_selection.py`
- `scripts/run_holdout_subset_protocol.py`
- `scripts/build_final_report_pack.py`

### 1) One-Command Orchestration (Extract + Analyze)
Create a default config:
```bash
python3 scripts/run_full_feature_pipeline.py --write-default-config pipeline_config.json
```

Edit `pipeline_config.json` (especially `extract.args.scenes_root`), then run:
```bash
python3 scripts/run_full_feature_pipeline.py --config pipeline_config.json --dry-run
python3 scripts/run_full_feature_pipeline.py --config pipeline_config.json
```

Useful toggles:
```bash
python3 scripts/run_full_feature_pipeline.py --config pipeline_config.json --extract-only
python3 scripts/run_full_feature_pipeline.py --config pipeline_config.json --analysis-only
```

### 2) Extraction Only
```bash
python3 scripts/build_feature_report_pipeline.py \
  --scenes-root n5k360p \
  --transforms-name transforms_train.json \
  --output-dir . \
  --resume \
  --num-workers 8 \
  --max-frames-per-scene 16 \
  --frame-stride 2 \
  --log-level INFO
```

Main extraction artifacts:
- `feats2.csv`
- `feature_summary_stats.json`
- `feature_missingness.csv`
- `feature_correlation_matrix.csv`
- `feature_top_correlations.csv`
- `feature_failure_summary.csv`
- `feature_extraction_failures.json`
- `feature_extraction_report.json`
- `feature_extraction_report.md`

### 3) Analysis Only (Normalization + Clustering + Representatives)
```bash
python3 scripts/run_feature_analysis_pipeline.py \
  --input-csv feats2.csv \
  --normalized-csv feats_normalized.csv \
  --output-prefix clustered_scenes \
  --clustered-features-csv clustered_feats2.csv \
  --cluster-hist-csv clustered_scenes_cluster_size_histogram.csv \
  --analysis-report-json clustered_scenes_analysis_report.json \
  --analysis-report-md clustered_scenes_analysis_report.md \
  --method kmeans \
  --k-min 2 --k-max 10 \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --log-level INFO
```

Main analysis artifacts:
- `feats_normalized.csv`
- `clustered_feats2.csv`
- `clustered_scenes_cluster_size_histogram.csv`
- `clustered_scenes_representatives_centroid.txt` (or medoid equivalent)
- `clustered_scenes_reps_ingp_metrics_summary.csv`
- `clustered_scenes_analysis_report.json`
- `clustered_scenes_analysis_report.md`

### 4) Sweep Cluster Counts
```bash
python3 scripts/sweep_feature_clusters.py \
  --input-csv feats2.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --k-values 4,6,8,10,12,16 \
  --cluster-drop-cols num_images num_frames_total num_frames_used \
  --n-per-cluster 2 \
  --output-dir sweep_cluster_k \
  --skip-existing \
  --log-level INFO
```

Summaries are written to `sweep_cluster_k/k_sweep_summary.{csv,json,md}`.

### 5) Visualize Cluster Sweep
```bash
python3 scripts/visualize_cluster_sweep.py \
  --sweep-dir sweep_cluster_k \
  --subset-manifest subset_k6_manifest.csv \
  --log-level INFO
```

Outputs go to `sweep_cluster_k/visuals/` (dashboard, per-k plots, and report markdown).

### 6) Budget Sweep and Auto-Selection
```bash
python3 scripts/sweep_subset_budgets.py \
  --cluster-mapping-csv sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --budgets 2,4,6,8,10,12 \
  --num-simulations 2000 \
  --max-abs-psnr-gap 0.5 \
  --max-abs-ssim-gap 0.01 \
  --confidence-level 0.95 \
  --target-joint-pass-rate 0.10 \
  --max-total-subset 60 \
  --output-dir sweep_cluster_k/budget_sweep_k6
```

Key outputs:
- `budget_sweep_summary.csv/json/md`
- `budget_recommendation.json`
- `recommended_subset.csv`
- `manifests/subset_budget_<b>.csv`

### 7) Joint k x Budget Search
```bash
python3 scripts/sweep_k_budget_selection.py \
  --k-sweep-summary-csv sweep_cluster_k/k_sweep_summary.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --budgets 2,4,6,8,10,12 \
  --num-simulations 1500 \
  --target-joint-pass-rate 0.10 \
  --max-total-subset 60 \
  --output-dir sweep_cluster_k/k_budget_sweep
```

### 8) Holdout Protocol (Selection/Tune/Test)
```bash
python3 scripts/run_holdout_subset_protocol.py \
  --cluster-mapping-csv sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv \
  --k-sweep-summary-csv sweep_cluster_k/k_sweep_summary.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --budgets 2,4,6,8,10,12 \
  --num-simulations 1200 \
  --target-joint-pass-rate 0.10 \
  --max-total-subset 60 \
  --output-dir sweep_cluster_k/holdout_protocol
```

### 9) Build Final Report Pack
```bash
python3 scripts/build_final_report_pack.py \
  --sweep-dir sweep_cluster_k \
  --subset-manifest sweep_cluster_k/k_budget_sweep/recommended_subset.csv \
  --cluster-mapping-csv sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --selected-k 6 \
  --output-dir final_report_pack
```

## Figure and Paper Assets

### LaTeX Paper
- Main manuscript: `latex/main.tex`
- Existing figure outputs: `latex/figures/`

### Export publication figures from a Prism/share bundle
```bash
python3 scripts/export_latex_figures.py \
  --bundle-dir sweep_cluster_k/share_bundle_prism_20260304 \
  --output-dir latex/figures
```

### Individual figure generators
- `scripts/generate_risk_selection_concept_figure.py`
- `scripts/generate_budget_pass_lcb_figure.py`
- `scripts/generate_baselines_min_size_figure.py`
- `scripts/generate_ks_tau_knee_figure.py`
- `scripts/generate_refinement_before_after_figure.py`
- `scripts/generate_umap_coverage_57d_figure.py`
- `scripts/export_before_after_hist_figure.py`
- `scripts/export_umap_coverage_figure.py`

## Merge Precomputed Scene JSON Features
If you already have per-scene JSON files in `features_raw/`:
```bash
python3 scripts/merge_nutrition5k_features.py \
  --input-dir features_raw \
  --output-dir . \
  --log-level INFO
```

Outputs:
- `features_raw_merged.csv`
- `summary_stats.json`
- `correlation_matrix.csv`

## Legacy Pre-NeRF / Reconstruction Scripts
Legacy shell workflows are kept under `scripts/` for reproducibility and historical comparisons (for example: `run_colmap.sh`, `run_colmap_geo.sh`, `run_ingp.sh`, `multi_scene_processing.sh`, `parallel_*`).

Notes:
- These scripts are legacy/experimental and may depend on local tools, data layout, or helper scripts outside the BASS pipeline.
- For containerized setup, use the Docker notes in `docker/README.md`.

## Docker
See [docker/README.md](docker/README.md) for image build and run instructions.

## License
- Code in this repository: [GNU GPL v3](LICENSE)
- Pre-NeRF data release: Creative Commons BY-NC-ND 4.0 (as described in the Pre-NeRF publication)

## Citation
Pre-NeRF citation:
```bibtex
@misc{almughrabi2023prenerf,
  title={Pre-NeRF 360: Enriching Unbounded Appearances for Neural Radiance Fields},
  author={Ahmad AlMughrabi and Umair Haroon and Ricardo Marques and Petia Radeva},
  year={2023},
  eprint={2303.12234},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

For BASS, please cite the paper version corresponding to `latex/main.tex` once finalized.
