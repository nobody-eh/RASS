# Pre-NeRF 360: Enriching Unbounded Appearances for Neural Radiance Fields
The repository contains the code release for paper: [Pre-NeRF](https://amughrabi.github.io/prenerf/)

## Nutrition5k Feature Pipeline (Current)
This repository now also includes a reproducible feature extraction + analysis pipeline for large Nutrition5k scene sets.

Main scripts:
- `scripts/build_feature_report_pipeline.py` for per-scene feature extraction and reporting
- `scripts/run_feature_analysis_pipeline.py` for normalization, clustering, representatives, and OC/FI summary
- `scripts/run_full_feature_pipeline.py` to orchestrate both steps from one JSON config
- `scripts/merge_nutrition5k_features.py` to merge precomputed `features_raw/*.json` into a single table

### Quick Start (One Command Pipeline)
Generate a default config:
```bash
python3 scripts/run_full_feature_pipeline.py --write-default-config pipeline_config.json
```

Review and edit `pipeline_config.json` (at minimum set `extract.args.scenes_root` to your scene root), then run:
```bash
python3 scripts/run_full_feature_pipeline.py --config pipeline_config.json --dry-run
python3 scripts/run_full_feature_pipeline.py --config pipeline_config.json
```

For faster extraction runs, tune:
- `extract.args.max_frames_per_scene` (e.g., `16`)
- `extract.args.frame_stride` (e.g., `2`)

### Extraction Only
```bash
python3 scripts/build_feature_report_pipeline.py \
  --scenes-root n5k360p \
  --transforms-name transforms_train.json \
  --output-dir . \
  --resume \
  --num-workers 8 \
  --log-level INFO
```

Speed-oriented variant (deterministic frame subsampling):
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

Key extraction artifacts:
- `feats2.csv`
- `feature_summary_stats.json`
- `feature_missingness.csv`
- `feature_correlation_matrix.csv`
- `feature_top_correlations.csv`
- `feature_failure_summary.csv`
- `feature_extraction_failures.json`
- `feature_extraction_report.json`
- `feature_extraction_report.md`

### Analysis Only
```bash
python3 scripts/run_feature_analysis_pipeline.py \
  --input-csv feats2.csv \
  --normalized-csv feats_normalized.csv \
  --output-prefix clustered_scenes \
  --clustered-features-csv clustered_feats2.csv \
  --cluster-hist-csv clustered_scenes_cluster_size_histogram.csv \
  --analysis-report-json clustered_scenes_analysis_report.json \
  --analysis-report-md clustered_scenes_analysis_report.md \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --log-level INFO
```

Key analysis artifacts:
- `feats_normalized.csv`
- `clustered_feats2.csv`
- `clustered_scenes_cluster_size_histogram.csv`
- `clustered_scenes_representatives_centroid.txt` (or medoid variant)
- `clustered_scenes_reps_ingp_metrics_summary.csv`
- `clustered_scenes_analysis_report.json`
- `clustered_scenes_analysis_report.md`

### Sweep Visuals
After running `scripts/sweep_feature_clusters.py`, generate interactive visuals:
```bash
python3 scripts/visualize_cluster_sweep.py \
  --sweep-dir sweep_cluster_k \
  --subset-manifest subset_k6_manifest.csv \
  --log-level INFO
```

This writes (by default) to `sweep_cluster_k/visuals/`:
- `k_sweep_dashboard.html`
- `k<k>_cluster_histogram.html`
- `k<k>_rep_metrics.html`
- `k<k>_cluster_profile_heatmap.html`
- `k<k>_subset_coverage.html` (if subset manifest provided)
- `visual_report.md`

### Subset Metric-Matching Validation
Use OC/FI PSNR/SSIM matching as a validation gate for a selected subset:
```bash
python3 scripts/validate_subset_metric_matching.py \
  --subset-manifest subset_k6_manifest.csv \
  --cluster-mapping-csv sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --num-simulations 3000 \
  --max-abs-psnr-gap 0.5 \
  --max-abs-ssim-gap 0.01 \
  --output-json sweep_cluster_k/visuals/k6_metric_matching_report.json \
  --output-markdown sweep_cluster_k/visuals/k6_metric_matching_report.md \
  --output-gap-csv sweep_cluster_k/visuals/k6_metric_matching_gaps.csv \
  --output-cluster-gap-csv sweep_cluster_k/visuals/k6_metric_matching_cluster_gaps.csv
```

This writes:
- `k6_metric_matching_report.json`
- `k6_metric_matching_report.md`
- `k6_metric_matching_gaps.csv`
- `k6_metric_matching_cluster_gaps.csv`

### Sweep Multiple Budgets
To estimate metric-matching stability across subset budgets and generate one
best manifest per budget:
```bash
python3 scripts/sweep_subset_budgets.py \
  --cluster-mapping-csv sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --budgets 2,4,6,8,10,12 \
  --num-simulations 2000 \
  --max-abs-psnr-gap 0.5 \
  --max-abs-ssim-gap 0.01 \
  --target-joint-pass-rate 0.10 \
  --max-total-subset 60 \
  --output-dir sweep_cluster_k/budget_sweep_k6
```

Outputs:
- `budget_sweep_summary.csv/json/md`
- `budget_recommendation.json` (auto-selected budget + rationale)
- `recommended_subset.csv` (copied from selected budget manifest)
- `manifests/subset_budget_<b>.csv` for each budget `b`

### Refine A Subset (Metric-Aware Swaps)
Refine an existing manifest while preserving cluster counts:
```bash
python3 scripts/refine_subset_metric_matching.py \
  --input-manifest subset_k6_manifest.csv \
  --cluster-mapping-csv sweep_cluster_k/k_6/clustered_scenes_k6_dish_cluster_mapping.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --max-iter 6 \
  --candidate-eval-limit 150 \
  --random-restarts 4 \
  --output-manifest sweep_cluster_k/refined_subset_k6_manifest.csv \
  --output-report-json sweep_cluster_k/refined_subset_k6_report.json \
  --output-report-md sweep_cluster_k/refined_subset_k6_report.md
```

### Joint k x Budget Selection
Run one sweep over both cluster count and subset budget:
```bash
python3 scripts/sweep_k_budget_selection.py \
  --k-sweep-summary-csv sweep_cluster_k/k_sweep_summary.csv \
  --oc-csv ingp_oc.csv \
  --fi-csv ingp_fi.csv \
  --budgets 2,4,6,8,10,12 \
  --num-simulations 1500 \
  --max-abs-psnr-gap 0.5 \
  --max-abs-ssim-gap 0.01 \
  --target-joint-pass-rate 0.10 \
  --max-total-subset 60 \
  --output-dir sweep_cluster_k/k_budget_sweep
```

Outputs:
- `k_budget_sweep_summary.csv/json/md`
- `k_budget_recommendation.json`
- `recommended_subset.csv`
- `manifests/subset_k<k>_budget_<b>.csv`

### Holdout Protocol (Selection/Tune/Test)
Create split and run holdout selection+validation:
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

Outputs:
- `holdout_split.csv` (if not provided)
- `joint_selection/` (selection-split k×budget sweep)
- `recommended_tune_validation.*`
- `recommended_test_validation.*`
- `holdout_protocol_report.json/md`

### One-Click Final Report Pack
Build visuals + validation + consolidated report:
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

Outputs:
- `final_report_pack/final_report_pack.json`
- `final_report_pack/final_report_pack.md`
- `final_report_pack/visuals/*`
- `final_report_pack/subset_validation.*`

### Merge Precomputed Raw JSON Features
If you already have one JSON per scene in `features_raw/`:
```bash
python3 scripts/merge_nutrition5k_features.py \
  --input-dir features_raw \
  --output-dir . \
  --log-level INFO
```

This writes:
- `features_raw_merged.csv`
- `summary_stats.json`
- `correlation_matrix.csv`

## How it run?
We highly recommend to use the docker image in [/docker](/docker). Please make sure that your installation is a GPU friendly. 
```bash
DATA_DIR=path/to/the/scene/directory
docker run -v $(pwd):/workspace \
           -v /path/to/360_v2_nk:/workspace/360_v2_nk \
           --gpus all --shm-size 24G --name prenerf --rm -it \
           --entrypoint bash -d prenerf/prenerf:latest \
           single_scene_processing.sh $DATA_DIR
```

## Do it yourself?
If you want to run the scripts instead of downloading the data. You can do the following:
```bash
bash rsync.sh 
```
Then, you need to run our `multi_scene_processing.sh`
```bash
bash multi_scene_processing.sh n5k360
```
Or for a single scene
```bash
bash single_scene_processing.sh n5k360/dish_1550705786
```

## Run on your custom data?
Each scene should be as one or more videos in a directory. For example
```bash
n5k360l/
├── dish_1550705786 (scene 1)
│   ├── camera_A.h264
│   ├── camera_B.h264
│   ├── camera_C.h264
│   └── camera_D.h264
├── dish_1550705888 (scene 2)
│   ├── camera_A.h264
│   ├── camera_B.h264
│   ├── camera_C.h264
│   └── camera_D.h264
└── dish_1550705939 (scene 3)
    ├── camera_A.h264
    ├── camera_B.h264
    ├── camera_C.h264
    └── camera_D.h264

```

Or if you have the scene as images, each scene folder should have `images` folder
```bash
n5k360/dish_1550704903/
└── images
   ├── 0001.png
   ├── 0002.png
   ├── 0003.png
   ├── 0004.png
   ├── ...
   ├── 0063.png
   └── 0064.png

```

## How fast is it?
[We] used [GNU Parallel](https://www.gnu.org/software/parallel/), it also shows a better utilisation for the 
multiprocessing on the GPU. However, the extending it takes `1-2hrs` of preparing data.

## License & Contact
We release all Pre-NeRF data under the <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives V4.0</a> license. You are free to share and adapt this data for any purpose, even commercially. If you found this dataset useful, please consider citing our [paper](https://arxiv.org/abs/2303.12234).
```

@misc{almughrabi2023prenerf,
      title={Pre-NeRF 360: Enriching Unbounded Appearances for Neural Radiance Fields}, 
      author={Ahmad AlMughrabi and Umair Haroon and Ricardo Marques and Petia Radeva},
      year={2023},
      eprint={2303.12234},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
If you have any questions about the Pre-NeRF dataset or paper, please email the authors, or feel free to file a ticket.
