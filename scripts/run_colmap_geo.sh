#!/bin/bash

DATASET_PATH=$1
PROJECT_DIR=$(pwd)

# Cleanup
rm -rf "$DATASET_PATH"/sparse/  "$DATASET_PATH"/mvs/ "$DATASET_PATH"/database.db

colmap feature_extractor --database_path "$DATASET_PATH"/database.db \
                         --image_path "$DATASET_PATH"/images \
                         --ImageReader.single_camera "1" \
                         --ImageReader.default_focal_length_factor "2.6" \
                         --SiftExtraction.use_gpu "1" \
                         --SiftExtraction.max_image_size "1000" \
                         --SiftExtraction.max_num_features "2048"

colmap exhaustive_matcher --database_path "$DATASET_PATH"/database.db \
                          --SiftMatching.use_gpu "1"


mkdir -p  "$DATASET_PATH"/sparse

colmap mapper --database_path "$DATASET_PATH"/database.db \
              --image_path "$DATASET_PATH"/images \
              --output_path "$DATASET_PATH"/sparse \
              --Mapper.ba_global_function_tolerance=0.000001 \
              --Mapper.init_min_tri_angle=16 \
              --Mapper.min_focal_length_ratio=0.1 \
              --Mapper.max_focal_length_ratio=10 \
              --Mapper.ba_local_max_num_iterations=25 \
              --Mapper.ba_global_max_num_iterations=50 \
              --Mapper.ba_global_images_ratio=1.1 \
              --Mapper.ba_global_points_ratio=1.1 \
              --Mapper.ba_global_max_refinements=5

# For very edge cases, COLMAP split the models into disjoints models (folders), this happened because there is no general
# relationships between these subsets. https://github.com/colmap/colmap/issues/1225
cp -r "$DATASET_PATH"/sparse/0/*.bin "$DATASET_PATH"/sparse/
for path in ${1}/sparse/*/; do
    subset=$(basename ${path})
    if [ ${subset} != "0" ]; then
      echo "Warning, found disjoints models"
      colmap model_merger \
          --input_path1="$DATASET_PATH"/sparse \
          --input_path2="$DATASET_PATH"/sparse/${subset} \
          --output_path="$DATASET_PATH"/sparse
      colmap bundle_adjuster \
          --input_path="$DATASET_PATH"/sparse \
          --output_path="$DATASET_PATH"/sparse
    fi
done

mkdir -p "$DATASET_PATH"/sparse/geo
colmap model_aligner --input_path "$DATASET_PATH"/sparse \
                     --output_path "$DATASET_PATH"/sparse/geo \
                     --ref_images_path "$DATASET_PATH"/locations.txt \
                     --ref_is_gps 0 \
                     --alignment_type ecef \
                     --robust_alignment 1 \
                     --robust_alignment_max_error 3

mkdir -p "$DATASET_PATH"/sparse/txt

colmap model_converter \
    --input_path "$DATASET_PATH"/sparse/0 \
    --output_path "$DATASET_PATH"/sparse/txt \
    --output_type TXT

## Convert COLMAP data to something parsable belnder format
python src/colmap2nerf.py --colmap_db "$DATASET_PATH"/database.db \
                                          --colmap_camera_model SIMPLE_RADIAL \
                                          --images "$DATASET_PATH"/images/ \
                                          --text "$DATASET_PATH"/sparse/txt \
                                          --out "$DATASET_PATH"/transforms.json \
                                          --keep_colmap_coords

# Update NeuS2 configuration
python3 src/update_configs.py "$DATASET_PATH"/transforms.json
