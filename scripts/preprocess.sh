#!/bin/bash

DATASET_PATH=$1
MESHSET_PATH=$2
# Dataset preparation
#find "$DATASET_PATH" -name Mask -type d -exec rm -rf {} +
#find "$DATASET_PATH" -name Depth -type d -exec rm -rf {} +
#find "$DATASET_PATH" -name Original -type d -execdir mv {} images \;

#echo "re-naming..."
# find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 6 --linebuffer bash scripts/rename.sh ?

# find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/resize_colmap.sh ?

# Camera caliberation: generate locations.txt file
#find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 8 --linebuffer python pose/pose_extraction.py --dataset_path ?

# Run Colmap
# find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_colmap.sh ?

#Run Foodmem
#find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_foodmem_docker.sh ?

#Run segmenter
# find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 8 --linebuffer python src/binary_img_seg_rgba.py --mask_dir ?/masks --rgb_dir ?/images --output ?/rgba

# Run NeuS2
find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_neus2.sh ?/transforms.json "$MESHSET_PATH"

# Mesh clean up
find "$MESHSET_PATH" -mindepth 1 -maxdepth 1 -type f | parallel -I? --max-args 1 --jobs 7 --linebuffer python src/mesh_cleanup.py --mesh_path "?"  --mesh_cleaned_path "?"

for i in {1..14} ; do
  # echo "$DATASET_PATH/$i"
    if [ "$i" -eq 12 ]; then
      continue
    fi
  python src/measurements.py --dataset_path "$DATASET_PATH/$i" --mesh "$MESHSET_PATH"/"$i".obj
done


