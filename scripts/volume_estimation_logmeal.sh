#!/bin/bash
# This is a near-realtime colmap-free implementation based on logmeal data.

DATASET_PATH=$1
MESH_PATH=$1

# Convert ARKit to transform.json
find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 8 --linebuffer python3 src/ARCORE_TO_NERF_LOGMEAL.py --dataset_path ?

# Run
#find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_foodmem_docker.sh ?

#Run segmenter
find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 8 --linebuffer python3 src/binary_img_seg_rgba.py --mask_dir ?/masks --rgb_dir ?/images --output ?/rgba

find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 8 --linebuffer python3 src/update_configs.py "?/transforms.json"

# Remove old files
find "$DATASET_PATH" -mindepth 1 -maxdepth 2 -type f -name "*.obj" | parallel -I? --max-args 1 --jobs 7 --linebuffer rm -rf ?

#Run i-NGP
find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_ingp.sh "?/transforms.json"

# Run NeuS2
#find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_neus2.sh "?/transforms.json"

# Mesh cleanup, scaling factor, and Volume estimation
find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 7 --linebuffer python3 src/measurements.py --dataset_path "?"

