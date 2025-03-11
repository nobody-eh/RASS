#!/bin/bash
# This is a near-realtime colmap-free implementation based on logmeal data.

DATASET_PATH=$1
DATASET_NAME=$(basename "$DATASET_PATH")

python src/scaling.py --mesh_path "$DATASET_PATH"/"$DATASET_NAME".obj --transform "$DATASET_PATH/arcore2nerf_transforms.json" --mesh_out "$DATASET_PATH/$DATASET_NAME"_scaled.obj