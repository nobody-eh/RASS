#!/bin/bash
# Run Semantic segmentation on the overhead images.
TRANSFORMATION_PATH="$1"
DATASET_PATH="$(dirname $(readlink -f "$TRANSFORMATION_PATH"))"
DATASET_NAME=$(basename "$DATASET_PATH")

#if [ -e "$DATASET_PATH/mesh.obj" ]; then
#    echo "Skip: $DATASET_PATH/mesh.obj"
#    exit 0
#fi

echo "${DATASET_PATH}/transforms.json"

# Measure time taken for the python script and report it
START_TIME=$(date +%s)

pushd instant-ngp
python scripts/run.py --mode nerf --scene "$DATASET_PATH/transforms.json"  --save_snapshot "$DATASET_PATH"/ckpts/"$DATASET_NAME".msgpack --save_mesh "$DATASET_PATH"/"$DATASET_NAME".obj  --train --n_steps 1000 --marching_cubes_res 256
popd

END_TIME=$(date +%s)
TIME_TAKEN=$((END_TIME - START_TIME))

# Convert time to hours, minutes, and seconds
HOURS=$((TIME_TAKEN / 3600))
MINUTES=$(( (TIME_TAKEN % 3600) / 60 ))
SECONDS=$((TIME_TAKEN % 60))

# Print and append the time taken to the log file in a human-readable format
TIME_TAKEN_STR=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)
echo "Time taken for dataset $DATASET_NAME: $TIME_TAKEN_STR (HH:MM:SS)"
echo "Time taken for dataset $DATASET_NAME: $TIME_TAKEN_STR (HH:MM:SS)" >> timing_log.txt

# clean up
rm -rf output/"$DATASET_NAME"

