#!/bin/bash

DATASET_PATH=$1


if find "$DATASET_PATH/depths_compute" -mindepth 1 -maxdepth 1 | read; then
  echo "Does exist = $DATASET_PATH/depths_compute"
  exit 1
fi

mkdir -p "$DATASET_PATH"/depths_compute

for depth in "$DATASET_PATH"/depths/*; do
  if [ ! -f "$depth" ]; then
    continue
  fi
  mask="$DATASET_PATH"/masks/$(basename "$depth")
  if [[ ! -f "$mask" ]]; then
     # Check file extension and convert accordingly
     if [[ "$mask" == *.jpg || "$mask" == *.jpeg ]]; then
         mask="${mask%.*}.png"
     elif [[ "$file" == *.png ]]; then
         mask="${mask%.*}.jpg"
     else
         echo "File is not a JPG/JPEG or PNG."
     fi
  fi
  out="$DATASET_PATH"/depths_compute/"$(basename "$depth")"
  out="${out%.*}.npy"
  python src/pose/convert_depth.py --depth_path "$depth" --mask_path "$mask" --depth_out "$out"
done