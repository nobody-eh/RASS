#!/bin/bash

DATASET_PATH=$1
MESHSET_PATH=$2
echo $DATASET_PATH
# Check if the "sparse" directory exists inside DATASET_PATH
if [[ -f "$DATASET_PATH/locations.txt" ]]; then
    echo "Does not exist = $DATASET_PATH/locations.txt"
    exit 1
fi
# Dataset preparation
#find "$DATASET_PATH" -name Mask -type d -exec rm -rf {} +
#find "$DATASET_PATH" -name Depth -type d -exec rm -rf {} +
#find "$DATASET_PATH" -name original -type d -execdir mv {} images \;
#find "$DATASET_PATH" -name depth -type d -execdir mv {} depths \;
#find "$DATASET_PATH" -name Mask -type d -execdir mv {} masks \;

#echo "re-naming..."
# bash scripts/rename.sh "$DATASET_PATH"

# bash scripts/resize_colmap.sh "$DATASET_PATH"

# Camera caliberation: generate locations.txt file
#python src/pose/pose_extraction.py --dataset_path "$DATASET_PATH"

if [[ ! -f "$DATASET_PATH/locations.txt" ]]; then
    echo "Does not exist = $DATASET_PATH/locations.txt"
    exit 1
fi
# Run Colmap
bash scripts/run_colmap.sh "$DATASET_PATH"

#Run Foodmem
#find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_foodmem_docker.sh ?

#Run segmenter
#find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 8 --linebuffer python src/binary_img_seg_rgba.py --mask_dir ?/masks --rgb_dir ?/images --output ?/rgba

# Run NeuS2
#find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d | parallel -I? --max-args 1 --jobs 1 --linebuffer bash scripts/run_neus2.sh ?/transforms.json "$MESHSET_PATH"

# Mesh clean up
# find "$MESHSET_PATH" -mindepth 1 -maxdepth 1 -type f | parallel -I? --max-args 1 --jobs 7 --linebuffer python src/mesh_cleanup.py --mesh_path "?"  --mesh_cleaned_path "?"
#echo -e "\t\t\tUnits are in meter"
#echo -e "ID\tETA\tGT\tDiff\tSF\tCali.\tGeo\tDepth"
#for i in {1..14} ; do
#  # echo "$DATASET_PATH/$i"
#    if [ "$i" -eq 12 ]; then
#      continue
#    fi
#  python src/measurements.py --dataset_path "$DATASET_PATH/$i" --mesh "$MESHSET_PATH"/"$i".obj
#done


