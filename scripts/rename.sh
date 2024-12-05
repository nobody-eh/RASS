#!/bin/bash

DATASET_PATH=$1
# Numerically rename all files in a given dataset.
echo "re-naming..."
ls "$DATASET_PATH/images" | cat -n | while read n f; do mv "$DATASET_PATH/images/$f" `printf "$DATASET_PATH/images/%04d.jpg" $n`; done
# depth
ls "$DATASET_PATH/depth" | cat -n | while read n f; do mv "$DATASET_PATH/depth/$f" `printf "$DATASET_PATH/depth/%04d.jpg" $n`; done
# masks_ref
ls "$DATASET_PATH/masks_ref" | cat -n | while read n f; do mv "$DATASET_PATH/masks_ref/$f" `printf "$DATASET_PATH/masks_ref/%04d.png" $n`; done