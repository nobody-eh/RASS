#!/bin/bash

DATASET_PATH=$1
# Numerically rename all files in a given dataset.
echo $1
echo "re-naming..."
ls "$DATASET_PATH/images" | cat -n | while read n f; do mv "$DATASET_PATH/images/$f" `printf "$DATASET_PATH/images/%04d.jpg" $n`; done
# depth
ls "$DATASET_PATH/depths" | cat -n | while read n f; do mv "$DATASET_PATH/depths/$f" `printf "$DATASET_PATH/depths/%04d.jpg" $n`; done
# masks_ref
ls "$DATASET_PATH/masks" | cat -n | while read n f; do mv "$DATASET_PATH/masks/$f" `printf "$DATASET_PATH/masks/%04d.png" $n`; done