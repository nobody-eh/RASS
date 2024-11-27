#!/bin/bash

DATASET_PATH=$1
# Numerically rename all files in a given dataset.
echo "re-naming..."
ls "$DATASET_PATH/images" | cat -n | while read n f; do mv "$DATASET_PATH/images/$f" `printf "$DATASET_PATH/images/%04d.jpg" $n`; done