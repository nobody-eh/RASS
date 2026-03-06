#!/bin/bash

DATASET_PATH=$1
BASENAME=$(basename $DATASET_PATH)

echo "$BASENAME,$(python -u src/feature_extractor.py $DATASET_PATH/transforms_train.json)"