#!/bin/bash

DATASET_PATH=$1
CURRENT_PATH=$(pwd)
echo $DATASET_PATH

pushd FoodMem
docker run --gpus all --rm -e DISPLAY=:1 -v /tmp/.X11-unix:/tmp/.X11-unix -v "$DATASET_PATH":/data -v $(pwd):/app gcvcg/foodmem bash run.sh /data
popd