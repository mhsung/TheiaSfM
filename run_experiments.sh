#!/usr/bin/env bash

DATASET_DIR=$HOME/Developer/data/ICL-NUIM
IMAGE_WILDCARD=scene_00_%04d.png

# DATASET_DIR=$HOME/Developer/data/7-scenes/office
# IMAGE_WILDCARD=frame-%06d.png

echo ${DATASET_DIR}
echo ${IMAGE_WILDCARD}
./run_batch.py --dataset_dir=${DATASET_DIR} --image_wildcard=${IMAGE_WILDCARD}

