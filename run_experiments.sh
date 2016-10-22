#!/usr/bin/env bash

#### Set Paths ####
# Add caffe python path
export PYTHONPATH=$HOME/Developer/external/caffe/python:$PYTHONPATH

# Add OSMesaViewer
export PATH=$HOME/Developer/app/mesh-viewer/build/OSMesaViewer/build/Build/bin:$PATH
#### --------- ####


# DATASET_DIR=$HOME/Developer/data/ICL-NUIM
# IMAGE_WILDCARD=scene_00_%04d.png

DATASET_DIR=$HOME/Developer/data/7-scenes/chess
IMAGE_WILDCARD=frame-%06d.png

echo ${DATASET_DIR}
echo ${IMAGE_WILDCARD}
./run_batch.py --dataset_dir=${DATASET_DIR} --image_wildcard=${IMAGE_WILDCARD}

