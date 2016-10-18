#!/usr/bin/env bash

#### Set Paths ####
DATA_DIR=$HOME/Developer/data/ICL-NUIM/lr_kt2
echo ${DATA_DIR}

IMAGE_WILDCARD=scene_00_%04d.png

# Add caffe python path
export PYTHONPATH=$HOME/Developer/external/caffe/python:$PYTHONPATH

# Add OSMesaViewer
export PATH=$HOME/Developer/app/mesh-viewer/build/OSMesaViewer/build/Build/bin:$PATH
#### --------- ####


# Detect objects.
../py-faster-rcnn/detect_multi.py --data_dir=${DATA_DIR}

# Track objects.
../SORT/sort_multi.py --data_dir=${DATA_DIR}

# Crop images.
../py-faster-rcnn/crop_multi.py --data_dir=${DATA_DIR}

# Estimate best orientation.
# ./estimate_best_multi.py --data_dir=${DATA_DIR}

# Estimate orientation scores.
./estimate_scores_multi.py --data_dir=${DATA_DIR}

# Do seam fitting in camera parameter space.
./seam_fitting_multi.py --data_dir=${DATA_DIR}

# Render 3D model.
./render_multi.py --data_dir=${DATA_DIR}

# Composite images.
./composite_multi.py --data_dir=${DATA_DIR}

# Create video.
ffmpeg -i ${DATA_DIR}/convnet/object_composite_fitted/${IMAGE_WILDCARD} \
	-pix_fmt yuv420p -r 24 -b:v 8000k \
	${DATA_DIR}/convnet/composite_fitted.mp4

# ffmpeg -i ${DATA_DIR}/convnet/object_composite_fitted/${IMAGE_WILDCARD} \
# 	-pix_fmt yuv420p -r 24 -b:v 8000k -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
# 	${DATA_DIR}/convnet/composite_fitted.mp4

