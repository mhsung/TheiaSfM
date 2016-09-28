#!/usr/bin/env bash

# Change variables.
DATA_DIR=$HOME/home/data/sfm/MVI_0206

# Detect objects.
../../py-faster-rcnn/multi/detect_multi.py --data_dir=${DATA_DIR}

# Track objects.
../../SORT/multi/sort_multi.py --data_dir=${DATA_DIR}

# Crop images.
../../py-faster-rcnn/multi/crop_multi.py --data_dir=${DATA_DIR}

# Estimate best orientation.
./estimate_best_multi.py --data_dir=${DATA_DIR}

# Estimate orientation scores.
./estimate_scores_multi.py --data_dir=${DATA_DIR}

# Render 3D model.
./render_best_multi.py --data_dir=${DATA_DIR}

# Composite images.
./composite_best_multi.py --data_dir=${DATA_DIR}

# Create video.
ffmpeg -i ${DATA_DIR}/convnet/object_composite_best/MVI_0206_%04d.png \
    -pix_fmt yuv420p -r 24 -b:v 8000k -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    ${DATA_DIR}/convnet/composite_best.mp4
