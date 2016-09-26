#!/usr/bin/env bash

# Change variables.
DATA_DIR=$HOME/home/data/sfm/MVI_0206

# Detect objects.
# ../../py-faster-rcnn/multi/detect_multi.py --data_dir=${DATA_DIR}

# Crop images.
# ../../py-faster-rcnn/multi/crop_multi.py --data_dir=${DATA_DIR}

# Estimate best orientation.
# ./estimate_best_multi.py --data_dir=${DATA_DIR}

# Estimate orientation scores.
# ./estimate_scores_multi.py --data_dir=${DATA_DIR}

# Render 3D model.
./render_best_multi.py --data_dir=${DATA_DIR}
