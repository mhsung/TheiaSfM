#!/usr/bin/env bash

# Change variables.
#DATA_ADDRESS=data/mit_dorm_mcc_eflr6/dorm_mcc_eflr6_oct_31_2012_scan1_erika
#DATA_DIR=$HOME/home/data/sun3d.cs.princeton.edu/$DATA_ADDRESS
DATA_DIR=$HOME/home/data/cmu_localization/data_collection_20100901/c0
echo ${DATA_DIR}

# Download Sun3D data.
# ../../get_sun3d_data.py --data_address=${DATA_ADDRESS}

# Detect objects.
../../py-faster-rcnn/multi/detect_multi.py --data_dir=${DATA_DIR} --gpu_id=0

# Track objects.
../../SORT/multi/sort_multi.py --data_dir=${DATA_DIR}

# Crop images.
../../py-faster-rcnn/multi/crop_multi.py --data_dir=${DATA_DIR}

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
ffmpeg -i ${DATA_DIR}/convnet/object_composite_fitted/%07d.png \
    -pix_fmt yuv420p -r 24 -b:v 8000k -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    ${DATA_DIR}/convnet/composite_fitted.mp4
# ffmpeg -i ${DATA_DIR}/convnet/object_composite_fitted/%07d.png \
#     -pix_fmt yuv420p -r 24 -b:v 8000k \
#     ${DATA_DIR}/convnet/composite_fitted.mp4
