#!/usr/bin/env bash

#### Set Paths ####
DATA_DIR=$HOME/Developer/data/7-scenes/office_test/seq-01
echo ${DATA_DIR}

IMAGE_WILDCARD=frame-%06d.png

# Add caffe python path
export PYTHONPATH=$HOME/Developer/external/caffe/python:$PYTHONPATH

# Add OSMesaViewer
export PATH=$HOME/Developer/app/mesh-viewer/build/OSMesaViewer/build/Build/bin:$PATH

SUBCNN_PATH=$HOME/Developer/external/SubCNN/fast-rcnn

SELECTIVE_SEARCH_PATH=$HOME/Developer/external/selective_search_py
#### --------- ####

if [! -d "$SUBCNN_PATH/data/ObjectNet3D/" ]; then
	mkdir $SUBCNN_PATH/data/ObjectNet3D/
fi
rm -rf $SUBCNN_PATH/data/ObjectNet3D/Images
ln -s $DATA_DIR/images $SUBCNN_PATH/data/ObjectNet3D/Images

rm -rf $SUBCNN_PATH/data/ObjectNet3D/Image_sets
mkdir $SUBCNN_PATH/data/ObjectNet3D/Image_sets
find $DATA_DIR/images -name "*.png" -printf "%f\n" \
	> $SUBCNN_PATH/data/ObjectNet3D/Image_sets/test.txt

# Run selective search.
$SELECTIVE_SEARCH_PATH/batch_selective_search.py \
	--image_dir=$DATA_DIR/images --out_dir=$DATA_DIR/subcnn/selective_search
if [! -d "$SUBCNN_PATH/data/ObjectNet3D/region_proposals" ]; then
	mkdir $SUBCNN_PATH/data/ObjectNet3D/region_proposals
fi
rm -rf $SUBCNN_PATH/data/ObjectNet3D/region_proposals/selective_search
ln -s $DATA_DIR/subcnn/selective_search $SUBCNN_PATH/data/ObjectNet3D/region_proposals/selective_search

# Run SubCNN
CWD=$(pwd)
cd $SUBCNN_PATH
./experiments/scripts/test_subcnn_models_custom.sh
cd $CWD

# Detect objects.
# ../py-faster-rcnn/detect_multi.py --data_dir=${DATA_DIR}

# Track objects.
# ../SORT/sort_multi.py --data_dir=${DATA_DIR}

# Crop images.
# ../py-faster-rcnn/crop_multi.py --data_dir=${DATA_DIR}

# Estimate best orientation.
# ./estimate_best_multi.py --data_dir=${DATA_DIR}

# Estimate orientation scores.
# ./estimate_scores_multi.py --data_dir=${DATA_DIR}

# Do seam fitting in camera parameter space.
# ./seam_fitting_multi.py --data_dir=${DATA_DIR}

# Render 3D model.
# ./render_multi.py --data_dir=${DATA_DIR} \
#   --bbox_file=convnet/raw_bboxes.csv \
#   --orientation_file=convnet/raw_orientations.csv \
#   --out_render_dir=convnet/object_render_raw \
# 	--with_object_index=false \

# Composite images.
# ./composite_multi.py --data_dir=${DATA_DIR} \
#	--bbox_file=convnet/raw_bboxes.csv \
#	--render_dir=convnet/object_render_raw \
#	--out_composite_dir=convnet/object_composite_raw \
#	--with_object_index=false

