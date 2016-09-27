#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../../3rdparty/RenderForCNN'))
sys.path.append(BASE_DIR)
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'view_estimation'))
from evaluation_helper import viewpoint

import cnn_utils
import gflags
import glob
import numpy as np


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('class_name_file', os.path.join(
    BASE_DIR, '../../script/RenderForCNN/multi/class_names.txt'), '')
gflags.DEFINE_string('bbox_file', 'convnet/bboxes.csv', '')
gflags.DEFINE_string('crop_dir', 'convnet/crop', '')
gflags.DEFINE_string('out_best_orientation_file', 'convnet/orientations_best.csv', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read class names to be detected.
    class_names = cnn_utils.read_class_names(FLAGS.class_name_file)

    # Read bounding boxes.
    df, num_digits = cnn_utils.read_bboxes(
        os.path.join(FLAGS.data_dir, FLAGS.bbox_file))

    # Collect images.
    img_filenames = []
    class_idxs = []

    for bbox_idx, row in df.iterrows():
        crop_im_file = os.path.join(FLAGS.data_dir, FLAGS.crop_dir,
                                    str(bbox_idx).zfill(num_digits) + '.png')
        img_filenames.append(crop_im_file)

        class_name = class_names[int(row['class_index'])]
        class_idx = g_shape_names.index(class_name)
        class_idxs.append(class_idx)

    # Estimate viewpoints.
    preds = viewpoint(img_filenames, class_idxs)

    # Save results.
    np.savetxt(os.path.join(FLAGS.data_dir, FLAGS.out_best_orientation_file),
               preds, fmt='%i', delimiter=",")
