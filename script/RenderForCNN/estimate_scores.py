#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../3rdparty/RenderForCNN'))
sys.path.append(BASE_DIR)
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'view_estimation_score'))
from evaluation_helper import viewpoint_scores

import gflags
import glob


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('output_dir', 'convnet/output', '')
gflags.DEFINE_string('target_class', 'chair', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    class_idx = g_shape_names.index(FLAGS.target_class)

    # Collect images.
    img_filenames = []
    class_idxs = []
    out_filenames = []

    for dirname in os.listdir(os.path.join(FLAGS.data_dir, FLAGS.output_dir)):
        basepath = os.path.join(FLAGS.data_dir, FLAGS.output_dir, dirname)
        if os.path.isdir(basepath):
            img_files = glob.glob(os.path.join(
                basepath, FLAGS.target_class + '_*.png'))
            print ('# {}(s) in {}: {}'.format(
                FLAGS.target_class, basepath, len(img_files)))
            for img_file in img_files:
                img_filenames.append(img_file)
                class_idxs.append(class_idx)
                # NOTE:
                # Will be stored as 'npy' file.
                out_filenames.append(
                    os.path.splitext(img_file)[0] + '_pred')

    # Estimate viewpoint scores.
    viewpoint_scores(img_filenames, class_idxs. out_filenames)

