#!/usr/bin/python
# -*- coding: utf-8 -*-

import gflags
import glob
import os
import sys
import argparse

BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../3rdparty/RenderForCNN'))
sys.path.append(BASE_DIR)
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'view_estimation'))
from evaluation_helper import viewpoint

# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('output_dir', 'convnet/output', '')
gflags.DEFINE_string('target_class', 'chair', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    class_idxs = g_shape_names.index(FLAGS.target_class)

    in_img_file_list = []
    view_file_list = []

    for dirname in os.listdir(os.path.join(FLAGS.data_dir, FLAGS.output_dir)):
        basepath = os.path.join(FLAGS.data_dir, FLAGS.output_dir, dirname)
        if os.path.isdir(basepath):
            in_img_files = glob.glob(os.path.join(
                basepath, FLAGS.target_class + '_*.png'))
            print ('# {}(s) in {}: {}'.format(
                FLAGS.target_class, basepath, len(in_img_files)))
            for in_img_file in in_img_files:
                in_img_file_list.append(in_img_file)
                view_file_list.append(
                    os.path.splitext(in_img_file)[0] + '_view.txt')

    viewpoint(in_img_file_list, class_idxs, view_file_list)

