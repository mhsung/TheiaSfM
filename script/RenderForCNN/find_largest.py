#!/usr/bin/env python

import argparse
import gflags
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
from PIL import Image

BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../3rdparty/RenderForCNN'))
sys.path.append(BASE_DIR)
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'render_pipeline'))

# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('output_dir', 'convnet/output', '')
gflags.DEFINE_string('largest_dir', 'convnet/largest', '')
gflags.DEFINE_string('target_class', 'chair', '')


def findLargestBbox(test_name):
    basepath = os.path.join(FLAGS.data_dir, FLAGS.output_dir, test_name)
    assert(os.path.isdir(basepath))

    bbox_file_list = glob.glob(os.path.join(
        basepath, FLAGS.target_class + '*_bbox.txt'))

    # Find the largest bounding box.
    max_bbox_area = 0
    for bbox_file in bbox_file_list:
        estimated_bbox = [[float(x) for x in line.rstrip().split(' ')]
                          for line in open(bbox_file, 'r')]
        c = estimated_bbox[0]
        dx = c[2] - c[0]
        dy = c[3] - c[1]
        bbox_area = dx * dy

        basename = os.path.splitext(os.path.basename(bbox_file))[0]
        # Remove postfix.
        basename = basename[:len(basename)-len('_bbox')]

        if bbox_area > max_bbox_area:
            max_bbox_area = bbox_area
            max_bbox_name = basename

    assert(max_bbox_area > 0)
    return max_bbox_name


if __name__ == '__main__':
    FLAGS(sys.argv)

    if not os.path.isdir(os.path.join(FLAGS.data_dir, FLAGS.largest_dir)):
        os.makedirs(os.path.join(FLAGS.data_dir, FLAGS.largest_dir))

    image_names = os.listdir(os.path.join(FLAGS.data_dir, FLAGS.output_dir))
    for image_name in image_names:
        basepath = os.path.join(FLAGS.data_dir, FLAGS.output_dir, image_name)
        if not os.path.isdir(basepath):
            continue

        max_bbox_name = findLargestBbox(image_name)

        # Copy view file.
        if not os.path.isdir(os.path.join(
                FLAGS.data_dir, FLAGS.largest_dir, 'views')):
            os.makedirs(os.path.join(
                FLAGS.data_dir, FLAGS.largest_dir, 'views'))
        view_file = os.path.join(FLAGS.data_dir, FLAGS.output_dir, image_name,
                                 max_bbox_name + '_view.txt')
        out_view_file = os.path.join(FLAGS.data_dir, FLAGS.largest_dir,
                                     'views', image_name + '.txt')
        shutil.copy(view_file, out_view_file)

        # Copy bbox file.
        if not os.path.isdir(os.path.join(
                FLAGS.data_dir, FLAGS.largest_dir, 'bboxes')):
            os.makedirs(os.path.join(
                FLAGS.data_dir, FLAGS.largest_dir, 'bboxes'))
        bbox_file = os.path.join(FLAGS.data_dir, FLAGS.output_dir, image_name,
                                 max_bbox_name + '_bbox.txt')
        out_bbox_file = os.path.join(FLAGS.data_dir, FLAGS.largest_dir,
                                     'bboxes', image_name + '.txt')
        shutil.copy(bbox_file, out_bbox_file)

        # Copy pred file.
        if not os.path.isdir(os.path.join(
                FLAGS.data_dir, FLAGS.largest_dir, 'preds')):
            os.makedirs(os.path.join(
                FLAGS.data_dir, FLAGS.largest_dir, 'preds'))
        pred_file = os.path.join(FLAGS.data_dir, FLAGS.output_dir, image_name,
                                 max_bbox_name + '_pred.npy')
        out_pred_file = os.path.join(FLAGS.data_dir, FLAGS.largest_dir,
                                     'preds', image_name + '.npy')
        shutil.copy(pred_file, out_pred_file)

