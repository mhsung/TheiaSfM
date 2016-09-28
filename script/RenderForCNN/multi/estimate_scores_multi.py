#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../../'))
sys.path.append(os.path.join(BASE_DIR, '3rdparty', 'RenderForCNN'))
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'view_estimation'))
from evaluation_helper import viewpoint_scores

import cnn_utils
import gflags
import glob


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('class_name_file', os.path.join(
    BASE_DIR, 'script/RenderForCNN/multi/class_names.txt'), '')
gflags.DEFINE_string('bbox_file', 'convnet/object_bboxes.csv', '')
gflags.DEFINE_string('crop_dir', 'convnet/object_crop', '')
gflags.DEFINE_string('out_orientation_score_dir', 'convnet/object_score', '')

gflags.DEFINE_bool('with_object_index', True, '')
gflags.DEFINE_integer('gpu_id', 0, 'GPU device id to use [0]')


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read class names to be detected.
    class_names = cnn_utils.read_class_names(FLAGS.class_name_file)

    # Read bounding boxes.
    df, num_digits = cnn_utils.read_bboxes(
        os.path.join(FLAGS.data_dir, FLAGS.bbox_file),
        FLAGS.with_object_index)

    if not os.path.exists(os.path.join(
            FLAGS.data_dir, FLAGS.out_orientation_score_dir)):
        os.makedirs(os.path.join(
            FLAGS.data_dir, FLAGS.out_orientation_score_dir))

    # Collect images.
    img_filenames = []
    class_idxs = []
    out_filenames = []

    for bbox_idx, row in df.iterrows():
        crop_im_file = os.path.join(FLAGS.data_dir, FLAGS.crop_dir,
                                    str(bbox_idx).zfill(num_digits) + '.png')
        img_filenames.append(crop_im_file)

        class_name = class_names[int(row['class_index'])]
        class_idx = g_shape_names.index(class_name)
        class_idxs.append(class_idx)

        # NOTE:
        # Will be stored as 'npy' file.
        out_file = os.path.join(FLAGS.data_dir,
                                FLAGS.out_orientation_score_dir,
                                str(bbox_idx).zfill(num_digits))
        out_filenames.append(out_file)

    # Estimate viewpoint scores.
    cfg.GPU_ID = FLAGS.gpu_id
    viewpoint_scores(img_filenames, class_idxs, out_filenames)
