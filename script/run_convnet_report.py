#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(BASE_DIR, 'script', 'tools'))

from collections import namedtuple
import lib_result_report as librr
import gflags
import glob
import json


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('out_file', '', '')


attr_names = ['Num_Views', 'Num_Objects', 'Num_BBoxes',
              'Mean_Num_BBoxes_Per_Object', 'View_Proportion_With_BBoxes',
              'Mean_ConvNet_Rotation_Error', 'Median_ConvNet_Rotation_Error',
              'Mean_ConvNet_Position_Error', 'Median_ConvNet_Position_Error']

attr_types = [librr.AttrType.number, librr.AttrType.number, librr.AttrType.number,
              librr.AttrType.number, librr.AttrType.number,
              librr.AttrType.number, librr.AttrType.number,
              librr.AttrType.number, librr.AttrType.number]

OutputInstance = namedtuple('OutputInstance', attr_names)


def load_instances():
    instances = []

    dirnames = glob.glob(FLAGS.data_dir + '/*')
    for dirname in dirnames:
        if not os.path.isdir(dirname):
            continue

        prefix = os.path.basename(dirname)
        if not prefix.startswith('sfm'):
            continue
        if prefix != "sfm_track_uio":
            continue

        with open(os.path.join(dirname, 'convnet_stats.json')) as eval_file:
            eval = json.load(eval_file)

        num_views = int(eval["num_views"])
        num_objects = int(eval["num_objects"])
        num_bboxes = int(eval["num_bboxes"])
        mean_num_bboxes_per_object = float(eval["mean_num_bboxes_per_object"])
        view_proportion_with_bboxes = float(eval["view_proportion_with_bboxes"])
        mean_convnet_rotation_error = float(eval["mean_convnet_rotation_error"])
        median_convnet_rotation_error =\
            float(eval["median_convnet_rotation_error"])
        mean_convnet_position_error = float(eval["mean_convnet_position_error"])
        median_convnet_position_error =\
            float(eval["median_convnet_position_error"])

        instance = OutputInstance(num_views, num_objects, num_bboxes,
                                  mean_num_bboxes_per_object,
                                  view_proportion_with_bboxes,
                                  mean_convnet_rotation_error,
                                  median_convnet_rotation_error,
                                  mean_convnet_position_error,
                                  median_convnet_position_error)
        instances.append(instance)

    return instances


if __name__ == '__main__':
    FLAGS(sys.argv)

    dataset_name = os.path.splitext(os.path.basename(FLAGS.out_file))[0]
    dataset_name = dataset_name.replace('_', '\_')
    print(dataset_name)

    out_dir = os.path.dirname(FLAGS.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    instances = load_instances()
    data_name = os.path.basename(FLAGS.data_dir)
    librr.write_csv_file(FLAGS.out_file, data_name,
            instances, attr_names, attr_types)

