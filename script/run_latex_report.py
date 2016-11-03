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


attr_names = ['Name', 'Trajectory_Image',
              'Num_Views', 'Num_Estimated_Views',
              'Mean_Rotation_Error', 'Median_Rotation_Error',
              'Mean_Aligned_Rotation_Error', 'Median_Aligned_Rotation_Error',
              'Mean_Aligned_Position_Error', 'Median_Aligned_Position_Error']

attr_types = [librr.AttrType.text, librr.AttrType.image,
              librr.AttrType.number, librr.AttrType.number,
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
        if prefix != "sfm_track" and prefix != "sfm_track_uio":
            continue
        print (prefix)

        trajectory_image_filepath = os.path.join(dirname, 'snapshot.png')

        with open(os.path.join(dirname, 'evaluation.json')) as eval_file:
            eval = json.load(eval_file)

        num_views = int(eval["num_views"])
        num_estimated_views = int(eval["num_estimated_views"])
        mean_rotation_error = float(eval["mean_rotation_error"])
        median_rotation_error = float(eval["median_rotation_error"])
        mean_aligned_rotation_error = float(eval["mean_aligned_rotation_error"])
        median_aligned_rotation_error =\
            float(eval["median_aligned_rotation_error"])
        mean_aligned_position_error = float(eval["mean_aligned_position_error"])
        median_aligned_position_error =\
            float(eval["median_aligned_position_error"])

        instance = OutputInstance(prefix, trajectory_image_filepath,
                                  num_views, num_estimated_views,
                                  mean_rotation_error, median_rotation_error,
                                  mean_aligned_rotation_error,
                                  median_aligned_rotation_error,
                                  mean_aligned_position_error,
                                  median_aligned_position_error)
        instances.append(instance)

    return instances


if __name__ == '__main__':
    FLAGS(sys.argv)

    dataset_name = os.path.splitext(os.path.basename(FLAGS.out_file))[0]
    dataset_name = dataset_name.replace('_', '\_')

    out_dir = os.path.dirname(FLAGS.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    instances = load_instances()

    file = librr.open_latex_table(
        FLAGS.out_file, dataset_name, attr_names, attr_types)
    librr.write_latex_table(file, instances, attr_names, attr_types)
    librr.close_latex_table(file)
