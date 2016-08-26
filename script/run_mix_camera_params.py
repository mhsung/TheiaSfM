#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

from scipy import optimize

import gflags
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import plot_utils
import shutil
import sys


FLAGS = gflags.FLAGS
ANGLE_TYPES = ['Azimuth', 'Elevation', 'Theta']
NUM_ANGLE_TYPES = len(ANGLE_TYPES)
NUM_ANGLE_SAMPLES = 360


# Set input1 files.
gflags.DEFINE_string('data_dir', '/Users/msung/Developer/data/MVI_0206', '')
gflags.DEFINE_string('images', 'images/*.png', '')
gflags.DEFINE_string('input1_param_data_dir', 'sfm_10/interp_orientation', '')
gflags.DEFINE_string('input2_param_data_dir',
                     'convnet/interp_seam_fitting_params', '')
gflags.DEFINE_float('input1_weight', 1.0, '')
gflags.DEFINE_float('input2_weight', 0.0, '')
gflags.DEFINE_string('output_param_data_dir',
                     'sfm_track_test/target_orientation', '')


def mix_camera_params(y1_values, w1, y2_values, w2):
    assert (y1_values.shape[0] == y2_values.shape[0])
    num_frames = y1_values.shape[0]
    mixed_y_values = np.full((num_frames, NUM_ANGLE_TYPES), np.nan)

    is_not_nan1 = ~np.isnan(y1_values).any(axis=1)
    is_not_nan2 = ~np.isnan(y2_values).any(axis=1)

    for i in range(num_frames):
        if is_not_nan1[i] and is_not_nan2[i]:
            mixed_y_values[i, :] = w1 * y1_values[i, :] + w2 * y2_values[i, :]
        elif is_not_nan1[i] and not is_not_nan2[i]:
            mixed_y_values[i, :] = y1_values[i, :]

    return mixed_y_values


if __name__ == '__main__':
    FLAGS(sys.argv)

    image_wildcard = os.path.join(FLAGS.data_dir, FLAGS.images)
    file_prefix, min_frame, max_frame = plot_utils.read_images(image_wildcard)
    print('File prefix: {}'.format(file_prefix))
    print('Frame range: [{}, {}]'.format(min_frame, max_frame))


    # Read camera param data.
    input1_data_path = os.path.join(FLAGS.data_dir, FLAGS.input1_param_data_dir)
    x1_values, y1_values = plot_utils.read_frame_values(
        input1_data_path, file_prefix, min_frame, max_frame)
    print("Loaded '{}'.".format(input1_data_path))

    input2_data_path = os.path.join(FLAGS.data_dir, FLAGS.input2_param_data_dir)
    x2_values, y2_values = plot_utils.read_frame_values(
        input2_data_path, file_prefix, min_frame, max_frame)
    print("Loaded '{}'.".format(input2_data_path))

    assert (np.array_equal(x1_values, x2_values))
    mixed_y_values = mix_camera_params(
        y1_values, FLAGS.input1_weight, y2_values, FLAGS.input2_weight)

    output_data_path = os.path.join(FLAGS.data_dir, FLAGS.output_param_data_dir)
    if os.path.exists(output_data_path):
        shutil.rmtree(output_data_path)

    plot_utils.write_frame_values(
        output_data_path, file_prefix, x1_values, mixed_y_values)
    print("Saved '{}'.".format(output_data_path))
