#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

from scipy import optimize

import gflags
import os
import plot_utils
import sys


FLAGS = gflags.FLAGS
ANGLE_TYPES = ['Azimuth', 'Elevation', 'Theta']
NUM_ANGLE_TYPES = len(ANGLE_TYPES)
NUM_ANGLE_SAMPLES = 360


# Set input files.
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('images', 'images/*.png', '')
gflags.DEFINE_string('input_param_data_dir', 'sfm/orientation', '')
gflags.DEFINE_string('output_param_data_dir', 'sfm/interp_orientation', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    image_wildcard = os.path.join(FLAGS.data_dir, FLAGS.images)
    file_prefix, min_frame, max_frame = plot_utils.read_images(image_wildcard)
    print('File prefix: {}'.format(file_prefix))
    print('Frame range: [{}, {}]'.format(min_frame, max_frame))

    # Read camera param data.
    input_data_path = os.path.join(FLAGS.data_dir, FLAGS.input_param_data_dir)
    x_values, y_values = plot_utils.read_frame_values(
        input_data_path, file_prefix, min_frame, max_frame)
    print("Loaded '{}'.".format(input_data_path))

    interp_y_values = plot_utils.linearly_interpolate_angles(y_values)
    output_data_path = os.path.join(FLAGS.data_dir, FLAGS.output_param_data_dir)
    plot_utils.write_frame_values(
        output_data_path, file_prefix, x_values, interp_y_values)
    print("Saved '{}'.".format(output_data_path))
