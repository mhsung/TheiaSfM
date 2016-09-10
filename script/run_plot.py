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


# Set input files.
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('images', 'images/*.png', '')
gflags.DEFINE_string('param_data_names', 'Theia,Theia_track', '')
gflags.DEFINE_string('param_data_dirs',
                     'sfm/orientation,sfm_track/orientation', '')
gflags.DEFINE_string('convnet_dir', '', '')
gflags.DEFINE_string('output_plot_file', 'plot.png', '')
#gflags.DEFINE_string('output_convnet_max_score', '', '')
gflags.DEFINE_string('output_convnet_seam_fitting',
                     'convnet/fitted_params', '')


def plot_data(data_name_list, data_x_list, data_y_list,
              cn_pred_x=None, cn_pred_y=None, cn_pred_z=None):
    num_data = len(data_name_list)
    assert (len(data_x_list) == num_data)
    assert (len(data_y_list) == num_data)

    colors = cm.jet(np.linspace(0., 1., num_data))
    fig = plt.figure(figsize=(24, 12))

    for type_index in range(NUM_ANGLE_TYPES):
        plt.subplot(3, 1, type_index + 1)

        # Determine Y range shift using the first data.
        shift_y_range = False
        if num_data > 0:
            y_values = data_y_list[0][:, type_index]
            shift_y_range = plot_utils.good_to_shift_angle_range(y_values)

        y_lim = np.array([0, 360])
        if shift_y_range:
            y_lim += 180

        if cn_pred_x is not None and\
            cn_pred_y is not None and\
            cn_pred_z is not None:
            x_values = cn_pred_x
            y_values = cn_pred_y
            z_values = cn_pred_z[type_index]
            if shift_y_range:
                y_values = np.hstack((
                    y_values[:, 180:], y_values[:, :180] + 360))
                z_values = np.hstack((z_values[:, 180:], z_values[:, :180]))
            plt.pcolor(x_values, y_values, z_values, cmap='Oranges', vmin=0)
            plt.colorbar()

        for data_index in range(num_data):
            x_values = data_x_list[data_index]
            y_values = data_y_list[data_index][:, type_index]
            if shift_y_range:
                y_values[y_values < 180] += 360
            plt.plot(x_values, y_values, '.-', label=data_name_list[data_index],
                     color=colors[data_index])

        plt.ylim(y_lim)
        plt.xlabel("Frame Index")
        plt.ylabel("Angle (Degree)")
        plt.legend(loc='upper left', fontsize=8)
        plt.title(ANGLE_TYPES[type_index])

    plot_file = os.path.join(FLAGS.data_dir, FLAGS.output_plot_file)
    plt.savefig(plot_file)
    # os.system('open ' + plot_file)


if __name__ == '__main__':
    FLAGS(sys.argv)

    image_wildcard = os.path.join(FLAGS.data_dir, FLAGS.images)
    file_prefix, min_frame, max_frame = plot_utils.read_images(image_wildcard)
    print('File prefix: {}'.format(file_prefix))
    print('Frame range: [{}, {}]'.format(min_frame, max_frame))


    # Read camera param data.
    data_name_list = []
    data_dir_list =[]
    data_x_list = []
    data_y_list = []

    if FLAGS.param_data_names != '':
        data_name_list = FLAGS.param_data_names.split(',')
        data_dir_list = FLAGS.param_data_dirs.split(',')
        assert (len(data_name_list) == len(data_dir_list))

    for i in range(len(data_dir_list)):
        data_path = os.path.join(FLAGS.data_dir, data_dir_list[i])
        x_values, y_values = plot_utils.read_frame_values(
            data_path, file_prefix, min_frame, max_frame)

        data_x_list.append(x_values)
        data_y_list.append(y_values)
        print("Loaded '{}'.".format(data_path))


    # Read ConvNet outputs if exist.
    if FLAGS.convnet_dir:
        convnet_path = os.path.join(FLAGS.data_dir, FLAGS.convnet_dir)
        cn_pred_x, cn_pred_y, cn_pred_z, cv_loaded_frame_indices = \
            plot_utils.read_convnet_preds(
                convnet_path, file_prefix, min_frame, max_frame)
        print("Loaded '{}'.".format(convnet_path))

        # print('Compute max score curves...')
        # cn_max_score_x, cn_max_score_y = plot_utils.compute_max_score_curve(
        #         cn_pred_z, min_frame, max_frame, cv_loaded_frame_indices)
        # data_name_list.append('ConvNetMaxScore')
        # data_x_list.append(cn_max_score_x)
        # data_y_list.append(cn_max_score_y)
        # print('Done.')
        #
        # if FLAGS.output_convnet_max_score:
        #     output_path = os.path.join(
        #         FLAGS.data_dir, FLAGS.output_convnet_max_score)
        #     plot_utils.write_camera_params(
        #         output_path, file_prefix, cn_max_score_x, cn_max_score_y)
        #     print("Loaded '{}'.".format(output_path))


        print('Compute seam fitting curves...')
        cn_seam_fitting_x, cn_seam_fitting_y =\
            plot_utils.compute_seam_fitting_curve(
            cn_pred_z, min_frame, max_frame, cv_loaded_frame_indices)
        data_name_list.append('ConvNetSeamFitting')
        data_x_list.append(cn_seam_fitting_x)
        data_y_list.append(cn_seam_fitting_y)
        print('Done.')

        if FLAGS.output_convnet_seam_fitting:
            output_path = os.path.join(
                FLAGS.data_dir, FLAGS.output_convnet_seam_fitting)
            plot_utils.write_frame_values(
                output_path, file_prefix, cn_seam_fitting_x, cn_seam_fitting_y)
            print("Saved '{}'.".format(output_path))


    print('Draw plots...')
    if FLAGS.convnet_dir:
        plot_data(data_name_list, data_x_list, data_y_list,
                  cn_pred_x, cn_pred_y, cn_pred_z)
    else:
        plot_data(data_name_list, data_x_list, data_y_list)
    print('Done.')
