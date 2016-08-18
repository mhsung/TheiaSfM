#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

from scipy import optimize

import gflags
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


FLAGS = gflags.FLAGS

# Set input files.
# gflags.DEFINE_string('data_dir', '/Users/msung/Developer/data/MVI_0206', '')
# gflags.DEFINE_integer('x_min', 1, '')
# gflags.DEFINE_integer('x_max', 3369, '')
gflags.DEFINE_string('data_dir', '/Users/msung/Developer/data/7-scenes/office/seq-01/convnet/object_0', '')
gflags.DEFINE_integer('x_min', 0, '')
gflags.DEFINE_integer('x_max', 1000, '')
gflags.DEFINE_integer('num_plots', 3, '')
gflags.DEFINE_integer('num_angle_samples', 360, '')


def read_preds(data_dir):
    filenames = glob.glob(os.path.join(data_dir, '*.npy'))
    num_files = len(filenames)

    # Sort by file names.
    filenames.sort()

    x_range = FLAGS.x_max - FLAGS.x_min
    all_z_values = []
    for plot_index in range(FLAGS.num_plots):
        all_z_values.append(np.zeros([x_range, FLAGS.num_angle_samples]))

    for i in range(num_files):
        filename = filenames[i]
        filepath = os.path.join(data_dir, filename)
        basename, fileext = os.path.splitext(filename)

        # NOTE: Frame index is number after the last '_'.
        #x_value = int(basename[basename.rfind('_')+1:])
        x_value = int(basename[basename.rfind('-') + 1:])
        z_value = np.load(filepath)
        for plot_index in range(FLAGS.num_plots):
            all_z_values[plot_index][x_value, :] = z_value[plot_index, :]


    all_x_values = np.tile(np.arange(FLAGS.x_min, FLAGS.x_max),
                           (FLAGS.num_angle_samples, 1)).transpose()
    all_y_values = np.tile(np.arange(FLAGS.num_angle_samples),
                           (x_range, 1))

    return all_x_values, all_y_values, all_z_values


def read_camera_params(data_dir):
    filenames = glob.glob(os.path.join(data_dir, '*.txt'))

    # Sort by file names.
    filenames.sort()

    num_files = len(filenames)
    values = np.zeros([num_files, 4])

    for i in range(num_files):
        filename = filenames[i]
        filepath = os.path.join(data_dir, filename)
        basename, fileext = os.path.splitext(filename)

        # NOTE: Frame index is number after the last '_'.
        #x_value = int(basename[basename.rfind('_')+1:])
        x_value = int(basename[basename.rfind('-') + 1:])
        y_value = np.genfromtxt(filepath, delimiter=' ')
        values[i, :] = np.hstack((x_value, y_value))

    return values


if __name__ == '__main__':
    FLAGS(sys.argv)

    # data_names = [
    #     'TheiaSfM_10',
    #     'TheiaSfM_Track',
    #     'ConvNet_Best',
    # ]
    #
    # data_paths = [
    #     os.path.join(FLAGS.data_dir, 'sfm_10', 'orientation'),
    #     os.path.join(FLAGS.data_dir, 'sfm_track', 'orientation'),
    #     os.path.join(FLAGS.data_dir, 'convnet', 'output_params'),
    # ]
    #
    # convnet_path = os.path.join(FLAGS.data_dir, 'convnet', 'output_params')

    data_names = [
        'GroundTruth',
        'TheiaSfM_Track',
        'TheiaSfM_Track_GT',
        'ConvNet_Best',
    ]

    data_paths = [
        os.path.join(FLAGS.data_dir, 'camera_params'),
        os.path.join(FLAGS.data_dir, 'sfm_track', 'orientation'),
        os.path.join(FLAGS.data_dir, 'sfm_track_uio', 'orientation'),
        os.path.join(FLAGS.data_dir, 'output_params'),
    ]

    convnet_path = os.path.join(FLAGS.data_dir, 'output_params')

    num_data = len(data_names)
    assert(num_data > 0)
    assert(len(data_paths) == num_data)

    data_values = []
    for data_index in range(num_data):
        data_path = data_paths[data_index]
        print(data_path)
        data_values.append(read_camera_params(data_path))

    (convnet_x, convnet_y, convnet_z) = read_preds(convnet_path)


    # Draw plot.
    colors = cm.jet(np.linspace(0., 1., num_data))
    plot_names = ['Azimuth', 'Elevation', 'Theta']

    fig = plt.figure(figsize=(24, 12))
    for plot_index in range(len(plot_names)):
        plt.subplot(3, 1, plot_index + 1)

        y_min = 0
        y_max = 360
        y_offset = 5

        x_values = convnet_x
        y_values = convnet_y
        z_values = convnet_z[plot_index]

        convnet_z_min = 0.0
        convnet_z_max = z_values.max()

        if plot_index == 1 or plot_index == 2:
            y_min += 180
            y_max += 180
            y_values = np.hstack((y_values[:, 180:], y_values[:, :180] + 360))
            z_values = np.hstack((z_values[:, 180:], z_values[:, :180]))

        plt.pcolor(x_values, y_values, z_values,
                   cmap='Oranges', vmin=convnet_z_min, vmax=convnet_z_max)

        for data_index in range(num_data):
            x_values = data_values[data_index][:, 0]
            y_values = data_values[data_index][:, plot_index + 1]

            if plot_index == 1 or plot_index == 2:
                y_values[y_values < 180] += 360

            plt.plot(x_values, y_values,
                     label=data_names[data_index], color=colors[data_index])
            #y_min = min(y_min, min(y_values))
            #y_max = max(y_max, max(y_values))

        plt.ylim([y_min - y_offset, y_max + y_offset])
        plt.xlabel("Frame Index")
        plt.ylabel("Angle (Degree)")
        plt.colorbar()
        plt.legend(loc='upper left', fontsize=8)
        plt.title(plot_names[plot_index])

    plot_file = os.path.join(FLAGS.data_dir, 'plot.png')
    plt.savefig(plot_file)
    os.system('open ' + plot_file)

