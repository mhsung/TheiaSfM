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


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir', '/Users/msung/Developer/data/7-scenes/office/seq-01/convnet/object_0', '')


def read_camera_params(data_dir, ref_y_value=None):
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

        if ref_y_value is None:
            ref_y_value = y_value

        if i > 0:
            # Add +- 360 to plot consistently with the reference values.
            # Use previous values if reference is not given.
            while y_value[0] > ref_y_value[0] + 180.0:
                y_value[0] = y_value[0] - 360.0
            while y_value[0] < ref_y_value[0] - 180.0:
                y_value[0] = y_value[0] + 360.0

            while y_value[2] > ref_y_value[2] + 180.0:
                y_value[2] = y_value[2] - 360.0
            while y_value[2] < ref_y_value[2] - 180.0:
                y_value[2] = y_value[2] + 360.0

        values[i, :] = np.hstack((x_value, y_value))

    return values


if __name__ == '__main__':
    FLAGS(sys.argv)

    data_names = [
        'GroundTruth',
        'TheiaSfM_Track',
        'TheiaSfM_Track_GT',
        'ConvNet',
    ]

    data_paths = [
        os.path.join(FLAGS.data_dir, 'camera_params'),
        os.path.join(FLAGS.data_dir, 'sfm_track', 'orientation'),
        os.path.join(FLAGS.data_dir, 'sfm_track_uio', 'orientation'),
        os.path.join(FLAGS.data_dir, 'output_params'),
    ]

    num_data = len(data_names)
    assert(num_data > 0)
    assert(len(data_paths) == num_data)

    data_values = []
    for data_index in range(num_data):
        data_path = data_paths[data_index]
        print(data_path)
        if data_index == 0:
            data_values.append(read_camera_params(data_path))
        else:
            # Use first data as reference values.
            data_values.append(read_camera_params(
                data_path, data_values[0][0, 1:]))


    # Draw plot.
    colors = cm.jet(np.linspace(0., 1., num_data))
    plot_names = ['Azimuth', 'Elevation', 'Theta']

    '''
    ##  TEST: Polynomial fitting ##
    fitfunc = lambda p, x:\
        p[0] +\
        p[1]*x +\
        p[2]*x*x +\
        p[3]*x*x*x +\
        p[4]*x*x*x*x +\
        p[5]*x*x*x*x*x
    errfunc = lambda p, x, y: y - fitfunc(p, x)

    data_index = 3
    for plot_index in range(len(plot_names)):
        x_values = data_values[data_index][:, 0]
        y_values = data_values[data_index][:, plot_index + 1]
        x0 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        res = optimize.least_squares(
            errfunc, x0,# loss='cauchy', f_scale=0.1,
            args=(x_values, y_values))
        data_values[data_index][:, plot_index + 1] = fitfunc(res.x, x_values)
    ####
    '''

    fig = plt.figure(figsize=(18, 6))
    for plot_index in range(len(plot_names)):
        plt.subplot(3, 1, plot_index + 1)

        for data_index in range(num_data):
            x_values = data_values[data_index][:, 0]
            y_values = data_values[data_index][:, plot_index + 1]
            plt.plot(x_values, y_values,
                     label=data_names[data_index], color=colors[data_index])

        ymin = data_values[0][:, plot_index + 1].min()
        ymax = data_values[0][:, plot_index + 1].max()
        plt.ylim([ymin - 20, ymax + 20])

        plt.xlabel("Frame Index")
        plt.ylabel("Angle (Degree)")
        plt.legend(loc='upper right', fontsize=8)
        plt.title(plot_names[plot_index])

    plot_file = os.path.join(FLAGS.data_dir, 'plot.png')
    plt.savefig(plot_file)
    #os.system('open ' + plot_file)
