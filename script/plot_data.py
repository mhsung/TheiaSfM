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


FLAGS = gflags.FLAGS
ANGLE_TYPES = ['Azimuth', 'Elevation', 'Theta']
NUM_ANGLE_TYPES = len(ANGLE_TYPES)
NUM_ANGLE_SAMPLES = 360


# Set input files.
gflags.DEFINE_string('data_dir', '/Users/msung/Developer/data/MVI_0206', '')
gflags.DEFINE_string('images', 'images/*.png', '')
gflags.DEFINE_string('param_data_names',
                     'Theia_10,Theia_track', '')
gflags.DEFINE_string('param_data_dirs',
                     'sfm_10/orientation,sfm_track/orientation', '')
gflags.DEFINE_string('convnet_dir', 'convnet/output_params', '')
gflags.DEFINE_string('output_plot_file', 'plot.png', '')


# Read image file prefix and min/max frame indices.
def read_images(data_wildcard):
    image_files = [os.path.splitext(os.path.basename(x))[0]
                   for x in glob.glob(data_wildcard)]
    file_prefix = os.path.commonprefix(image_files)
    min_frame = None
    max_frame = None

    for image_file in image_files:
        frame_str = image_file[len(file_prefix):]
        assert (frame_str.isdigit())
        frame = int(frame_str)
        min_frame = frame if min_frame is None else min(min_frame, frame)
        max_frame = frame if max_frame is None else max(max_frame, frame)

    assert (min_frame is not None)
    assert (max_frame is not None)

    return file_prefix, min_frame, max_frame


# Read camera params from files.
def read_camera_params(data_dir, file_prfix, min_frame, max_frame):
    # X: frame indices.
    x_values = np.zeros([0, 1])

    # Y: angle values.
    y_values = np.zeros([0, NUM_ANGLE_TYPES])

    filenames = [os.path.splitext(os.path.basename(x))[0] for x in
                 glob.glob(os.path.join(data_dir, file_prfix + '*.txt'))]

    # Sort by file names.
    filenames.sort()

    for filename in filenames:
        frame_str = filename[len(file_prfix):]
        assert (frame_str.isdigit())
        frame = int(frame_str)

        filepath = os.path.join(data_dir, filename + '.txt')
        angles = np.genfromtxt(filepath, delimiter=' ')
        x_values = np.vstack((x_values, frame))
        y_values = np.vstack((y_values, angles))

    return x_values, y_values


# Read ConvNet prediction distributions from files.
def read_convnet_preds(data_dir, file_prfix, min_frame, max_frame):
    num_frames = max_frame - min_frame + 1

    # X: frames.
    # [[0, 0, 0, ..., 0],
    #  [1, 1, 1, ..., 1],
    #  ...]
    x_values = np.tile(np.arange(min_frame, max_frame + 1),
                       (NUM_ANGLE_SAMPLES, 1)).transpose()

    # Y: angles.
    # [[0, 1, 2, ..., 359],
    #  [0, 1, 2, ..., 359],
    #  ...]
    y_values = np.tile(np.arange(NUM_ANGLE_SAMPLES), (num_frames, 1))

    # Z: ConvNet scores for all angle sample.
    z_values = []
    for type_index in range(NUM_ANGLE_TYPES):
        z_values.append(np.zeros([num_frames, NUM_ANGLE_SAMPLES]))

    loaded_frame_indices = []

    filenames = [os.path.splitext(os.path.basename(x))[0] for x in
                 glob.glob(os.path.join(data_dir, file_prfix + '*.npy'))]

    # Sort by file names.
    filenames.sort()

    for filename in filenames:
        frame_str = filename[len(file_prfix):]
        assert (frame_str.isdigit())
        frame = int(frame_str)

        filepath = os.path.join(data_dir, filename + '.npy')
        preds = np.load(filepath)
        assert (preds.shape[0] == NUM_ANGLE_TYPES)
        assert (preds.shape[1] == NUM_ANGLE_SAMPLES)

        frame_index = frame - min_frame
        loaded_frame_indices.append(frame_index)

        for type_index in range(NUM_ANGLE_TYPES):
            z_values[type_index][frame_index, :] = preds[type_index, :]

    return x_values, y_values, z_values, loaded_frame_indices


# Find the 'maximum' score angles for each frame.
# @scores: (frame_index, angle_index)
def compute_max_score_angles(scores):
    num_frames = scores.shape[0]
    num_indices = scores.shape[1]

    # FIXME:
    # Here we assumed that @angle_index equals angle value.
    assert (num_indices == NUM_ANGLE_SAMPLES)
    assert (NUM_ANGLE_SAMPLES == 360)

    best_angle_indices = np.argmax(scores, axis=1)
    return best_angle_indices


# Compute the 'maximum' score seam from ConvNet distribution.
# Use dynamic programming inspired by seam carving.
# @scores: (frame_index, angle_index)
def compute_seam_fitting_angles(scores, neighbor_range=10, distance_weight=0.5):
    num_frames = scores.shape[0]
    num_angles = scores.shape[1]

    # FIXME:
    # Here we assumed that @angle_index equals angle value.
    assert (num_angles == NUM_ANGLE_SAMPLES)
    assert (NUM_ANGLE_SAMPLES == 360)

    acc_scores = np.full((num_frames, num_angles), -np.inf)
    prev_best_angle_index = np.full((num_frames, num_angles), -1).astype(int)

    # Compute dynamic programming.
    for frame_index in range(num_frames):
        if frame_index == 0:
            acc_scores[frame_index, :] = scores[frame_index, :]
            continue

        # @i and @j are sample indices.
        for i in range(num_angles):
            # Start from @i itself and move to the farther neighbors so that
            # @i is chosen as the best previous index when scores are the same.
            sample_neighbors =\
                list(range(i, i - neighbor_range, -1)) +\
                list(range(i + 1, i + neighbor_range, +1))

            for j in sample_neighbors:
                distance_penalty = distance_weight * abs(j - i)

                # NOTE:
                # The angle index is 'circular'.
                j = j + num_angles if j < 0 else j
                j = j - num_angles if j >= num_angles else j

                acc_value_ij = acc_scores[frame_index - 1, j] +\
                               scores[frame_index, i]
                # Subtract distance penalty.
                acc_value_ij -= distance_penalty

                if acc_value_ij > acc_scores[frame_index, i]:
                    acc_scores[frame_index, i] = acc_value_ij
                    prev_best_angle_index[frame_index, i] = j

    best_angle_indices = np.full((num_frames, ), -1).astype(int)
    # Find the last index of the seam.
    best_angle_indices[num_frames - 1] =\
        np.argmax(acc_scores[num_frames - 1, :])

    # Back-track to the first index.
    for frame_index in range(num_frames - 2, -1, -1):
        next_angle_index = best_angle_indices[frame_index + 1]
        best_angle_indices[frame_index] =\
            prev_best_angle_index[frame_index + 1, next_angle_index]

    return best_angle_indices


def compute_max_score_curve(
        z_values, min_frame, max_frame, loaded_frame_indices):
    num_frames = max_frame - min_frame + 1

    # X: frames.
    x_values = np.arange(min_frame, max_frame + 1)

    y_values = np.zeros([num_frames, NUM_ANGLE_TYPES])
    for type_index in range(NUM_ANGLE_TYPES):
        y_values[:, type_index] = compute_max_score_angles(z_values[type_index])

    # Take values only for loaded frames.
    x_values = x_values[loaded_frame_indices]
    y_values = y_values[loaded_frame_indices]

    return x_values, y_values


def compute_seam_fitting_curve(
        z_values, min_frame, max_frame, loaded_frame_indices):
    num_frames = max_frame - min_frame + 1

    # X: frames.
    x_values = np.arange(min_frame, max_frame + 1)

    y_values = np.zeros([num_frames, NUM_ANGLE_TYPES])
    for type_index in range(NUM_ANGLE_TYPES):
        y_values[:, type_index] = \
            compute_seam_fitting_angles(z_values[type_index])

    # Take values only for loaded frames.
    x_values = x_values[loaded_frame_indices]
    y_values = y_values[loaded_frame_indices]

    return x_values, y_values


# Test whether angle range is smaller after shifting 180 degree.
def good_to_shift_angle_range(angles):
    angle_range = max(angles) - min(angles)

    shifted_angles = np.copy(angles)
    shifted_angles[shifted_angles < 180] += 360
    shifted_angle_range = max(shifted_angles) - min(shifted_angles)

    return (1.10 * shifted_angle_range < angle_range)


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
            shift_y_range = good_to_shift_angle_range(y_values)

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
            plt.plot(x_values, y_values, label=data_name_list[data_index],
                     color=colors[data_index])

        plt.ylim(y_lim)
        plt.xlabel("Frame Index")
        plt.ylabel("Angle (Degree)")
        plt.legend(loc='upper left', fontsize=8)
        plt.title(ANGLE_TYPES[type_index])

    plot_file = os.path.join(FLAGS.data_dir, FLAGS.output_plot_file)
    plt.savefig(plot_file)
    os.system('open ' + plot_file)


if __name__ == '__main__':
    FLAGS(sys.argv)

    image_wildcard = os.path.join(FLAGS.data_dir, FLAGS.images)
    file_prefix, min_frame, max_frame = read_images(image_wildcard)
    print('File prefix: {}'.format(file_prefix))
    print('Frame range: [{}, {}]'.format(min_frame, max_frame))


    # Read camera param data.
    data_name_list = []
    data_x_list = []
    data_y_list = []

    data_name_list = FLAGS.param_data_names.split(',')
    data_dir_list = FLAGS.param_data_dirs.split(',')
    assert (len(data_name_list) == len(data_dir_list))

    for i in range(len(data_dir_list)):
        data_path = os.path.join(FLAGS.data_dir, data_dir_list[i])
        x_values, y_values = read_camera_params(
            data_path, file_prefix, min_frame, max_frame)
        data_x_list.append(x_values)
        data_y_list.append(y_values)
        print("Loaded '{}'.".format(data_path))


    # Read ConvNet outputs if exist.
    if FLAGS.convnet_dir:
        convnet_path = os.path.join(FLAGS.data_dir, FLAGS.convnet_dir)
        cn_pred_x, cn_pred_y, cn_pred_z, cv_loaded_frame_indices =\
            read_convnet_preds(convnet_path, file_prefix, min_frame, max_frame)
        print("Loaded '{}'.".format(convnet_path))

        # print('Compute max score curves...')
        # cn_max_score_x, cn_max_score_y = compute_max_score_curve(
        #         cn_pred_z, min_frame, max_frame, cv_loaded_frame_indices)
        # data_name_list.append('ConvNetMaxScore')
        # data_x_list.append(cn_max_score_x)
        # data_y_list.append(cn_max_score_y)
        # print('Done.')

        print('Compute seam fitting curves...')
        cn_seam_fitting_x, cn_seam_fitting_y = compute_seam_fitting_curve(
            cn_pred_z, min_frame, max_frame, cv_loaded_frame_indices)
        data_name_list.append('ConvNetSeamFitting')
        data_x_list.append(cn_seam_fitting_x)
        data_y_list.append(cn_seam_fitting_y)
        print('Done.')


    print('Draw plots...')
    plot_data(data_name_list, data_x_list, data_y_list,
              cn_pred_x, cn_pred_y, cn_pred_z)
    print('Done.')