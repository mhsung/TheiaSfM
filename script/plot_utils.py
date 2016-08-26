#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

from scipy import optimize

import gflags
import glob
import numpy as np
import os
import shutil
import sys


FLAGS = gflags.FLAGS
ANGLE_TYPES = ['Azimuth', 'Elevation', 'Theta']
NUM_ANGLE_TYPES = len(ANGLE_TYPES)
NUM_ANGLE_SAMPLES = 360


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
def read_frame_values(data_dir, file_prfix, min_frame, max_frame, sep=' '):
    num_frames = max_frame - min_frame + 1

    # X: frame indices.
    x_values = np.arange(min_frame, max_frame + 1)

    # Y: angle values.
    y_values = []

    filenames = [os.path.splitext(os.path.basename(x))[0] for x in
                 glob.glob(os.path.join(data_dir, file_prfix + '*.txt'))]

    # Sort by file names.
    filenames.sort()

    for filename in filenames:
        frame_str = filename[len(file_prfix):]
        assert (frame_str.isdigit())
        frame = int(frame_str)
        assert (frame >= min_frame and frame <= max_frame)

        filepath = os.path.join(data_dir, filename + '.txt')
        values = np.genfromtxt(filepath, delimiter=sep)

        frame_index = frame - min_frame

        if len(y_values) == 0:
            # Initialize y-values with the dimension.
            dim = len(values)
            y_values = np.full((num_frames, dim), np.nan)

        y_values[frame_index, :] = values

    return x_values, y_values


# Write camera params to files.
def write_frame_values(data_dir, file_prefix, x_values, y_values, sep=' '):
    assert (x_values.shape[0] == y_values.shape[0])

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in range(x_values.shape[0]):
        frame = x_values[i]
        values = y_values[i, :]
        if np.isnan(values).any():
            continue

        filepath = os.path.join(data_dir, file_prefix +
                                '{:04d}.txt'.format(frame))
        np.savetxt(filepath, values, delimiter=sep, newline=' ', fmt='%f')


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

    # Y: angle values.
    y_values = np.full((num_frames, NUM_ANGLE_TYPES), np.nan)

    for type_index in range(NUM_ANGLE_TYPES):
        angles = compute_max_score_angles(z_values[type_index])

        # Take values only for loaded frames.
        y_values[loaded_frame_indices, type_index] = \
            angles[loaded_frame_indices]

    return x_values, y_values


def compute_seam_fitting_curve(
        z_values, min_frame, max_frame, loaded_frame_indices):
    num_frames = max_frame - min_frame + 1

    # X: frames.
    x_values = np.arange(min_frame, max_frame + 1)

    # Y: angle values.
    y_values = np.full((num_frames, NUM_ANGLE_TYPES), np.nan)

    for type_index in range(NUM_ANGLE_TYPES):
        angles = compute_seam_fitting_angles(z_values[type_index])

        # Take values only for loaded frames.
        y_values[loaded_frame_indices, type_index] =\
            angles[loaded_frame_indices]

    return x_values, y_values


# Test whether angle range is smaller after shifting 180 degree.
def good_to_shift_angle_range(angles):
    angle_range = np.nanmax(angles) - np.nanmin(angles)

    shifted_angles = np.copy(angles)
    shifted_angles[shifted_angles < 180] += 360
    shifted_angle_range = np.nanmax(shifted_angles) - np.nanmin(shifted_angles)

    return (1.10 * shifted_angle_range < angle_range)


# Linearly interpolate NaN values by rows.
def linearly_interpolate_angles(values):
    interp_values = np.copy(values)
    is_not_nan = ~np.isnan(values).any(axis=1)
    not_nan_indices = np.nonzero(is_not_nan)[0]
    num_indices = values.shape[0]

    # Cannot interpolate if number of observed values are less than two.
    if len(not_nan_indices) < 2:
        return interp_values

    # NOTE:
    # Do not use numpy.interp() since values (angles) are 'circular'.
    for i in range(len(not_nan_indices) - 1):
        index1 = not_nan_indices[i]
        index2 = not_nan_indices[i + 1]
        assert (index1 < index2)

        y1 = np.copy(values[index1, :])
        y2 = np.copy(values[index2, :])
        for type_index in range(values.shape[1]):
            # Values (angles) are 'circular'.
            while y2[type_index] > y1[type_index] + 180.0:
                y2[type_index] -= 360.0
            while y2[type_index] < y1[type_index] - 180.0:
                y2[type_index] += 360.0

        for index in range(index1 + 1, index2):
            assert (~is_not_nan[index])
            w1 = float(index2 - index) / (index2 - index1)
            w2 = float(index - index1) / (index2 - index1)
            interp_values[index, :] = w1 * y1 + w2 * y2

    # Extrapolate at two end points
    # index1 = not_nan_indices[0]
    # index2 = not_nan_indices[1]
    # for index in range(index1):
    #     assert (~is_not_nan[index])
    #     w1 = float(index2 - index) / (index2 - index1)
    #     w2 = float(index - index1) / (index2 - index1)
    #     interp_values[index, :] = w1 * y1 + w2 * y2
    #
    # index1 = not_nan_indices[-2]
    # index2 = not_nan_indices[-1]
    # for index in range(index2 + 1, num_indices):
    #     assert (~is_not_nan[index])
    #     w1 = float(index2 - index) / (index2 - index1)
    #     w2 = float(index - index1) / (index2 - index1)
    #     interp_values[index, :] = w1 * y1 + w2 * y2

    return interp_values
