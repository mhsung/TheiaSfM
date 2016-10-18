#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../'))
sys.path.append(os.path.join(BASE_DIR, 'script'))
sys.path.append(os.path.join(BASE_DIR, '3rdparty', 'py-faster-rcnn', 'lib'))

from utils.timer import Timer
import cnn_utils
import gflags
import glob
import image_list
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('bbox_file', 'convnet/object_bboxes.csv', '')
gflags.DEFINE_string('orientation_score_dir', 'convnet/object_score', '')
gflags.DEFINE_string('out_plot_dir', 'convnet/object_plot', '')
gflags.DEFINE_string('out_fitted_orientation_file',
    'convnet/object_orientations_fitted.csv', '')

gflags.DEFINE_bool('with_object_index', True, '')
gflags.DEFINE_bool('plot_data', False, '')

ANGLE_TYPES = ['Azimuth', 'Elevation', 'Theta']
NUM_ANGLE_TYPES = len(ANGLE_TYPES)
NUM_ANGLE_SAMPLES = 360


def read_convnet_scores(im_names, df, num_digits):
    num_frames = len(im_names)

    # X: frames.
    # [[0, 0, 0, ..., 0],
    #  [1, 1, 1, ..., 1],
    #  ...]
    x_values = np.tile(np.arange(num_frames),
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

    bbox_idxs = np.full(num_frames, np.nan, dtype=np.int)
    loaded_frames = []

    for frame, im_name in enumerate(im_names):
        bbox_df = df[df['image_name'] == im_name]
        if len(bbox_df.index) == 0:
            continue

        # There must be at most one object bounding box.
        assert (len(bbox_df.index) <= 1)
        bbox_idx = bbox_df.index[0]
        bbox_idxs[frame] = bbox_idx
        score_file = os.path.join(FLAGS.data_dir,
                FLAGS.orientation_score_dir,
                str(bbox_idx).zfill(num_digits) + '.npy')
        preds = np.load(score_file)
        assert (preds.shape[0] == NUM_ANGLE_TYPES)
        assert (preds.shape[1] == NUM_ANGLE_SAMPLES)
        loaded_frames.append(frame)

        for type_index in range(NUM_ANGLE_TYPES):
            z_values[type_index][frame, :] = preds[type_index, :]

    return x_values, y_values, z_values, bbox_idxs, loaded_frames


def plot_data(x_values, y_values, z_values, fitted_x, fitted_y, out_file):
    fig = plt.figure(figsize=(24, 12))

    for type_index in range(NUM_ANGLE_TYPES):
        plt.subplot(3, 1, type_index + 1)

        # Determine Y range shift using the first data.
        shift_y_range = False
        shift_y_range = plot_utils.good_to_shift_angle_range(
                fitted_y[:, type_index])

        y_lim = np.array([0, 360])
        if shift_y_range:
            y_lim += 180

        # Draw color map.
        x = x_values
        y = y_values
        z = z_values[type_index]
        if shift_y_range:
            y = np.hstack((y[:, 180:], y[:, :180] + 360))
            z = np.hstack((z[:, 180:], z[:, :180]))
        plt.pcolor(x, y, z, cmap='Oranges', vmin=0)
        plt.colorbar()

        # Draw fitted values.
        x = fitted_x
        y = fitted_y[:, type_index]
        if shift_y_range:
            y[y < 180] += 360
        plt.plot(x, y, '.-', label='Fitted', color='k')

        plt.ylim(y_lim)
        plt.xlabel("Frame Index")
        plt.ylabel("Angle (Degree)")
        plt.legend(loc='upper left', fontsize=8)
        plt.title(ANGLE_TYPES[type_index])

    plt.savefig(out_file)


if __name__ == '__main__':
    FLAGS(sys.argv)

    # 'object_index' must be given.
    assert(FLAGS.with_object_index)

    # Read image names.
    im_names = image_list.get_image_filenames(
            os.path.join(FLAGS.data_dir, 'images', '*.png'))
    num_frames = len(im_names)

    # Read bounding boxes.
    df, num_digits = cnn_utils.read_bboxes(
        os.path.join(FLAGS.data_dir, FLAGS.bbox_file),
        FLAGS.with_object_index)

    object_idxs = np.unique(df['object_index'])

    if not os.path.exists(os.path.join(
            FLAGS.data_dir, FLAGS.out_plot_dir)):
        os.makedirs(os.path.join(
            FLAGS.data_dir, FLAGS.out_plot_dir))

    # Store the minimum score at the end.
    fitted_preds = np.empty([len(df.index), len(ANGLE_TYPES) + 1])
    fitted_preds.fill(np.nan)

    count_objects = 0
    for object_idx in object_idxs:
        object_df = df[df['object_index'] == object_idx]
        num_object_bboxes = len(object_df.index)
        print ('Object ID {}: {} bbox(es).'.format(
            object_idx, num_object_bboxes))
        count_objects = count_objects + 1

        # Read scores.
        timer = Timer()
        timer.tic()
        x_values, y_values, z_values, bbox_idxs, loaded_frames = \
                read_convnet_scores(im_names, object_df, num_digits)
        timer.toc()
        print ('Reading scores took {:.3f}s for {:d} bboxes').format(
                timer.total_time, len(object_df.index))

        # Do seam fitting.
        timer = Timer()
        timer.tic()
        fitted_x = np.arange(num_frames)
        fitted_y = np.full((num_frames, NUM_ANGLE_TYPES), np.nan,
                           dtype=np.float)
        fitted_z = np.full((num_frames, NUM_ANGLE_TYPES), np.nan,
                           dtype=np.float)
        frame_range = range(min(loaded_frames), max(loaded_frames) + 1)

        for type_index in range(NUM_ANGLE_TYPES):
            angles = plot_utils.compute_seam_fitting_angles(
                    z_values[type_index][frame_range, :])
            fitted_y[frame_range, type_index] = angles
            fitted_z[frame_range, type_index] = z_values[type_index][
                frame_range, angles]
        timer.toc()
        print ('Seam fitting took {:.3f}s for {:d} bboxes').format(
                timer.total_time, len(object_df.index))

        if FLAGS.plot_data:
            # Plot result.
            timer = Timer()
            timer.tic()
            plot_file = os.path.join(
                    FLAGS.data_dir, FLAGS.out_plot_dir,
                    str(object_idx) + '.png')
            plot_data(x_values, y_values, z_values, fitted_x, fitted_y,
                    plot_file)
            timer.toc()
            print ('Drawing plot took {:.3f}s for {:d} bboxes').format(
                    timer.total_time, len(object_df.index))

        # Store fitted orientations.
        for frame in loaded_frames:
            pred = fitted_y[frame, :]
            bbox_idx = bbox_idxs[frame]
            assert (not np.any(np.isnan(pred)))
            assert (np.any(np.isnan(fitted_preds[bbox_idx, :])))
            fitted_preds[bbox_idx, :NUM_ANGLE_TYPES] = pred

            # Store minimum score.
            scores = fitted_z[frame, :]
            fitted_preds[bbox_idx, -1] = np.min(scores)

    assert (not np.any(np.isnan(fitted_preds)))

    # Save results.
    np.savetxt(os.path.join(FLAGS.data_dir, FLAGS.out_fitted_orientation_file),
               fitted_preds, fmt='%i,%i,%i,%f')

    print ('{} object(s) are tracked.'.format(count_objects))

