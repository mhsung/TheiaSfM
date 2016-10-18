#!/usr/bin/env python

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Python 2/3 compatibility
from __future__ import print_function

import os, sys
BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(os.path.join(BASE_DIR, 'script'))
sys.path.append(os.path.join(BASE_DIR, 'script', 'RenderForCNN'))

from filterpy.kalman import KalmanFilter
from sklearn.utils.linear_assignment_ import linear_assignment
import cnn_utils
import cv2
import gflags
import glob
#from numba import jit
# @mhsung
import image_list
import numpy as np
import pandas as pd
import time


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('raw_bbox_file', 'convnet/raw_bboxes.csv', '')
gflags.DEFINE_string('out_object_bbox_file', 'convnet/object_bboxes.csv', '')

gflags.DEFINE_integer('max_age', 10, '')
gflags.DEFINE_integer('min_hits', 3, '')
gflags.DEFINE_float('iou_threshold', 0.3, '')
gflags.DEFINE_float('measurement_noise', 1000.0, '')
gflags.DEFINE_bool('ignore_bbox_on_boundary', True, '')
gflags.DEFINE_integer('object_track_length_tol', 10, '')


#@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2].
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r
    is the aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / h
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the form [x,y,s,r] and returns it in the form
    [x1,y1,x2,x2] where x1,y1 is the top left and x2,y2 is the bottom right.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2.,
                         x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2.,
                         x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
        #                       [0, 1, 0, 0, 0, 1, 0],
        #                       [0, 0, 1, 0, 0, 0, 1],
        #                       [0, 0, 0, 1, 0, 0, 0],
        #                       [0, 0, 0, 0, 1, 0, 0],
        #                       [0, 0, 0, 0, 0, 1, 0],
        #                       [0, 0, 0, 0, 0, 0, 1]])
        # self.kf.H = np.array(
        #     [[1, 0, 0, 0, 0, 0, 0],
        #      [0, 1, 0, 0, 0, 0, 0],
        #      [0, 0, 1, 0, 0, 0, 0],
        #      [0, 0, 0, 1, 0, 0, 0]])
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0]])

        # self.kf.R[2:, 2:] *= 10.
        self.kf.R[:, :] *= FLAGS.measurement_noise
        # Give high uncertainty to the unobservable initial velocities.
        self.kf.P[4:, 4:] *= 10000.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # @mhsung
        self.score = 0
        self.class_idx = -1
        # bbox: [x0, y0, x1, y2, score, class_index, ...]
        # Store score and class index if given.
        if len(bbox) >= 5:
            self.score = bbox[4]
        if len(bbox) >= 6:
            self.class_idx = bbox[5]


    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

        # @mhsung
        # bbox: [x0, y0, x1, y2, score, class_index, ...]
        if len(bbox) >= 5:
            self.score = bbox[4]
        if len(bbox) >= 6:
            assert(self.class_idx == bbox[5])


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]


    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), \
               np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        is_unmatched = (iou_matrix[m[0], m[1]] < FLAGS.iou_threshold)

        # @mhsung
        # det: [x0, y0, x1, y2, score, class_index, ...]
        # If class indices are given, bboxes with the same class index are
        # only matched.
        if (detections.shape[1] >= 6 and trackers.shape[1] >= 6):
            det_cls = detections[m[0], 5]
            trk_cls = trackers[m[1], 5]
            is_unmatched = is_unmatched or (det_cls != trk_cls)

        if is_unmatched:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    """
    Sets key parameters for SORT
    """
    def __init__(self):
        self.max_age = FLAGS.max_age
        self.min_hits = FLAGS.min_hits
        self.trackers = []
        self.frame_count = 0


    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],
          [x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty
        detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of
        detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            score = self.trackers[t].score
            class_idx = self.trackers[t].class_idx
            trk[:] = [pos[0], pos[1], pos[2], pos[3], score, class_idx]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if ((trk.time_since_update < 1) and
                    (trk.hit_streak >= self.min_hits or
                             self.frame_count <= self.min_hits)):
                # @mhsung
                # Add class index if given.
                if trk.class_idx >= 0:
                    ret.append(np.concatenate(
                        (d, [trk.score, trk.id + 1, trk.class_idx])).reshape(
                        1, -1))
                else:
                    # +1 as MOT benchmark requires positive
                    ret.append(np.concatenate(
                        (d, [trk.score, trk.id + 1])).reshape(1, -1))

            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


# @mhsung
# output: [frame, x1, y1, x2, y2, score, class_index]
def read_dets(df, im_names):
    seq_dets = np.ndarray(shape=(0, 7))

    for frame, im_name in enumerate(im_names):
        subset_df = df[df['image_name'] == im_name]
        for _, row in subset_df.iterrows():
            det = np.array([
                frame, row['x1'], row['y1'], row['x2'], row['y2'],
                row['score'], row['class_index']])
            assert (row['class_index'] >= 0)
            seq_dets = np.vstack([seq_dets, det])

    return seq_dets


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read image file names.
    im_names = image_list.get_image_filenames(
        os.path.join(FLAGS.data_dir, 'images', '*.png'))

    # Read bounding boxes.
    df, _ = cnn_utils.read_bboxes(
        os.path.join(FLAGS.data_dir, FLAGS.raw_bbox_file),
        with_object_index=False)

    # Get bounding boxes in image sequence.
    seq_dets = read_dets(df, im_names)

    total_time = 0.0
    total_frames = 0

    # Create instance of the SORT tracker
    mot_tracker = Sort()

    object_df = cnn_utils.create_bbox_data_frame(with_object_index=True)

    for frame, im_name in enumerate(im_names):
        # Load the input image.
        im_file = os.path.join(FLAGS.data_dir, 'images', im_name)
        im = cv2.imread(im_file)
        im_size_x = im.shape[1]
        im_size_y = im.shape[0]

        dets = seq_dets[seq_dets[:, 0] == frame, 1:]
        total_frames += 1

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            # d: [x1, y1, x2, y2, score, track_index, class_index]
            assert (len(d) >= 6)
            if FLAGS.ignore_bbox_on_boundary:
                # Ignore bounding boxes on the frame boundary.
                if d[0] <= 0 or d[2] >= (im_size_x - 1) or \
                        d[1] <= 0 or d[3] >= (im_size_y - 1):
                    continue
            # Append a row.
            # ['image_name', 'class_index',
            #         'x1', 'y1', 'x2', 'y2', 'score', 'object_index')
            object_df.loc[len(object_df)] = [
                im_name, d[6], d[0], d[1], d[2], d[3], d[4], d[5]]

    # Remove short object tracks.
    object_idxs = object_df.object_index.unique()
    for object_idx in object_idxs:
        num_object_frames = len(object_df[
                object_df.object_index == object_idx])
        if num_object_frames < FLAGS.object_track_length_tol:
            print ('Remove short object track (id: {:d}, length: {:d})'.format(
                int(object_idx), num_object_frames))
            object_df = object_df[
                object_df.object_index != object_idx]

    # Reset bounding indices.
    object_df = object_df.reset_index(drop=True)

    out_file = os.path.join(FLAGS.data_dir, FLAGS.out_object_bbox_file)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # Save class indices as integer.
    object_df['class_index'] = object_df['class_index'].map(
            lambda x: '%i' % x)
    object_df['object_index'] = object_df['object_index'].map(
            lambda x: '%i' % x)
    object_df.to_csv(out_file, header=False)
    num_bboxes = len(object_df)
    print ('{:d} bounding box(es) are saved.'.format(num_bboxes))

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))

