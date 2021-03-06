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


import os, sys
BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../../'))
sys.path.append(os.path.join(BASE_DIR, 'script'))

# Python 2/3 compatibility
from __future__ import print_function

import gflags
import glob
#from numba import jit
# @mhsung
import image_list
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import time
from filterpy.kalman import KalmanFilter


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('images', '', '')
gflags.DEFINE_string('bbox_dir', '', '')
gflags.DEFINE_string('output_file', '', '')
gflags.DEFINE_bool('display', True, '')


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
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        # Give high uncertainty to the unobservable initial velocities.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
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


    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))


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


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
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
        is_matched = (iou_matrix[m[0], m[1]] < iou_threshold)

        # @mhsung
        if (detections.shape[1] >= 6 and trackers.shape[1] >= 6):
            # det: [x0, y0, x1, y2, score, class_index, ...]
            # If class indices are given, bboxes with the same class index are
            # only matched.
            det_cls = detections[m[0], 5]
            trk_cls = trackers[m[0], 5]
            is_matched = is_matched and (det_cls == trk_cls)

        if is_matched:
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
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
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
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
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
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


# @mhsung
# output: [frame, x1, y1, x2, y2, score]
def read_dets(dir, image_filenames):
    seq_dets = np.ndarray(shape=(0, 6))

    for frame, image_filename in enumerate(image_filenames):
        image_name = os.path.splitext(image_filename)[0]
        bbox_files = glob.glob(os.path.join(dir, image_name, '*_bbox.txt'))
        for bbox_file in bbox_files:
            dets = np.loadtxt(bbox_file, delimiter=' ')
            dets = np.insert(dets, 0, frame)
            seq_dets = np.vstack([seq_dets, dets])

    return seq_dets


if __name__ == '__main__':
    FLAGS(sys.argv)

    # @mhsung
    image_filenames = image_list.get_image_filenames(FLAGS.images)
    seq_dets = read_dets(FLAGS.bbox_dir, image_filenames)

    total_time = 0.0
    total_frames = 0

    if FLAGS.display:
        colours = np.random.rand(32, 3)
        plt.ion()
        fig = plt.figure()

    if not os.path.exists(os.path.dirname(FLAGS.output_file)):
        os.makedirs(os.path.dirname(FLAGS.output_file))

    # Create instance of the SORT tracker
    mot_tracker = Sort()

    with open(FLAGS.output_file, 'w') as out_file:
        for frame, image_filename in enumerate(image_filenames):
            dets = seq_dets[seq_dets[:, 0] == frame, 1:6]
            total_frames += 1

            if FLAGS.display:
                ax1 = fig.add_subplot(111, aspect='equal')
                image_file = os.path.join(
                    os.path.dirname(FLAGS.images), image_filename)
                im = io.imread(image_file)
                ax1.imshow(im)
                plt.title('Tracked Targets')

            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            for d in trackers:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                      (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                      file=out_file)
                if FLAGS.display:
                    d = d.astype(np.uint32)
                    ax1.add_patch(patches.Rectangle(
                        (d[0], d[1]), d[2] - d[0], d[3] - d[1],
                        fill=False, lw=3, ec=colours[d[4] % 32, :]))
                    ax1.set_adjustable('box-forced')

            if FLAGS.display:
                fig.canvas.flush_events()
                #plt.draw()
                plt.pause(0.0001)
                ax1.cla()


    print("Total Tracking took: %.3f for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))

    if FLAGS.display:
        print("Note: to get real runtime results run without the option: "
              "--display")
