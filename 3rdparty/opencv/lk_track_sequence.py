#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../'))
sys.path.append(os.path.join(BASE_DIR, 'script'))

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import gflags
import glob
import image_list
# import video
# from common import anorm2, draw_str
# from time import clock


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('images', '', '')
gflags.DEFINE_string('output_image_dir', '', '')
gflags.DEFINE_string('output_image_list_file', '', '')
gflags.DEFINE_string('output_feature_tracks_file', '', '')
gflags.DEFINE_bool('display', False, '')


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# feature_params = dict( maxCorners = 500,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.1,
                       minDistance = 10,
                       blockSize = 3 )

class App:
    def __init__(self):
        # self.track_len = 10
        self.track_len = np.inf
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0


    def write_track(self, file, track, end_frame_idx):
        track_length = len(track)
        if track_length <= 1:
            return

        start_frame_index = end_frame_idx - track_length
        file.write('{} {} '.format(start_frame_index, track_length))
        for point in track:
            file.write('{} {} '.format(point[0], point[1]))
        file.write('\n')


    def run(self):
        # @mhsung
        if not os.path.exists(FLAGS.output_image_dir):
            os.makedirs(FLAGS.output_image_dir)

        # Read image file names.
        image_filenames = image_list.get_image_filenames(FLAGS.images)

        track_file = open(FLAGS.output_feature_tracks_file, 'w')
        assert(track_file)

        for image_filename in image_filenames:
            image_file = os.path.join(
                os.path.dirname(FLAGS.images), image_filename)

            #ret, frame = self.cam.read()
            im = cv2.imread(image_file)
            frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            vis = im.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        # @mhsung
                        # Save disconnected tracks.
                        self.write_track(track_file, tr, self.frame_idx)
                        continue

                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks

                # @mhsung
                # Do not draw track lines and number of tracks.
                # cv2.polylines(vis, [np.int32(tr) for tr in self.tracks],
                #               False, (0, 255, 0))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        # @mhsung
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)


            self.frame_idx += 1
            self.prev_gray = frame_gray
            if FLAGS.display:
                cv2.imshow('lk_track', vis)
            # else:
            #     log.info('{}: track count = {}'.format(
            #         image_name, len(self.tracks)))

            # @mhsung
            # output_file = os.path.join(
            #     FLAGS.output_image_dir, image_name + '.png')
            # cv2.imwrite(output_file, vis)

            # ch = 0xFF & cv2.waitKey(1)
            # if ch == 27:
            #     break

        track_file.close()


def main():
    FLAGS(sys.argv)

    #print(__doc__)
    App().run()
    if FLAGS.display:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
