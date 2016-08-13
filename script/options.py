#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import gflags


def initialize():
    gflags.DEFINE_bool('clean', False, '') # Override 'overwrite'
    gflags.DEFINE_bool('overwrite', False, '')
    gflags.DEFINE_integer('num_threads', 1, '')

    gflags.DEFINE_bool('every_10', False, '')
    gflags.DEFINE_integer('seq_range', 0, '') # '<= 0' indicates all pair matches.

    gflags.DEFINE_bool('track_features', False, 'track')
    gflags.DEFINE_bool('less_num_inliers', False, 'lni')
    gflags.DEFINE_bool('less_sampson_error', False, 'lse')
    gflags.DEFINE_bool('no_two_view_bundle', False, 'ntb')
    gflags.DEFINE_bool('no_only_symmetric', False, 'nos')
    gflags.DEFINE_bool('use_initial_orientations', False, 'uio')


def show(FLAGS, PATHS):
    print('== Options ==')
    if FLAGS.overwrite: print('Overwrite: On')

    if FLAGS.every_10: print('Every 10 frame: On')
    if FLAGS.seq_range > 0: print('Sequence range: ' + str(FLAGS.seq_range))
    if FLAGS.track_features: print('Track features: On (OpenCV Lukas-Kanade)')

    if FLAGS.less_num_inliers:
        print('--min_num_inliers_for_valid_match=10')
    if FLAGS.less_sampson_error:
        print('--max_sampson_error_for_verified_match=10.0')
    if FLAGS.no_two_view_bundle:
        print('--bundle_adjust_two_view_geometry=false')
    if FLAGS.no_only_symmetric:
        print('--keep_only_symmetric_matches=false')
    if FLAGS.use_initial_orientations:
        print('--initial_orientations_data_type=' +
              FLAGS.ground_truth_type)
        print('--initial_orientations_filepath=' +
              PATHS.ground_truth_path)
    print('')
