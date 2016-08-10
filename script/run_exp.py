#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

from collections import namedtuple
import gflags
import os
import sys

import calibration
import feature
import match
import match_info
import options
import orientation
import reconstruction


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir', '', '')

gflags.DEFINE_string('ground_truth_type', 'pose', '')
gflags.DEFINE_string('ground_truth_filename', 'pose', '')

gflags.DEFINE_string('bin_dir', os.path.join('..', 'build', 'bin'), '')


PATHS = namedtuple('PATHS', [
    'image_path',
    'feature_path',
    'image_wildcard',
    'feature_wildcard',
    'output_path',
    'log_path',
    'script_path',
    'calibration_file',
    'matches_file',
    'matches_info_path',
    'reconstruction_file',
    'ground_truth_path',
    'orientation_path'])


def set_output_name():
    output_name = 'sfm'
    if FLAGS.every_10:      output_name += '_10'
    if FLAGS.seq_range > 0: output_name += ('_seq_' + str(FLAGS.seq_range))

    if FLAGS.less_num_inliers:          output_name += '_lni'
    if FLAGS.less_sampson_error:        output_name += '_lse'
    if FLAGS.no_two_view_bundle:        output_name += '_ntb'
    if FLAGS.no_only_symmetric:         output_name += '_nos'
    if FLAGS.use_initial_orientations:  output_name += '_uio'
    return output_name


def set_paths():
    # Image path.
    PATHS.image_path = os.path.join(FLAGS.data_dir, 'images')
    if not os.path.isdir(PATHS.image_path):
        print('Image directory does not exist: "' + PATHS.image_path + '"')
        exit(-1)

    # Feature path.
    PATHS.feature_path = os.path.join(FLAGS.data_dir, 'features')
    if not os.path.isdir(PATHS.feature_path):
        os.makedirs(PATHS.feature_path)

    if FLAGS.every_10:
        PATHS.image_wildcard = os.path.join(PATHS.image_path, '*0.png')
        PATHS.feature_wildcard =\
            os.path.join(PATHS.feature_path, '*0.png.features')
    else:
        PATHS.image_wildcard = os.path.join(PATHS.image_path, '*.png')
        PATHS.feature_wildcard =\
            os.path.join(PATHS.feature_path, '*.png.features')

    # Output path.
    PATHS.output_path = os.path.join(FLAGS.data_dir, set_output_name())
    if not os.path.isdir(PATHS.output_path):
        os.makedirs(PATHS.output_path)

    PATHS.log_path = os.path.join(PATHS.output_path, 'log')
    if not os.path.isdir(PATHS.log_path):
        os.makedirs(PATHS.log_path)

    PATHS.script_path = os.path.join(PATHS.output_path, 'script')
    if not os.path.isdir(PATHS.script_path):
        os.makedirs(PATHS.script_path)

    # Output files.
    PATHS.calibration_file = os.path.join(FLAGS.data_dir, 'calibration.txt')
    PATHS.matches_file = os.path.join(PATHS.output_path, 'matches.bin')
    PATHS.matches_info_path = os.path.join(PATHS.output_path, 'matches')
    PATHS.reconstruction_file = os.path.join(PATHS.output_path, 'output')

    if not FLAGS.ground_truth_filename.isspace():
        PATHS.ground_truth_path =\
            os.path.join(FLAGS.data_dir, FLAGS.ground_truth_filename)
        PATHS.orientation_path =\
            os.path.join(PATHS.output_path, 'orientation')


def print_paths():
    print('== Paths ==')
    print('Image files: ' + PATHS.image_wildcard)
    print('Calibrations File: ' + PATHS.calibration_file)
    print('Features files: ' + PATHS.feature_wildcard)
    print('Matches file: ' + PATHS.matches_file)
    print('Matches info directory: ' + PATHS.matches_info_path)
    print('Reconstruction file: ' + PATHS.reconstruction_file)
    print('')


if __name__ == '__main__':
    # This must be called before 'FLAGS(sys.argv)'.
    options.initialize()
    FLAGS(sys.argv)
    FLAGS.bin_dir = os.path.normpath(os.path.join(sys.path[0], FLAGS.bin_dir))

    set_paths()
    print_paths()
    options.show(FLAGS)

    exist_calibration = os.path.exists(PATHS.calibration_file)
    if FLAGS.overwrite or (not exist_calibration):
        calibration.run(FLAGS, PATHS)

    exist_features = os.listdir(PATHS.feature_path)
    if FLAGS.overwrite or (not exist_features):
        feature.run(FLAGS, PATHS)

    exist_matches = os.path.exists(PATHS.matches_file)
    if FLAGS.overwrite or (not exist_matches):
        match.run(FLAGS, PATHS)

    exist_reconstruction = os.path.exists(PATHS.reconstruction_file + '-0')
    if FLAGS.overwrite or (not exist_reconstruction):
        reconstruction.run(FLAGS, PATHS)

    if not PATHS.ground_truth_path.isspace():
        match_info.run(FLAGS, PATHS)
        orientation.run(FLAGS, PATHS)

