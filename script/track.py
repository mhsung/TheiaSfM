#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os
import shutil


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.feature_track_info_path):
        os.remove(PATHS.feature_track_info_path)
        print("Removed '" + PATHS.feature_track_info_path + "'.")

    if os.path.exists(PATHS.feature_track_path):
        shutil.rmtree(PATHS.feature_track_path)
        print("Removed '" + PATHS.feature_track_path + "'.")


def track_features(FLAGS, PATHS):
    print('== Track features ==')
    cmd = ''
    cmd += FLAGS.bin_dir + '/../../3rdparty/opencv/lk_track_sequence.py' +\
           ' \\\n'
    cmd += '--images=' + PATHS.image_wildcard + ' \\\n'
    cmd += '--output_image_dir=' + PATHS.feature_track_path + ' \\\n'
    cmd += '--output_feature_tracks_file=' + PATHS.feature_track_info_path

    #cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(
        PATHS.script_path, 'track_features.sh'))


def extract_matches(FLAGS, PATHS):
    print('== Extract matches ==')
    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_convert_feature_track_file' + ' \\\n'

    #cmd += '--num_threads=' + str(FLAGS.num_threads) + ' \\\n'
    cmd += '--images=' + PATHS.image_wildcard + ' \\\n'
    cmd += '--calibration_file=' + PATHS.calibration_file + ' \\\n'
    cmd += '--feature_tracks_file=' + PATHS.feature_track_info_path + ' \\\n'
    cmd += '--output_matches_file=' + PATHS.matches_file + ' \\\n'

    if FLAGS.seq_range > 0:
        print("Warning: 'seq_range' option is not supported "
              "when tracking features.")
        raw_input('Press Enter to continue...')
    if FLAGS.less_num_inliers:
        cmd += '--min_num_inliers_for_valid_match=10' + ' \\\n'
    if FLAGS.less_sampson_error:
        cmd += '--max_sampson_error_for_verified_match=10.0' + ' \\\n'
    if FLAGS.no_two_view_bundle:
        cmd += '--bundle_adjust_two_view_geometry=false' + ' \\\n'
    if FLAGS.no_only_symmetric:
        print("Warning: 'no_only_symmetric' option is not supported "
              "when tracking features.")
        raw_input('Press Enter to continue...')
    if FLAGS.use_initial_orientations:
        cmd += '--initial_orientations_data_type=' + \
               FLAGS.ground_truth_type + ' \\\n'
        cmd += '--initial_orientations_filepath=' + \
               PATHS.ground_truth_path + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(
        PATHS.script_path, 'extract_matches.sh'))
