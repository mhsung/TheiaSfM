#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.matches_file):
        os.remove(PATHS.matches_file)
        print("Removed '" + PATHS.matches_file + "'.")


def run(FLAGS, PATHS):
    print('== Match features ==')
    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_match_features' + ' \\\n'

    cmd += '--num_threads=' + str(FLAGS.num_threads) + ' \\\n'
    cmd += '--input_features=' + PATHS.feature_wildcard + ' \\\n'
    cmd += '--calibration_file=' + PATHS.calibration_file + ' \\\n'
    cmd += '--matching_max_num_images_in_cache=128' + ' \\\n'
    cmd += '--output_matches_file=' + PATHS.matches_file + ' \\\n'

    if FLAGS.seq_range > 0:
        cmd += '--match_only_consecutive_pairs=true' + ' \\\n'
        cmd += '--consecutive_pair_frame_range=' + \
               str(FLAGS.seq_range) + ' \\\n'
    # if FLAGS.less_num_inliers:
    #     cmd += '--min_num_inliers_for_valid_match=10' + ' \\\n'
    # if FLAGS.less_sampson_error:
    #     cmd += '--max_sampson_error_for_verified_match=10.0' + ' \\\n'
    # if FLAGS.no_two_view_bundle:
    #     cmd += '--bundle_adjust_two_view_geometry=false' + ' \\\n'
    # if FLAGS.no_only_symmetric:
    #     cmd += '--keep_only_symmetric_matches=false' + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path + ' \\\n'
    cmd += '--alsologtostderr'
    run_cmd.save_and_run_cmd(cmd, os.path.join(PATHS.script_path, 'match.sh'))
