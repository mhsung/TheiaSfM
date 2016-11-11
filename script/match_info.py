#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os
import shutil


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.matches_info_path):
        shutil.rmtree(PATHS.matches_info_path)
        print("Removed '" + PATHS.matches_info_path + "'.")


def run(FLAGS, PATHS):
    print('== Extract match information ==')
    if not os.path.isdir(PATHS.matches_info_path):
        os.makedirs(PATHS.matches_info_path)

    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_extract_match_info' + ' \\\n'
    cmd += '--images_dir=' + PATHS.image_path + ' \\\n'
    cmd += '--matches_file=' + PATHS.matches_file + ' \\\n'
    cmd += '--output_dir=' + PATHS.matches_info_path + ' \\\n'
    cmd += '--ground_truth_data_type=reconstruction \\\n'
    cmd += '--ground_truth_filepath=' + PATHS.ground_truth_reconstruction_path + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(
        cmd, os.path.join(PATHS.script_path, 'match_info.sh'))
