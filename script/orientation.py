#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os

def run(FLAGS, PATHS):
    print('== Compare orientation with ground truth ==')
    if not os.path.isdir(PATHS.matches_info_path):
        os.makedirs(PATHS.matches_info_path)

    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_compare_orientations' + ' \\\n'
    cmd += '--estimate_data_type=reconstruction' + ' \\\n'
    cmd += '--estimate_filepath=' + PATHS.reconstruction_file + '-0' + ' \\\n'
    cmd += '--reference_data_type=' + FLAGS.ground_truth_type + ' \\\n'
    cmd += '--reference_filepath=' + PATHS.ground_truth_path + ' \\\n'
    cmd += '--output_filepath=' + PATHS.orientation_path + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(
        cmd, os.path.join(PATHS.script_path, 'orientation.sh'))