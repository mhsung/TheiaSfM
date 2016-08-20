#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os
import shutil


def clean_ground_truth_camera_param(FLAGS, PATHS):
    if os.path.exists(PATHS.ground_truth_camera_param_path):
        shutil.rmtree(PATHS.ground_truth_camera_param_path)
        print("Removed '" + PATHS.ground_truth_camera_param_path + "'.")


def convert_ground_truth(FLAGS, PATHS):
    print('== Convert ground truth orientation to camera params ==')
    if not os.path.isdir(PATHS.ground_truth_camera_param_path):
        os.makedirs(PATHS.ground_truth_camera_param_path)

    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_export_orientations' + ' \\\n'
    cmd += '--input_data_type=' + FLAGS.ground_truth_type + ' \\\n'
    cmd += '--input_filepath=' + PATHS.ground_truth_path + ' \\\n'
    cmd += '--output_filepath=' + PATHS.ground_truth_camera_param_path + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(
        cmd, os.path.join(PATHS.script_path, 'convert_ground_truth.sh'))


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.orientation_path):
        shutil.rmtree(PATHS.orientation_path)
        print("Removed '" + PATHS.orientation_path + "'.")


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
