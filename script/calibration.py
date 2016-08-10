#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.calibration_file):
        os.remove(PATHS.calibration_file)
        print("Removed '" + PATHS.calibration_file + "'.")


def run(FLAGS, PATHS):
    print('== Create calibration file ==')
    cmd = ''
    cmd += FLAGS.bin_dir + '/create_calibration_file_from_exif' + ' \\\n'
    cmd += '--images=' + PATHS.image_wildcard + ' \\\n'
    cmd += '--initialize_uncalibrated_images_with_median_viewing_angle=true' \
           + ' \\\n'
    cmd += '--output_calibration_file=' + PATHS.calibration_file + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(
        cmd, os.path.join(PATHS.script_path, 'calibrattion.sh'))