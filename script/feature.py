#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os
import shutil


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.feature_path):
        shutil.rmtree(PATHS.feature_path)
        print("Removed '" + PATHS.feature_path + "'.")


def run(FLAGS, PATHS):
    print('== Extract features ==')
    if not os.path.isdir(PATHS.feature_path):
        os.makedirs(PATHS.feature_path)

    cmd = ''
    cmd += FLAGS.bin_dir + '/extract_features' + ' \\\n'
    cmd += '--num_threads=' + str(FLAGS.num_threads) + ' \\\n'
    cmd += '--input_images=' + PATHS.image_wildcard + ' \\\n'
    cmd += '--features_output_directory=' + PATHS.feature_path + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(PATHS.script_path, 'feature.sh'))