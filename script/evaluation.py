#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os


def clean(FLAGS, PATHS):
    # To Do.
    return


def run(FLAGS, PATHS):
    if not os.path.exists(PATHS.ground_truth_pose_path):
        print("Warning: ground truth does not exist : '"
              + PATHS.ground_truth_pose_path + "'")
        return

    # Create ground truth reconstruction.
    print('== Compare output with ground truth ==')
    if not os.path.exists(PATHS.ground_truth_reconstruction_path):
        cmd = ''
        cmd += FLAGS.bin_dir + '/exp_create_reconstruction_from_modelviews'\
               + ' \\\n'

        cmd += '--images=' + PATHS.image_path + ' \\\n'
        cmd += '--data_type=modelview \\\n'
        cmd += '--file_path=' + PATHS.ground_truth_pose_path + ' \\\n'
        cmd += '--calibration_file=' + PATHS.calibration_file + ' \\\n'
        cmd += '--output_reconstruction='\
               + PATHS.ground_truth_reconstruction_path + ' \\\n'

        cmd += '--log_dir=' + PATHS.log_path
        run_cmd.save_and_run_cmd(cmd, os.path.join(
            PATHS.script_path, 'create_ground_truth_reconstruction.sh'))

    # Compare with ground truth reconstruction.
    cmd = ''
    cmd += FLAGS.bin_dir + '/compare_reconstructions' + ' \\\n'

    cmd += '--reference_reconstruction='\
           + PATHS.ground_truth_reconstruction_path + ' \\\n'
    cmd += '--reconstruction_to_align=' + PATHS.output_path + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(
        PATHS.script_path, 'evaluate.sh'))

    # Visualize reconstruction.
    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_view_cameras' + ' \\\n'

    cmd += '--data_type_list=reconstruction,reconstruction' + ' \\\n'
    cmd += '--filepath_list=' + PATHS.ground_truth_reconstruction_path\
           + ',' + PATHS.output_path + ' \\\n'
    cmd += '--calibration_file=' + PATHS.calibration_file + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(
        PATHS.script_path, 'evaluate.sh'))
