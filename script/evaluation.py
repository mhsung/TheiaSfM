#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.ground_truth_reconstruction_path):
        os.remove(PATHS.ground_truth_reconstruction_path)
        print("Removed '" + PATHS.ground_truth_reconstruction_path + "'.")

    if os.path.exists(PATHS.snapshot_path):
        os.remove(PATHS.snapshot_path)
        print("Removed '" + PATHS.snapshot_path + "'.")


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

        cmd += '--images=' + PATHS.image_wildcard + ' \\\n'
        cmd += '--data_type=modelview \\\n'
        cmd += '--filepath=' + PATHS.ground_truth_pose_path + ' \\\n'
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
    cmd += '--reconstruction_to_align=' + PATHS.reconstruction_file\
        + '-0' + ' \\\n'
    # FIXME:
    # Make the parameter as an option.
    cmd += '--robust_alignment_threshold=0.5' + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(
        PATHS.script_path, 'evaluate.sh'))

    # Visualize reconstruction.
    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_view_cameras' + ' \\\n'

    cmd += '--data_type_list=reconstruction,reconstruction' + ' \\\n'
    cmd += '--filepath_list=' + PATHS.ground_truth_reconstruction_path\
           + ',' + PATHS.reconstruction_file + '-0' +' \\\n'
    cmd += '--calibration_file=' + PATHS.calibration_file + ' \\\n'
    # FIXME:
    # Make the parameter as an option.
    cmd += '--robust_alignment_threshold=0.5' + ' \\\n'
    cmd += '--snapshot_file=' + PATHS.snapshot_path + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(
        PATHS.script_path, 'visualize.sh'))
