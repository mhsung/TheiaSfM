#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.ground_truth_reconstruction_path):
        os.remove(PATHS.ground_truth_reconstruction_path)
        print("Removed '" + PATHS.ground_truth_reconstruction_path + "'.")

    if os.path.exists(PATHS.ground_truth_bbox_path):
        os.remove(PATHS.ground_truth_bbox_path)
        print("Removed '" + PATHS.ground_truth_bbox_path + "'.")

    if os.path.exists(PATHS.ground_truth_orientation_path):
        os.remove(PATHS.ground_truth_orientation_path)
        print("Removed '" + PATHS.ground_truth_orientation_path + "'.")


def run(FLAGS, PATHS):
    if not os.path.exists(PATHS.ground_truth_pose_path):
        print("Warning: ground truth does not exist : '"
              + PATHS.ground_truth_pose_path + "'")
        return

    print('== Evaluate neural net outputs ==')
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

    # Evaluate neural network outputs and compute ground truth object
    # information.
    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_evaluate_neural_net_outputs' + ' \\\n'

    cmd += '--calibration_file=' + PATHS.calibration_file + ' \\\n'
    cmd += '--ground_truth_data_type=reconstruction' + ' \\\n'
    cmd += '--ground_truth_filepath=' + \
           PATHS.ground_truth_reconstruction_path +' \\\n'
    cmd += '--bounding_boxes_filepath=' + PATHS.init_bbox_path +' \\\n'
    cmd += '--orientations_filepath=' + PATHS.init_orientation_path +' \\\n'
    cmd += '--out_fitted_bounding_boxes_filepath=' + \
           PATHS.ground_truth_bbox_path +' \\\n'
    cmd += '--out_fitted_orientations_filepath=' + \
           PATHS.ground_truth_orientation_path +' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(cmd, os.path.join(
        PATHS.script_path, 'create_ground_truth_objects.sh'))
