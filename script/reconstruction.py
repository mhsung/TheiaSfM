#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os


def clean(FLAGS, PATHS):
    if os.path.exists(PATHS.reconstruction_file + '-0'):
        os.remove(PATHS.reconstruction_file + '-0')
        print("Removed '" + PATHS.reconstruction_file + '-0' + "'.")


def run(FLAGS, PATHS):
    print('== Build reconstruction ==')
    cmd = ''
    cmd += FLAGS.bin_dir + '/exp_build_reconstruction' + ' \\\n'
    cmd += '--num_threads=' + str(FLAGS.num_threads) + ' \\\n'
    cmd += '--matches_file=' + PATHS.matches_file + ' \\\n'
    cmd += '--match_out_of_core=false' + ' \\\n'
    cmd += '--calibration_file=' + PATHS.calibration_file + ' \\\n'
    cmd += '--output_reconstruction=' + PATHS.reconstruction_file + ' \\\n'

    ############### General SfM Options ###############
    cmd += '--reconstruction_estimator=GLOBAL' + ' \\\n'
    cmd += '--max_track_length=1000' + ' \\\n'
    cmd += '--reconstruct_largest_connected_component=true' + ' \\\n'
    cmd += '--only_calibrated_views=false' + ' \\\n'
    cmd += '--shared_calibration=true' + ' \\\n'

    ############### Global SfM Options ###############
    cmd += '--global_position_estimator=LEAST_UNSQUARED_DEVIATION' + ' \\\n'
    cmd += '--global_rotation_estimator=ROBUST_L1L2' + ' \\\n'
    cmd += '--post_rotation_filtering_degrees=20.0' + ' \\\n'
    cmd += '--refine_relative_translations_after_rotation_estimation=true' \
           + ' \\\n'
    cmd += '--extract_maximal_rigid_subgraph=false' + ' \\\n'
    cmd += '--filter_relative_translations_with_1dsfm=true' + ' \\\n'
    cmd += '--position_estimation_min_num_tracks_per_view=10' + ' \\\n'
    cmd += '--position_estimation_robust_loss_width=0.01' + ' \\\n'
    cmd += '--num_retriangulation_iterations=2' + ' \\\n'

    ############### Incremental SfM Options ###############
    cmd += '--absolute_pose_reprojection_error_threshold=10' + ' \\\n'
    cmd += '--partial_bundle_adjustment_num_views=20' + ' \\\n'
    cmd += '--full_bundle_adjustment_growth_percent=5' + ' \\\n'
    cmd += '--min_num_absolute_pose_inliers=30' + ' \\\n'

    ############### Bundle Adjustment Options ###############
    cmd += '--bundle_adjustment_robust_loss_function=CAUCHY' + ' \\\n'
    cmd += '--bundle_adjustment_robust_loss_width=5.0' + ' \\\n'
    #cmd += '--intrinsics_to_optimize=FOCAL_LENGTH|RADIAL_DISTORTION' +
    # ' \\\n'
    cmd += '--intrinsics_to_optimize=FOCAL_LENGTH' + ' \\\n'
    cmd += '--max_reprojection_error_pixels=6.0' + ' \\\n'

    ############### Triangulation Options ###############
    cmd += '--min_triangulation_angle_degrees=4.0' + ' \\\n'
    cmd += '--triangulation_reprojection_error_pixels=10.0' + ' \\\n'
    cmd += '--bundle_adjust_tracks=true' + ' \\\n'

    if FLAGS.less_num_inliers:
        cmd += '--min_num_inliers_for_valid_match=10' + ' \\\n'
    if FLAGS.less_sampson_error:
        cmd += '--max_sampson_error_for_verified_match=10.0' + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path
    run_cmd.save_and_run_cmd(
        cmd, os.path.join(PATHS.script_path, 'reconstruction.sh'))