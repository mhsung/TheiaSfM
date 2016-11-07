#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import run_cmd
import os

kNNPriorWeight = 10
kGTPriorWeight = 1.0E6


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
    # cmd += '--reconstruction_estimator=GLOBAL' + ' \\\n'
    cmd += '--reconstruction_estimator=EXP_GLOBAL' + ' \\\n'
    cmd += '--max_track_length=100000' + ' \\\n'
    cmd += '--reconstruct_largest_connected_component=true' + ' \\\n'
    cmd += '--only_calibrated_views=false' + ' \\\n'
    cmd += '--shared_calibration=true' + ' \\\n'

    ############### Global SfM Options ###############
    if FLAGS.use_initial_orientations or FLAGS.use_gt_orientations:
        cmd += '--global_position_estimator=CONSTRAINED_NONLINEAR' + ' \\\n'
        cmd += '--global_rotation_estimator=CONSTRAINED_ROBUST_L1L2' + ' \\\n'
    else:
        # cmd += '--global_position_estimator=LEAST_UNSQUARED_DEVIATION' + '
        # \\\n'
        cmd += '--global_position_estimator=NONLINEAR' + ' \\\n'
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

    ############### Triangulation Options ###############
    #cmd += '--max_reprojection_error_pixels=6.0' + ' \\\n'
    #cmd += '--min_triangulation_angle_degrees=4.0' + ' \\\n'
    #cmd += '--triangulation_reprojection_error_pixels=10.0' + ' \\\n'
    cmd += '--max_reprojection_error_pixels=100000.0' + ' \\\n'
    cmd += '--min_triangulation_angle_degrees=0.0' + ' \\\n'
    cmd += '--triangulation_reprojection_error_pixels=100000.0' + ' \\\n'
    cmd += '--bundle_adjust_tracks=true' + ' \\\n'

    # if FLAGS.less_num_inliers:
    #     cmd += '--min_num_inliers_for_valid_match=10' + ' \\\n'
    # if FLAGS.less_sampson_error:
    #     cmd += '--max_sampson_error_for_verified_match=10.0' + ' \\\n'

    if FLAGS.no_bundle:
        cmd += '--exp_global_run_bundle_adjustment=false' + ' \\\n'
    else:
        cmd += '--exp_global_run_bundle_adjustment=true' + ' \\\n'

    if FLAGS.use_initial_orientations:
        cmd += '--initial_bounding_boxes_filepath=' + \
               PATHS.init_bbox_path + ' \\\n'
        cmd += '--initial_orientations_filepath=' + \
               PATHS.init_orientation_path + ' \\\n'
    if FLAGS.use_gt_orientations:
        cmd += '--initial_reconstruction_filepath=' + \
               PATHS.ground_truth_reconstruction_path + ' \\\n'

    # FIXME:
    # Make the weights as options.
    if FLAGS.use_initial_orientations:
        if FLAGS.use_score_based_weights:
            assert (FLAGS.use_initial_orientations)
            assert (not FLAGS.use_gt_orientations)
            cmd += '--use_per_object_view_pair_weights=true' + ' \\\n'
            cmd += '--rotation_constraint_weight_multiplier=' + \
                    str(2 * kNNPriorWeight) + ' \\\n'
            cmd += '--position_constraint_weight_multiplier=' + \
                    str(kNNPriorWeight) + ' \\\n'
        else:
            cmd += '--rotation_constraint_weight_multiplier=' + \
                    str(kNNPriorWeight) + ' \\\n'
            cmd += '--position_constraint_weight_multiplier=' + \
                    str(kNNPriorWeight) + ' \\\n'
    elif FLAGS.use_gt_orientations:
        cmd += '--rotation_constraint_weight_multiplier=' + \
               str(kGTPriorWeight) + ' \\\n'
        cmd += '--position_constraint_weight_multiplier=' + \
               str(kGTPriorWeight) + ' \\\n'

    if FLAGS.use_consecutive_camera_constraints:
        cmd += '--use_consecutive_camera_position_constraints=true' + ' \\\n'

    cmd += '--log_dir=' + PATHS.log_path + ' \\\n'
    cmd += '--alsologtostderr'
    run_cmd.save_and_run_cmd(
        cmd, os.path.join(PATHS.script_path, 'reconstruction.sh'))

