#!/usr/bin/python

import datetime
import gflags
import os
import sys


FLAGS = gflags.FLAGS
gflags.DEFINE_string('bin_dir', './build/bin/', '')
gflags.DEFINE_integer('num_threads', 8, '')

# User parameters.
gflags.DEFINE_string('name', 'MVI_0167', '')
gflags.DEFINE_bool('overwrite', False, '')
gflags.DEFINE_bool('every_10', True, '')
gflags.DEFINE_integer('seq_range', 0, '') # '<= 0' indicates all pair matches.


def save_and_run_cmd(cmd, filepath):
    print(cmd)
    with open(filepath, 'w') as f: f.write(cmd)
    os.system('chmod +x ' + filepath)
    start = datetime.datetime.now()
    os.system(filepath)
    end = datetime.datetime.now()
    elapsed = end - start
    print('Start time: ' + start.strftime("%Y-%m-%d %H:%M:%S"))
    print('End time: ' + end.strftime("%Y-%m-%d %H:%M:%S"))
    print('Elapsed time: ' + str(end - start))
    print('')


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Change the root directory if it is needed.
    data_root_path = os.path.normpath(
        os.path.join(os.getcwd(), '../../data/', FLAGS.name))

    # Image path.
    image_path = os.path.join(data_root_path, 'images')
    if not os.path.isdir(image_path):
        print('Image directory does not exist: "' + image_path + '"')
        exit(-1)

    # Feature path.
    feature_path = os.path.join(data_root_path, 'features')
    if not os.path.isdir(feature_path):
        os.makedirs(feature_path)

    if FLAGS.every_10:
        image_wildcard = os.path.join(image_path, '*0.png')
        feature_wildcard = os.path.join(feature_path, '*0.png.features')
    else:
        image_wildcard = os.path.join(image_path, '*.png')
        feature_wildcard = os.path.join(feature_path, '*.png.features')

    # Output path.
    output_dirname = 'sfm'
    if FLAGS.every_10:
        output_dirname += '_10'
    if FLAGS.seq_range > 0:
        output_dirname += ('_seq_' + str(FLAGS.seq_range))

    output_path = os.path.join(data_root_path, output_dirname)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    log_path = os.path.join(output_path, 'log')
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    # Output files.
    calibration_file = os.path.join(data_root_path, 'calibration.txt')
    matches_file = os.path.join(output_path, 'matches.bin')
    output_file = os.path.join(output_path, 'output')

    print('== Paths ==')
    print('Image files: ' + image_wildcard)
    print('Calibration File: ' + calibration_file)
    print('Feature files: ' + feature_wildcard)
    print('Match files: ' + matches_file)
    print('Output file: ' + output_file)
    print('')

    print('== Options ==')
    if FLAGS.overwrite: print('Overwrite: On')
    if FLAGS.every_10: print('Every 10 frame: On')
    if FLAGS.seq_range > 0: print('Sequence range: ' + str(FLAGS.seq_range))
    print('')


    exist_calibration = os.path.exists(calibration_file)
    if FLAGS.overwrite or (not exist_calibration):
        print('== Create calibration file ==')
        cmd = ''
        cmd += FLAGS.bin_dir + '/create_calibration_file_from_exif '
        cmd += '--log_dir=' + log_path + ' '
        cmd += '--images=' + image_wildcard + ' '
        cmd += '--initialize_uncalibrated_images_with_median_viewing_angle=true '
        cmd += '--output_calibration_file=' + calibration_file + ' '
        save_and_run_cmd(cmd, os.path.join(output_path, 'calibrate.sh'))


    exist_features = os.listdir(feature_path)
    if FLAGS.overwrite or (not exist_features):
        print('== Extract features ==')
        cmd = ''
        cmd += FLAGS.bin_dir + '/extract_features '
        cmd += '--log_dir=' + log_path + ' '
        cmd += '--num_threads=' + str(FLAGS.num_threads) + ' '
        cmd += '--input_images=' + image_wildcard + ' '
        cmd += '--features_output_directory=' + feature_path + ' '
        save_and_run_cmd(cmd, os.path.join(output_path, 'feature.sh'))


    exist_matches = os.path.exists(matches_file)
    if FLAGS.overwrite or (not exist_matches):
        print('== Match features ==')
        cmd = ''
        cmd += FLAGS.bin_dir + '/exp_match_features '
        cmd += '--log_dir=' + log_path + ' '
        cmd += '--num_threads=' + str(FLAGS.num_threads) + ' '
        cmd += '--input_features=' + feature_wildcard + ' '
        cmd += '--calibration_file=' + calibration_file + ' '
        cmd += '--matching_max_num_images_in_cache=128 '
        cmd += '--output_matches_file=' + matches_file + ' '

        if FLAGS.seq_range > 0:
            cmd += '--match_only_consecutive_pairs=true '
            cmd += '--consecutive_pair_frame_range=' +\
                   str(FLAGS.seq_range) + ' '

        save_and_run_cmd(cmd, os.path.join(output_path, 'match.sh'))


    exist_reconstruction = os.path.exists(output_file + '-0')
    if FLAGS.overwrite or (not exist_reconstruction):
        print('== Build reconstruction ==')
        cmd = ''
        cmd += FLAGS.bin_dir + '/exp_build_reconstruction '
        cmd += '--log_dir=' + log_path + ' '
        cmd += '--num_threads=' + str(FLAGS.num_threads) + ' '
        cmd += '--matches_file=' + matches_file + ' '
        cmd += '--match_out_of_core=false '
        cmd += '--calibration_file=' + calibration_file + ' '
        cmd += '--output_reconstruction=' + output_file + ' '

        ############### General SfM Options ###############
        cmd += '--reconstruction_estimator=GLOBAL '
        cmd += '--max_track_length=1000 '
        cmd += '--reconstruct_largest_connected_component=true '
        cmd += '--only_calibrated_views=false '
        cmd += '--shared_calibration=true '

        ############### Global SfM Options ###############
        cmd += '--global_position_estimator=LEAST_UNSQUARED_DEVIATION '
        cmd += '--global_rotation_estimator=ROBUST_L1L2 '
        cmd += '--post_rotation_filtering_degrees=20.0 '
        cmd += '--refine_relative_translations_after_rotation_estimation=true '
        cmd += '--extract_maximal_rigid_subgraph=false '
        cmd += '--filter_relative_translations_with_1dsfm=true '
        cmd += '--position_estimation_min_num_tracks_per_view=10 '
        cmd += '--position_estimation_robust_loss_width=0.01 '
        cmd += '--num_retriangulation_iterations=2 '

        ############### Incremental SfM Options ###############
        cmd += '--absolute_pose_reprojection_error_threshold=10 '
        cmd += '--partial_bundle_adjustment_num_views=20 '
        cmd += '--full_bundle_adjustment_growth_percent=5 '
        cmd += '--min_num_absolute_pose_inliers=30 '

        ############### Bundle Adjustment Options ###############
        cmd += '--bundle_adjustment_robust_loss_function=CAUCHY '
        cmd += '--bundle_adjustment_robust_loss_width=5.0 '
        #cmd += '--intrinsics_to_optimize=FOCAL_LENGTH|RADIAL_DISTORTION '
        cmd += '--intrinsics_to_optimize=FOCAL_LENGTH '
        cmd += '--max_reprojection_error_pixels=6.0 '

        ############### Triangulation Options ###############
        cmd += '--min_triangulation_angle_degrees=4.0 '
        cmd += '--triangulation_reprojection_error_pixels=10.0 '
        cmd += '--bundle_adjust_tracks=true '

        save_and_run_cmd(cmd, os.path.join(output_path, 'reconstruction.sh'))


    cmd = ''
    cmd += FLAGS.bin_dir + '/view_reconstruction '
    cmd += '--reconstruction=' + output_file + '-0 '
    os.system(cmd)
