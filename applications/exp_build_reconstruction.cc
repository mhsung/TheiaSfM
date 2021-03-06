// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <ceres/rotation.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <chrono>  // NOLINT
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "applications/command_line_helpers.h"
#include "applications/exp_bounding_box_utils.h"
#include "applications/exp_camera_param_io.h"
#include "applications/exp_camera_param_utils.h"
#include "applications/exp_neural_net_output_reader.h"

// Input/output files.
DEFINE_string(images, "", "Wildcard of images to reconstruct.");
DEFINE_string(matches_file, "", "Filename of the matches file.");
DEFINE_string(calibration_file, "",
              "Calibration file containing image calibration data.");
DEFINE_string(
    output_matches_file, "",
    "File to write the two-view matches to. This file can be used in "
    "future iterations as input to the reconstruction builder. Leave empty if "
    "you do not want to output matches.");
DEFINE_string(
    output_reconstruction, "",
    "Filename to write reconstruction to. The filename will be appended with "
    "the reconstruction number if multiple reconstructions are created.");

// Multithreading.
DEFINE_int32(num_threads, 1,
             "Number of threads to use for feature extraction and matching.");

// Feature and matching options.
DEFINE_string(descriptor, "SIFT",
              "Type of feature descriptor to use. Must be one of the "
                  "following: SIFT");
DEFINE_string(feature_density, "NORMAL",
              "Set to SPARSE, NORMAL, or DENSE to extract fewer or more "
              "features from each image.");
DEFINE_string(matching_strategy, "BRUTE_FORCE",
              "Strategy used to match features. Must be BRUTE_FORCE "
              " or CASCADE_HASHING");
DEFINE_bool(match_out_of_core, true,
            "Perform matching out of core by saving features to disk and "
            "reading them as needed. Set to false to perform matching all in "
            "memory.");
DEFINE_string(matching_working_directory, "",
              "Directory used during matching to store features for "
              "out-of-core matching.");
DEFINE_int32(matching_max_num_images_in_cache, 128,
             "Maximum number of images to store in the LRU cache during "
             "feature matching. The higher this number is the more memory is "
             "consumed during matching.");
DEFINE_double(lowes_ratio, 0.8, "Lowes ratio used for feature matching.");
DEFINE_double(max_sampson_error_for_verified_match, 4.0,
              "Maximum sampson error for a match to be considered "
                  "geometrically valid.");
DEFINE_int32(min_num_inliers_for_valid_match, 30,
             "Minimum number of geometrically verified inliers that a pair on "
             "images must have in order to be considered a valid two-view "
             "match.");
DEFINE_bool(bundle_adjust_two_view_geometry, true,
            "Set to false to turn off 2-view BA.");
DEFINE_bool(keep_only_symmetric_matches, true,
            "Performs two-way matching and keeps symmetric matches.");

// Reconstruction building options.
DEFINE_string(reconstruction_estimator, "GLOBAL",
              "Type of SfM reconstruction estimation to use.");
DEFINE_bool(reconstruct_largest_connected_component, false,
            "If set to true, only the single largest connected component is "
            "reconstructed. Otherwise, as many models as possible are "
            "estimated.");
DEFINE_bool(shared_calibration, false,
            "Set to true if all camera intrinsic parameters should be shared "
            "as a single set of intrinsics. This is useful, for instance, if "
            "all images in the reconstruction were taken with the same "
            "camera.");
DEFINE_bool(only_calibrated_views, false,
            "Set to true to only reconstruct the views where calibration is "
            "provided or can be extracted from EXIF");
DEFINE_int32(min_track_length, 2, "Minimum length of a track.");
DEFINE_int32(max_track_length, 50, "Maximum length of a track.");
DEFINE_string(intrinsics_to_optimize,
              "NONE",
              "Set to control which intrinsics parameters are optimized during "
              "bundle adjustment.");
DEFINE_double(max_reprojection_error_pixels, 4.0,
              "Maximum reprojection error for a correspondence to be "
              "considered an inlier after bundle adjustment.");

// Global SfM options.
DEFINE_string(global_rotation_estimator, "ROBUST_L1L2",
              "Type of global rotation estimation to use for global SfM.");
DEFINE_string(global_position_estimator, "NONLINEAR",
              "Type of global position estimation to use for global SfM.");
DEFINE_bool(refine_relative_translations_after_rotation_estimation, true,
            "Refine the relative translation estimation after computing the "
            "absolute rotations. This can help improve the accuracy of the "
            "position estimation.");
DEFINE_double(post_rotation_filtering_degrees, 5.0,
              "Max degrees difference in relative rotation and rotation "
              "estimates for rotation filtering.");
DEFINE_bool(extract_maximal_rigid_subgraph, false,
            "If true, only cameras that are well-conditioned for position "
            "estimation will be used for global position estimation.");
DEFINE_bool(filter_relative_translations_with_1dsfm, true,
            "Filter relative translation estimations with the 1DSfM algorithm "
            "to potentially remove outlier relativep oses for position "
            "estimation.");
DEFINE_int32(num_retriangulation_iterations, 1,
             "Number of times to retriangulate any unestimated tracks. Bundle "
             "adjustment is performed after retriangulation.");

// Nonlinear position estimation options.
DEFINE_int32(position_estimation_min_num_tracks_per_view, 0,
             "Minimum number of point to camera constraints for position "
                 "estimation.");
DEFINE_double(position_estimation_robust_loss_width, 0.1,
              "Robust loss width to use for position estimation.");

// Incremental SfM options.
DEFINE_double(absolute_pose_reprojection_error_threshold, 8.0,
              "The inlier threshold for absolute pose estimation.");
DEFINE_int32(min_num_absolute_pose_inliers, 30,
             "Minimum number of inliers in order for absolute pose estimation "
             "to be considered successful.");
DEFINE_double(full_bundle_adjustment_growth_percent, 5.0,
              "Full BA is only triggered for incremental SfM when the "
              "reconstruction has growth by this percent since the last time "
              "full BA was used.");
DEFINE_int32(partial_bundle_adjustment_num_views, 20,
             "When full BA is not being run, partial BA is executed on a "
             "constant number of views specified by this parameter.");

// Triangulation options.
DEFINE_double(min_triangulation_angle_degrees, 4.0,
              "Minimum angle between views for triangulation.");
DEFINE_double(triangulation_reprojection_error_pixels, 15.0,
              "Max allowable reprojection error on initial triangulation of points.");
DEFINE_bool(bundle_adjust_tracks, true,
            "Set to true to optimize tracks immediately upon estimation.");

// Bundle adjustment parameters.
DEFINE_string(bundle_adjustment_robust_loss_function, "NONE",
              "By setting this to an option other than NONE, a robust loss "
              "function will be used during bundle adjustment which can "
              "improve robustness to outliers. Options are NONE, HUBER, "
              "SOFTLONE, CAUCHY, ARCTAN, and TUKEY.");
DEFINE_double(bundle_adjustment_robust_loss_width, 10.0,
              "If the BA loss function is not NONE, then this value controls "
              "where the robust loss begins with respect to reprojection error "
              "in pixels.");

// @mhsung
// ---- //
DEFINE_string(initial_bounding_boxes_filepath, "", "");
DEFINE_string(initial_orientations_filepath, "", "");
DEFINE_string(initial_reconstruction_filepath, "", "");

DEFINE_bool(exp_global_run_bundle_adjustment, true,
            "Only used when 'EXP_GLOBAL' is chosen for rotation estimator "
                "type.");

DEFINE_bool(use_per_object_view_pair_weights, false,
            "Use different weight for each of object-view pair based on given"
                " scores. Otherwise, 'rotation_constraint_weight_multiplier' "
                "and 'position_constraint_weight_multiplier' are used as "
                "constant weights. Only used when 'CONSTRAINED_ROBUST_L1L2' "
                "is chosen for global rotation estimator type, and when "
                "'CONSTRAINED_NONLINEAR' is chosen for global position "
                "estimator type.");

DEFINE_double(rotation_constraint_weight_multiplier, 50.0,
              "Multiplied by per object-view weights, or used as a constant "
                  "weight if 'use_per_object_view_pair_weights' is set to"
                  " false. Only used when 'CONSTRAINED_ROBUST_L1L2' is chosen"
                  " for global rotation estimator type.");
DEFINE_double(position_constraint_weight_multiplier, 50.0,
              "Multiplied by per object-view weights, or used as a constant "
                  "weight if 'use_per_object_view_pair_weights' is set to"
                  " false. Only used when 'CONSTRAINED_NONLINEAR' is chosen "
                  "for global position estimator type.");

DEFINE_string(match_pairs_file, "",
              "Filename of match pair list. Each line has 'name1,name2' "
                  "format. Override 'match_only_consecutive_pairs'.");

DEFINE_bool(match_only_consecutive_pairs, false,
            "Set to true to match only consecutive pairs.");

DEFINE_int32(consecutive_pair_frame_range, 10,
             "Frame range of consecutive image pairs to be matched.");

DEFINE_bool(use_consecutive_camera_position_constraints, false, "");

DEFINE_double(consecutive_camera_position_constraint_weight, 1.0E3, "");
// ---- //


using theia::Reconstruction;
using theia::ReconstructionBuilder;
using theia::ReconstructionBuilderOptions;

// Sets the feature extraction, matching, and reconstruction options based on
// the command line flags. There are many more options beside just these located
// in //theia/vision/sfm/reconstruction_builder.h
ReconstructionBuilderOptions SetReconstructionBuilderOptions() {
  ReconstructionBuilderOptions options;
  options.num_threads = FLAGS_num_threads;
  options.output_matches_file = FLAGS_output_matches_file;

  options.descriptor_type = StringToDescriptorExtractorType(FLAGS_descriptor);
  options.feature_density = StringToFeatureDensity(FLAGS_feature_density);
  options.matching_options.match_out_of_core = FLAGS_match_out_of_core;
  options.matching_options.keypoints_and_descriptors_output_dir =
      FLAGS_matching_working_directory;
  options.matching_options.cache_capacity =
      FLAGS_matching_max_num_images_in_cache;
  options.matching_strategy =
      StringToMatchingStrategyType(FLAGS_matching_strategy);
  options.matching_options.lowes_ratio = FLAGS_lowes_ratio;
  options.matching_options.keep_only_symmetric_matches =
      FLAGS_keep_only_symmetric_matches;
  options.min_num_inlier_matches = FLAGS_min_num_inliers_for_valid_match;
  options.matching_options.perform_geometric_verification = true;
  options.matching_options.geometric_verification_options
      .estimate_twoview_info_options.max_sampson_error_pixels =
      FLAGS_max_sampson_error_for_verified_match;
  options.matching_options.geometric_verification_options.bundle_adjustment =
      FLAGS_bundle_adjust_two_view_geometry;
  options.matching_options.geometric_verification_options
      .triangulation_max_reprojection_error =
      FLAGS_triangulation_reprojection_error_pixels;
  options.matching_options.geometric_verification_options
      .min_triangulation_angle_degrees = FLAGS_min_triangulation_angle_degrees;
  options.matching_options.geometric_verification_options
      .final_max_reprojection_error = FLAGS_max_reprojection_error_pixels;

  options.min_track_length = FLAGS_min_track_length;
  options.max_track_length = FLAGS_max_track_length;

  // Reconstruction Estimator Options.
  theia::ReconstructionEstimatorOptions& reconstruction_estimator_options =
      options.reconstruction_estimator_options;
  reconstruction_estimator_options.min_num_two_view_inliers =
      FLAGS_min_num_inliers_for_valid_match;
  reconstruction_estimator_options.num_threads = FLAGS_num_threads;
  reconstruction_estimator_options.intrinsics_to_optimize =
    StringToOptimizeIntrinsicsType(FLAGS_intrinsics_to_optimize);
  options.reconstruct_largest_connected_component =
      FLAGS_reconstruct_largest_connected_component;
  options.only_calibrated_views = FLAGS_only_calibrated_views;
  reconstruction_estimator_options.max_reprojection_error_in_pixels =
      FLAGS_max_reprojection_error_pixels;

  // Which type of SfM pipeline to use (e.g., incremental, global, etc.);
  reconstruction_estimator_options.reconstruction_estimator_type =
      StringToReconstructionEstimatorType(FLAGS_reconstruction_estimator);

  // Global SfM Options.
  reconstruction_estimator_options.global_rotation_estimator_type =
      StringToRotationEstimatorType(FLAGS_global_rotation_estimator);
  reconstruction_estimator_options.global_position_estimator_type =
      StringToPositionEstimatorType(FLAGS_global_position_estimator);
  reconstruction_estimator_options.num_retriangulation_iterations =
      FLAGS_num_retriangulation_iterations;
  reconstruction_estimator_options
      .refine_relative_translations_after_rotation_estimation =
      FLAGS_refine_relative_translations_after_rotation_estimation;
  reconstruction_estimator_options.extract_maximal_rigid_subgraph =
      FLAGS_extract_maximal_rigid_subgraph;
  reconstruction_estimator_options.filter_relative_translations_with_1dsfm =
      FLAGS_filter_relative_translations_with_1dsfm;
  reconstruction_estimator_options
      .rotation_filtering_max_difference_degrees =
      FLAGS_post_rotation_filtering_degrees;
  reconstruction_estimator_options.nonlinear_position_estimator_options
      .min_num_points_per_view =
      FLAGS_position_estimation_min_num_tracks_per_view;

  // Incremental SfM Options.
  reconstruction_estimator_options
      .absolute_pose_reprojection_error_threshold =
      FLAGS_absolute_pose_reprojection_error_threshold;
  reconstruction_estimator_options.min_num_absolute_pose_inliers =
      FLAGS_min_num_absolute_pose_inliers;
  reconstruction_estimator_options
      .full_bundle_adjustment_growth_percent =
      FLAGS_full_bundle_adjustment_growth_percent;
  reconstruction_estimator_options.partial_bundle_adjustment_num_views =
      FLAGS_partial_bundle_adjustment_num_views;

  // Triangulation options (used by all SfM pipelines).
  reconstruction_estimator_options.min_triangulation_angle_degrees =
      FLAGS_min_triangulation_angle_degrees;
  reconstruction_estimator_options
      .triangulation_max_reprojection_error_in_pixels =
      FLAGS_triangulation_reprojection_error_pixels;
  reconstruction_estimator_options.bundle_adjust_tracks =
      FLAGS_bundle_adjust_tracks;

  // Bundle adjustment options (used by all SfM pipelines).
  reconstruction_estimator_options.bundle_adjustment_loss_function_type =
      StringToLossFunction(FLAGS_bundle_adjustment_robust_loss_function);
  reconstruction_estimator_options.bundle_adjustment_robust_loss_width =
      FLAGS_bundle_adjustment_robust_loss_width;

  // @mhsung
  reconstruction_estimator_options.exp_global_run_bundle_adjustment =
      FLAGS_exp_global_run_bundle_adjustment;
  reconstruction_estimator_options.use_per_object_view_pair_weights =
      FLAGS_use_per_object_view_pair_weights;
  reconstruction_estimator_options.rotation_constraint_weight_multiplier =
      FLAGS_rotation_constraint_weight_multiplier;
  reconstruction_estimator_options.position_constraint_weight_multiplier =
      FLAGS_position_constraint_weight_multiplier;

  if (FLAGS_use_consecutive_camera_position_constraints) {
    reconstruction_estimator_options.nonlinear_position_estimator_options
        .consecutive_camera_range = FLAGS_consecutive_pair_frame_range;
    reconstruction_estimator_options.nonlinear_position_estimator_options
        .consecutive_camera_position_constraint_weight =
        FLAGS_consecutive_camera_position_constraint_weight;
  } else {
    // Set zero weight.
    reconstruction_estimator_options
        .nonlinear_position_estimator_options
        .consecutive_camera_position_constraint_weight = 0.0;
  }

  return options;
}

// @mhsung
void ExtractFramesFromTwoImageFiles(
    const std::string& image1_file, const std::string& image2_file,
    int* image1_frame, int* image2_frame) {
  CHECK_NOTNULL(image1_frame);
  CHECK_NOTNULL(image2_frame);

  const std::string basename1 = stlplus::basename_part(image1_file);
  const std::string basename2 = stlplus::basename_part(image2_file);

  const std::string common_prefix(basename1.begin(), std::mismatch(
      basename1.begin(), basename1.end(), basename2.begin()).first);
  const int len_common_prefix = common_prefix.size();

  // NOTE:
  // Assume that the part after the common prefix is the frame number.
  const int len_image1_frame = basename1.size() - len_common_prefix;
  (*image1_frame) = std::stoi(
      basename1.substr(len_common_prefix, len_image1_frame));

  const int len_image2_frame = basename2.size() - len_common_prefix;
  (*image2_frame) = std::stoi(
      basename2.substr(len_common_prefix, len_image2_frame));
}

// @mhsung
bool IsImageFileSortedByFrame(const std::vector<std::string>& image_files) {
  const int num_images = image_files.size();
  CHECK_GE(num_images, 2);

  const std::string& image1_file = image_files[0];

  for (int count = 1; count < num_images; ++count) {
    const std::string& image2_file = image_files[count];

    int image1_frame_index, image2_frame_index;
    ExtractFramesFromTwoImageFiles(
        image1_file, image2_file, &image1_frame_index, &image2_frame_index);
    if ((image2_frame_index - image1_frame_index) != count)
      return false;
  }

  return true;
}

// @mhsung
// FIXME.
void ExtractFrameIndicesFromImages(
    const std::vector<std::string>& image_files,
    std::unordered_map<int, std::string>* frame_indices) {
  CHECK_NOTNULL(frame_indices);
  frame_indices->clear();

  for (const auto& image_file : image_files) {
    std::string filename;
    CHECK(theia::GetFilenameFromFilepath(image_file, false, &filename));

    // NOTE:
    // Frame index is number after the last '_'.
    const std::string frame_index_str =
        filename.substr(filename.rfind('_') + 1);
    CHECK(!frame_index_str.empty());
    const int frame_index = std::stoi(frame_index_str);

    frame_indices->emplace(frame_index, image_file);
  }
}

// @mhsung
bool ReadMatchPairs(
    const std::string& filename,
    std::vector<std::pair<std::string, std::string> >* pairs_to_match) {
  CHECK_NOTNULL(pairs_to_match);

  std::ifstream file(filename);
  if (!file.good()) {
    LOG(WARNING) << "Can't read file: '" << filename << "'.";
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {

    // FIXME:
    // Remove '\r' in 'std::getline' function.
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    std::stringstream sstr(line);
    std::string name1, name2;
    if (!std::getline(sstr, name1, ',') || !std::getline(sstr, name2, ',')) {
      LOG(WARNING) << "Wrong format: '" << line << "'.";
      return false;
    }

    // NOTE:
    // Add extension '.png'.
    name1 += ".png";
    name2 += ".png";
    pairs_to_match->emplace_back(name1, name2);
  }

  CHECK(!pairs_to_match->empty()) << "No image pair to match.";
  LOG(INFO) << "# of pairs: " << pairs_to_match->size();
  return true;
}

// @mhsung
void GetConsecutivePairsToMatch(
    const int frame_range,
    const std::vector<std::string>& image_files,
    std::vector<std::pair<std::string, std::string> >* pairs_to_match) {
  CHECK_NOTNULL(pairs_to_match);
  CHECK_GT(frame_range, 0);

  std::unordered_map<int, std::string> frame_indices;
  ExtractFrameIndicesFromImages(image_files, &frame_indices);

  pairs_to_match->clear();
  pairs_to_match->reserve(
      FLAGS_consecutive_pair_frame_range * frame_indices.size());

  for (const auto& frame : frame_indices) {
    const int i = frame.first;
    const std::string& image_file_i = frame.second;
    std::string image_name_i;
    theia::GetFilenameFromFilepath(image_file_i, true, &image_name_i);

    for (int j = i + 1; j <= i + FLAGS_consecutive_pair_frame_range; j++) {
      if (frame_indices.find(j) != frame_indices.end()) {
        const std::string& image_file_j = frame_indices[j];
        std::string image_name_j;
        theia::GetFilenameFromFilepath(image_file_j, true, &image_name_j);
        pairs_to_match->emplace_back(image_name_i, image_name_j);
      }
    }
  }

  CHECK(!pairs_to_match->empty()) << "No image pair to match.";
  LOG(INFO) << "# of consecutive pairs: " << pairs_to_match->size();
}

/*
// @mhsung
void SetInitialOrientations(ReconstructionBuilder* reconstruction_builder) {
  // Read orientation.
  std::unordered_map<std::string, Eigen::Matrix3d> init_orientations_with_names;
  CHECK(ReadOrientations(
      FLAGS_initial_orientations_data_type, FLAGS_initial_orientations_filepath,
      &init_orientations_with_names));
  std::unordered_map<theia::ViewId, Eigen::Matrix3d> init_orientations;
  MapViewNamesToIds(*reconstruction_builder->GetReconstruction(),
                    init_orientations_with_names, &init_orientations);

  for (const auto& init_orientation : init_orientations) {
    const theia::ViewId view_id = init_orientation.first;
    theia::View* view = reconstruction_builder->GetMutableReconstruction()
        ->MutableView(view_id);
    CHECK(view) << "View does not exist (View ID = " << view_id << ").";

    Eigen::Vector3d angle_axis;
    ceres::RotationMatrixToAngleAxis(
        ceres::ColumnMajorAdapter3x3(init_orientation.second.data()),
        angle_axis.data());
    reconstruction_builder->SetInitialObjectViewOrientations(
        0, view_id, angle_axis);
  }
}

// @mhsung
void SetInitialPositionDirections(ReconstructionBuilder*
    reconstruction_builder) {
  // Read orientation.
  std::unordered_map<std::string, Eigen::Vector4d>
      init_bounding_boxes_with_names;
  ReadBoundingBoxes(FLAGS_initial_bounding_boxes_filepath,
                    &init_bounding_boxes_with_names);
  std::unordered_map<theia::ViewId, Eigen::Vector4d> init_bounding_boxes;
  MapViewNamesToIds(*reconstruction_builder->GetReconstruction(),
                    init_bounding_boxes_with_names, &init_bounding_boxes);

  for (const auto& init_bounding_box : init_bounding_boxes) {
    const theia::ViewId view_id = init_bounding_box.first;
    theia::View* view = reconstruction_builder->GetMutableReconstruction()
        ->MutableView(view_id);
    CHECK(view) << "View does not exist (View ID = " << view_id << ").";

    // Store camera(origin) to object direction.
    const Eigen::Vector3d cam_coord_cam_to_obj_dir =
        ComputeCameraToObjectDirections(
            init_bounding_box.second, view->CameraIntrinsicsPrior());
    reconstruction_builder->SetInitialViewObjectPositionDirections(
        0, view_id, cam_coord_cam_to_obj_dir);
  }
}
*/

void SetInitialOrientationsAndPositions(ReconstructionBuilder*
    reconstruction_builder) {
  // Read bounding box information.
  std::unordered_map<ObjectId, DetectedBBoxPtrList> object_bboxes;
  ReadNeuralNetBBoxesAndOrientations(FLAGS_initial_bounding_boxes_filepath,
                                     FLAGS_initial_orientations_filepath,
                                     &object_bboxes);

  for (const auto& object : object_bboxes) {
    const theia::ObjectId object_id = object.first;
    for (const auto& bbox : object.second) {
      Reconstruction* reconstruction =
        reconstruction_builder->GetMutableReconstruction();
      const theia::ViewId view_id =
        reconstruction->ViewIdFromName(bbox->view_name_);
      if (view_id == kInvalidViewId) {
        LOG(WARNING) << "View does not exist (View ID = "
                     << bbox->view_name_ << ").";
        continue;
      }
      theia::View* view = reconstruction->MutableView(view_id);
      CHECK(view != nullptr);

      // Set orientation
      const Eigen::Matrix3d orientation =
        ComputeTheiaCameraRotationFromCameraParams(bbox->camera_param_);
      Eigen::Vector3d angle_axis;
      ceres::RotationMatrixToAngleAxis(
        ceres::ColumnMajorAdapter3x3(orientation.data()), angle_axis.data());
      reconstruction_builder->SetInitialObjectViewOrientation(
        object_id, view_id, angle_axis);
      if (FLAGS_use_per_object_view_pair_weights) {
        reconstruction_builder->SetInitialObjectViewOrientationWeight(
            object_id, view_id, bbox->orientation_score_);
      }

      // Set position.
      const Eigen::Vector3d cam_coord_cam_to_obj_dir =
        ComputeCameraToObjectDirections(
          bbox->bbox_, view->CameraIntrinsicsPrior());
      reconstruction_builder->SetInitialViewObjectPositionDirection(
          object_id, view_id, cam_coord_cam_to_obj_dir);
      if (FLAGS_use_per_object_view_pair_weights) {
        reconstruction_builder->SetInitialViewObjectPositionDirectionWeight(
            object_id, view_id, bbox->bbox_score_);
      }
    }
  }
}

ViewId FarthestView(const Reconstruction& reconstruction,
                    const ViewId query_view_id) {
  CHECK_NE(query_view_id, kInvalidViewId);
  const View* query_view = reconstruction.View(query_view_id);

  ViewId farthest_view_id = kInvalidViewId;
  double max_distance_from_query = 0.0;

  for (const ViewId view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    CHECK (view != nullptr && view->IsEstimated());

    const double distance_from_query =
        (view->Camera().GetPosition() -
            query_view->Camera().GetPosition()).norm();
    if (distance_from_query > max_distance_from_query) {
      max_distance_from_query = distance_from_query;
      farthest_view_id = view_id;
    }
  }
  CHECK_NE(farthest_view_id, kInvalidViewId);
  return farthest_view_id;
}

void CreateVirtualObjects(
    const Reconstruction& reconstruction,
    const int num_virtual_objects,
    std::vector<Eigen::Matrix3d>* object_orientations,
    std::vector<Eigen::Vector3d>* object_positions) {
  CHECK_NOTNULL(object_orientations)->clear();
  CHECK_NOTNULL(object_positions)->clear();
  CHECK_GE(reconstruction.ViewIds().size(), num_virtual_objects);
  object_orientations->reserve(num_virtual_objects);
  object_positions->reserve(num_virtual_objects);

  ViewId query_view_id = reconstruction.ViewIds().front();
  for (int i = 0; i < num_virtual_objects; i++) {
    const View* prev_query_view = reconstruction.View(query_view_id);

    query_view_id = FarthestView(reconstruction, query_view_id);
    LOG(INFO) << "View " << query_view_id << " used as object " << i << ".";

    const View* curr_query_view = reconstruction.View(query_view_id);

    // Set orientations.
    object_orientations->push_back(
        curr_query_view->Camera().GetOrientationAsRotationMatrix());

    // Set positions.
    // NOTE:
    // The virtual object positions must not be the same with any camera
    // position, otherwise translation direction optimization fails.
    // Thus, we interpolate between two pivot camera positions.
    object_positions->push_back(
        (prev_query_view->Camera().GetPosition() * 1.0) +
        (curr_query_view->Camera().GetPosition() * 2.0) / 3.0);
  }
}

void SetInitialReconstruction(ReconstructionBuilder* reconstruction_builder) {
  // Read initial reconstruction.
  std::unique_ptr<theia::Reconstruction> init_reconstruction(
      new theia::Reconstruction());
  CHECK(ReadReconstruction(FLAGS_initial_reconstruction_filepath,
                           init_reconstruction.get()))
  << "Could not read reconstruction file.";

  const int num_objects = 2;
  const double kViewProportion = 0.5;
  const double kObjectViewProportion = 1.0;
  const double kRotationAngleNoise = 0.0 / 180.0 * M_PI;
  const double kPositionAngleNoise = 0.0 / 180.0 * M_PI;

//  const int num_objects = 8;
//  const double kViewProportion = 0.5;
//  const double kObjectViewProportion = 0.5;
//  const double kRotationAngleNoise = 7.0 / 180.0 * M_PI;
//  const double kPositionAngleNoise = 25.0 / 180.0 * M_PI;
  std::srand(1234);

  // Create virtual objects.
  std::vector<Eigen::Matrix3d> object_orientations;
  std::vector<Eigen::Vector3d> object_positions;
  CreateVirtualObjects(
      *init_reconstruction, num_objects,
      &object_orientations, &object_positions);
  CHECK_EQ(object_orientations.size(), num_objects);
  CHECK_EQ(object_orientations.size(), num_objects);

  // Get views in initial reconstruction in a random order.
  std::vector<ViewId> init_view_ids = init_reconstruction->ViewIds();
  std::random_shuffle(init_view_ids.begin(), init_view_ids.end());
  const int num_selected_init_views =
      static_cast<int>(kViewProportion * init_view_ids.size());
  CHECK_LE(num_selected_init_views, init_view_ids.size());
  init_view_ids.resize(num_selected_init_views);
  LOG(INFO) << "# all views: " << init_view_ids.size();
  LOG(INFO) << "# selected views: " << num_selected_init_views;

  // Compute relative orientations and positions with virtual objects.
  for (ObjectId object_id = 0; object_id < num_objects; object_id++) {

    std::vector<ViewId> obj_init_view_ids = init_view_ids;
    if (kObjectViewProportion < 1.0) {
      std::random_shuffle(obj_init_view_ids.begin(), obj_init_view_ids.end());
      const int num_selected_obj_init_views =
          static_cast<int>(kObjectViewProportion * obj_init_view_ids.size());
      CHECK_LE(num_selected_obj_init_views, obj_init_view_ids.size());
      obj_init_view_ids.resize(num_selected_obj_init_views);
      LOG(INFO) << "# object views: " << num_selected_obj_init_views;
    }

    for (const ViewId init_view_id : obj_init_view_ids) {
      const View* init_view = init_reconstruction->View(init_view_id);
      CHECK (init_view != nullptr && init_view->IsEstimated());

      Reconstruction* reconstruction =
          reconstruction_builder->GetMutableReconstruction();
      const theia::ViewId view_id =
          reconstruction->ViewIdFromName(init_view->Name());
      if (view_id == kInvalidViewId) {
        LOG(WARNING) << "View does not exist (View name = "
                     << init_view->Name() << ").";
        continue;
      }

      // Set orientation.
      const Eigen::Matrix3d orientation =
          init_view->Camera().GetOrientationAsRotationMatrix() *
              object_orientations[object_id].transpose();
      Eigen::Vector3d angle_axis;
      ceres::RotationMatrixToAngleAxis(
          ceres::ColumnMajorAdapter3x3(orientation.data()), angle_axis.data());
      if (kRotationAngleNoise > 0) {
        const Eigen::AngleAxisd noise_rotation(
            kRotationAngleNoise, Eigen::Vector3d::Random().normalized());
        angle_axis = noise_rotation * angle_axis;
      }
      reconstruction_builder->SetInitialObjectViewOrientation(
          object_id, view_id, angle_axis);

      // Set position.
      const Eigen::Vector3d cam_to_obj_dir =
          (object_positions[object_id] -
              init_view->Camera().GetPosition()).normalized();
      Eigen::Vector3d cam_coord_cam_to_obj_dir =
          init_view->Camera().GetOrientationAsRotationMatrix() * cam_to_obj_dir;
      if (kPositionAngleNoise > 0) {
        const Eigen::AngleAxisd noise_rotation(
            kRotationAngleNoise, Eigen::Vector3d::Random().normalized());
        cam_coord_cam_to_obj_dir = noise_rotation * cam_coord_cam_to_obj_dir;
      }
      reconstruction_builder->SetInitialViewObjectPositionDirection(
          object_id, view_id, cam_coord_cam_to_obj_dir);
    }
  }
}

void AddMatchesToReconstructionBuilder(
    ReconstructionBuilder* reconstruction_builder) {
  // Load matches from file.
  std::vector<std::string> image_files;
  std::vector<theia::CameraIntrinsicsPrior> camera_intrinsics_prior;
  std::vector<theia::ImagePairMatch> image_matches;

  // Read in match file.
  theia::ReadMatchesAndGeometry(FLAGS_matches_file,
                                &image_files,
                                &camera_intrinsics_prior,
                                &image_matches);

  // @mhsung
  // Check whether image files are sorted based on frames.
  // This is checked to use consecutive camera constraints.
	if (FLAGS_use_consecutive_camera_position_constraints) {
		CHECK(IsImageFileSortedByFrame(image_files));
	}

  // Add all the views. When the intrinsics group id is invalid, the
  // reconstruction builder will assume that the view does not share its
  // intrinsics with any other views.
  theia::CameraIntrinsicsGroupId intrinsics_group_id =
      theia::kInvalidCameraIntrinsicsGroupId;
  if (FLAGS_shared_calibration) {
    intrinsics_group_id = 0;
  }

  for (int i = 0; i < image_files.size(); i++) {
    reconstruction_builder->AddImageWithCameraIntrinsicsPrior(
        image_files[i], camera_intrinsics_prior[i], intrinsics_group_id);
  }

  // Add the matches.
  for (const auto& match : image_matches) {
    CHECK(reconstruction_builder->AddTwoViewMatch(match.image1,
                                                  match.image2,
                                                  match));
  }
}

void AddImagesToReconstructionBuilder(
    ReconstructionBuilder* reconstruction_builder) {
  std::vector<std::string> image_files;
  CHECK(theia::GetFilepathsFromWildcard(FLAGS_images, &image_files))
      << "Could not find images that matched the filepath: " << FLAGS_images
      << ". NOTE that the ~ filepath is not supported.";

  CHECK_GT(image_files.size(), 0) << "No images found in: " << FLAGS_images;

  // @mhsung
  // Check whether image files are sorted based on frames.
  // This is checked to use consecutive camera constraints.
	if (FLAGS_use_consecutive_camera_position_constraints) {
		CHECK(IsImageFileSortedByFrame(image_files));
	}

  // Load calibration file if it is provided.
  std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
      camera_intrinsics_prior;
  if (FLAGS_calibration_file.size() != 0) {
    CHECK(theia::ReadCalibration(FLAGS_calibration_file,
                                 &camera_intrinsics_prior))
        << "Could not read calibration file.";
  }

  // Add images with possible calibration. When the intrinsics group id is
  // invalid, the reconstruction builder will assume that the view does not
  // share its intrinsics with any other views.
  theia::CameraIntrinsicsGroupId intrinsics_group_id =
      theia::kInvalidCameraIntrinsicsGroupId;
  if (FLAGS_shared_calibration) {
    intrinsics_group_id = 0;
  }

  for (const std::string& image_file : image_files) {
    std::string image_filename;
    CHECK(theia::GetFilenameFromFilepath(image_file, true, &image_filename));

    const theia::CameraIntrinsicsPrior* image_camera_intrinsics_prior =
      FindOrNull(camera_intrinsics_prior, image_filename);
    if (image_camera_intrinsics_prior != nullptr) {
      CHECK(reconstruction_builder->AddImageWithCameraIntrinsicsPrior(
          image_file, *image_camera_intrinsics_prior, intrinsics_group_id));
    } else {
      CHECK(reconstruction_builder->AddImage(image_file, intrinsics_group_id));
    }
  }

  // @mhsung
  if (FLAGS_match_pairs_file != "") {
    std::vector<std::pair<std::string, std::string> > pairs_to_match;
    CHECK(ReadMatchPairs(FLAGS_match_pairs_file, &pairs_to_match));
    reconstruction_builder->SetImagePairsToMatch(pairs_to_match);
  }
  else if (FLAGS_match_only_consecutive_pairs) {
    std::vector<std::pair<std::string, std::string> > pairs_to_match;
    GetConsecutivePairsToMatch(FLAGS_consecutive_pair_frame_range,
                               image_files, &pairs_to_match);
    reconstruction_builder->SetImagePairsToMatch(pairs_to_match);
  }

  // Extract and match features.
  CHECK(reconstruction_builder->ExtractAndMatchFeatures());
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  CHECK_GT(FLAGS_output_reconstruction.size(), 0)
      << "Must specify a filepath to output the reconstruction.";

  const ReconstructionBuilderOptions options =
      SetReconstructionBuilderOptions();

  ReconstructionBuilder reconstruction_builder(options);
  // If matches are provided, load matches otherwise load images.
  if (FLAGS_matches_file.size() != 0) {
    AddMatchesToReconstructionBuilder(&reconstruction_builder);
  } else if (FLAGS_images.size() != 0) {
    AddImagesToReconstructionBuilder(&reconstruction_builder);
  } else {
    LOG(FATAL)
        << "You must specify either images to reconstruct or a match file.";
  }

  /*
  // @mhsung
  if (FLAGS_initial_orientations_filepath != "") {
    SetInitialOrientations(&reconstruction_builder);
  }

  // @mhsung
  if (FLAGS_initial_bounding_boxes_filepath != "") {
    SetInitialPositionDirections(&reconstruction_builder);
  }
  */

  // @mhsung
  if (FLAGS_initial_orientations_filepath != "" &&
    FLAGS_initial_bounding_boxes_filepath != "") {
    SetInitialOrientationsAndPositions(&reconstruction_builder);
  } else if (FLAGS_initial_reconstruction_filepath != "") {
    SetInitialReconstruction(&reconstruction_builder);
  }

  std::vector<Reconstruction*> reconstructions;
  CHECK(reconstruction_builder.BuildReconstruction(&reconstructions))
      << "Could not create a reconstruction.";

  for (int i = 0; i < reconstructions.size(); i++) {
    const std::string output_file =
        theia::StringPrintf("%s-%d", FLAGS_output_reconstruction.c_str(), i);
    LOG(INFO) << "Writing reconstruction " << i << " to " << output_file;
    CHECK(theia::WriteReconstruction(*reconstructions[i], output_file))
        << "Could not write reconstruction to file.";
  }
}
