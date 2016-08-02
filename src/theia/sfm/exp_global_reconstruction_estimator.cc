// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "theia/sfm/exp_global_reconstruction_estimator.h"

#include <algorithm>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <memory>
#include <sstream>  // NOLINT

#include "theia/sfm/reconstruction_estimator_options.h"
#include "theia/sfm/reconstruction_estimator_utils.h"
#include "theia/sfm/set_camera_intrinsics_from_priors.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/view_graph/view_graph.h"
#include "theia/util/map_util.h"
#include "theia/util/timer.h"

namespace theia {

using Eigen::Vector3d;

namespace {

// All times are given in seconds.
struct ExpGlobalReconstructionEstimatorTimings {
  double initial_view_graph_filtering_time = 0.0;
  double camera_intrinsics_calibration_time = 0.0;
  double rotation_estimation_time = 0.0;
  double rotation_filtering_time = 0.0;
  double relative_translation_optimization_time = 0.0;
  double relative_translation_filtering_time = 0.0;
  double position_estimation_time = 0.0;
};

FilterViewPairsFromRelativeTranslationOptions
SetRelativeTranslationFilteringOptions(
    const ReconstructionEstimatorOptions& options) {
  FilterViewPairsFromRelativeTranslationOptions fvpfrt_options;
  fvpfrt_options.num_threads = options.num_threads;
  fvpfrt_options.num_iterations = options.translation_filtering_num_iterations;
  fvpfrt_options.translation_projection_tolerance =
      options.translation_filtering_projection_tolerance;
  return fvpfrt_options;
}

void SetUnderconstrainedAsUnestimated(Reconstruction* reconstruction) {
  int num_underconstrained_views = -1;
  int num_underconstrained_tracks = -1;
  while (num_underconstrained_views != 0 && num_underconstrained_tracks != 0) {
    num_underconstrained_views =
        SetUnderconstrainedViewsToUnestimated(reconstruction);
    num_underconstrained_tracks =
        SetUnderconstrainedTracksToUnestimated(reconstruction);
  }
}

Eigen::Vector3d ComputeRelativeRotationError(
    const Eigen::Vector3d& relative_rotation,
    const Eigen::Vector3d& rotation1,
    const Eigen::Vector3d& rotation2) {
  Eigen::Matrix3d relative_rotation_matrix, rotation_matrix1, rotation_matrix2;
  ceres::AngleAxisToRotationMatrix(
      relative_rotation.data(),
      ceres::ColumnMajorAdapter3x3(relative_rotation_matrix.data()));
  ceres::AngleAxisToRotationMatrix(
      rotation1.data(), ceres::ColumnMajorAdapter3x3(rotation_matrix1.data()));
  ceres::AngleAxisToRotationMatrix(
      rotation2.data(), ceres::ColumnMajorAdapter3x3(rotation_matrix2.data()));

  // Compute the relative rotation error.
  const Eigen::Matrix3d relative_rotation_matrix_error =
      rotation_matrix2.transpose() * relative_rotation_matrix *
      rotation_matrix1;
  Eigen::Vector3d relative_rotation_error;
  ceres::RotationMatrixToAngleAxis(
      ceres::ColumnMajorAdapter3x3(relative_rotation_matrix_error.data()),
      relative_rotation_error.data());
  return relative_rotation_error;
}

}  // namespace

ExpGlobalReconstructionEstimator::ExpGlobalReconstructionEstimator(
    const ReconstructionEstimatorOptions& options)
    : GlobalReconstructionEstimator(options) {
}

// The pipeline for estimating camera poses and structure is as follows:
//   1) Filter potentially bad pairwise geometries by enforcing a loop
//      constaint on rotations that form a triplet.
//   2) Initialize focal lengths.
//   3) Estimate the global rotation for each camera.
//   4) Remove any pairwise geometries where the relative rotation is not
//      consistent with the global rotation.
//   5) Optimize the relative translation given the known rotations.
//   6) Filter potentially bad relative translations.
//   7) Estimate positions.
//   8) Estimate structure.
//   9) Bundle adjustment.
//   10) Retriangulate, and bundle adjust.
//
// After each filtering step we remove any views which are no longer connected
// to the largest connected component in the view graph.
ReconstructionEstimatorSummary ExpGlobalReconstructionEstimator::Estimate(
    ViewGraph* view_graph, Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  view_graph_ = view_graph;
  orientations_.clear();
  positions_.clear();

  ReconstructionEstimatorSummary summary;
  ExpGlobalReconstructionEstimatorTimings global_estimator_timings;
  Timer total_timer;
  Timer timer;

  // Step 1. Filter the initial view graph and remove any bad two view
  // geometries.
  LOG(INFO) << "Filtering the intial view graph.";
  timer.Reset();
  if (!FilterInitialViewGraph()) {
    LOG(INFO) << "Insufficient view pairs to perform estimation.";
    return summary;
  }
  global_estimator_timings.initial_view_graph_filtering_time =
      timer.ElapsedTimeInSeconds();

  // Step 2. Calibrate any uncalibrated cameras.
  LOG(INFO) << "Calibrating any uncalibrated cameras.";
  timer.Reset();
  CalibrateCameras();
  summary.camera_intrinsics_calibration_time = timer.ElapsedTimeInSeconds();


  // @mhsung
  // Step 2-a. Filter initial orientation.
  FilterInitialOrientations();


  // Step 3. Estimate global rotations.
  LOG(INFO) << "Estimating the global rotations of all cameras.";
  timer.Reset();
  if (!EstimateGlobalRotations()) {
    LOG(WARNING) << "Rotation estimation failed!";
    summary.success = false;
    return summary;
  }
  global_estimator_timings.rotation_estimation_time =
      timer.ElapsedTimeInSeconds();

  // Step 4. Filter bad rotations.
  LOG(INFO) << "Filtering any bad rotation estimations.";
  timer.Reset();
  FilterRotations();
  global_estimator_timings.rotation_filtering_time =
      timer.ElapsedTimeInSeconds();


  // Initialize positions with origin.
  for (const auto& item : orientations_) {
    positions_[item.first] = Eigen::Vector3d::Zero();
  }

  // FIXME:
  // Remove the if-expression. Translation computation is not related with
  // bundle adjustment.
  if (options_.exp_global_run_bundle_adjustment) {
    // Step 5. Optimize relative translations.
    LOG(INFO) << "Optimizing the pairwise translation estimations.";
    timer.Reset();
    OptimizePairwiseTranslations();
    global_estimator_timings.relative_translation_optimization_time =
        timer.ElapsedTimeInSeconds();

    // Step 6. Filter bad relative translations.
    LOG(INFO) << "Filtering any bad relative translations.";
    timer.Reset();
    FilterRelativeTranslation();
    global_estimator_timings.relative_translation_filtering_time =
        timer.ElapsedTimeInSeconds();

    // Step 7. Estimate global positions.
    LOG(INFO) << "Estimating the positions of all cameras.";
    timer.Reset();
    if (!EstimatePosition()) {
      LOG(WARNING) << "Position estimation failed!";
      summary.success = false;
      return summary;
    }
    LOG(INFO) << positions_.size()
              << " camera positions were estimated successfully.";
    global_estimator_timings.position_estimation_time =
        timer.ElapsedTimeInSeconds();
  }

  summary.pose_estimation_time =
      global_estimator_timings.rotation_estimation_time +
      global_estimator_timings.rotation_filtering_time +
      global_estimator_timings.relative_translation_optimization_time +
      global_estimator_timings.relative_translation_filtering_time +
      global_estimator_timings.position_estimation_time;

  // Set the poses in the reconstruction object.
  SetReconstructionFromEstimatedPoses(orientations_,
                                      positions_,
                                      reconstruction_);

  if (options_.exp_global_run_bundle_adjustment) {
    // Always triangulate once, then retriangulate and remove outliers depending
    // on the reconstrucition estimator options.
    for (int i = 0; i < options_.num_retriangulation_iterations + 1; i++) {
      // Step 8. Triangulate features.
      LOG(INFO) << "Triangulating all features.";
      timer.Reset();
      EstimateStructure();
      summary.triangulation_time += timer.ElapsedTimeInSeconds();

      SetUnderconstrainedAsUnestimated(reconstruction_);

      // Step 9. Bundle Adjustment.
      LOG(INFO) << "Performing bundle adjustment.";
      timer.Reset();
      if (!BundleAdjustment()) {
        summary.success = false;
        LOG(WARNING) << "Bundle adjustment failed!";
        return summary;
      }
      summary.bundle_adjustment_time += timer.ElapsedTimeInSeconds();

      int num_points_removed = RemoveOutlierFeatures(
          options_.max_reprojection_error_in_pixels,
          options_.min_triangulation_angle_degrees,
          reconstruction_);
      LOG(INFO) << num_points_removed << " outlier points were removed.";
    }
  }

  // Set the output parameters.
  GetEstimatedViewsFromReconstruction(*reconstruction_,
                                      &summary.estimated_views);
  GetEstimatedTracksFromReconstruction(*reconstruction_,
                                       &summary.estimated_tracks);
  summary.success = true;
  summary.total_time = total_timer.ElapsedTimeInSeconds();

  // Output some timing statistics.
  std::ostringstream string_stream;
  string_stream
      << "Global Reconstruction Estimator timings:"
      << "\n\tInitial view graph filtering time = "
      << global_estimator_timings.initial_view_graph_filtering_time
      << "\n\tCamera intrinsic calibration time = "
      << summary.camera_intrinsics_calibration_time
      << "\n\tRotation estimation time = "
      << global_estimator_timings.rotation_estimation_time
      << "\n\tRotation filtering time = "
      << global_estimator_timings.rotation_filtering_time
      << "\n\tRelative translation optimization time = "
      << global_estimator_timings.relative_translation_optimization_time
      << "\n\tRelative translation filtering time = "
      << global_estimator_timings.relative_translation_filtering_time
      << "\n\tPosition estimation time = "
      << global_estimator_timings.position_estimation_time;
  summary.message = string_stream.str();

  return summary;
}

void ExpGlobalReconstructionEstimator::FilterInitialOrientations() {
  const double KErrorThreshold = 20.0;

  const auto& view_pairs = view_graph_->GetAllEdges();

  std::unordered_map<ViewId, Eigen::Vector3d> constrained_views;
  for (const ViewId view_id : reconstruction_->ViewIds()) {
    const View* view = reconstruction_->View(view_id);
    if (view->IsOrientationInitialized()) {
      constrained_views[view_id] = view->GetInitialOrientation();
    }
  }

  const int num_constrained_views = constrained_views.size();
  if (num_constrained_views == 0) return;

  std::unordered_map<ViewId, int> view_id_to_index;
  int index = 0;
  view_id_to_index.reserve(constrained_views.size());
  for (const auto& view : constrained_views) {
    view_id_to_index[view.first] = index;
    ++index;
  }

  // Accumulate pairwise errors to each view.
  Eigen::VectorXd accumulated_errors =
      Eigen::VectorXd::Zero(num_constrained_views);
  Eigen::VectorXi view_count_in_pairs =
      Eigen::VectorXi::Zero(num_constrained_views);

  for (const auto& view_pair : view_pairs) {
    const ViewId view_id_1 = view_pair.first.first;
    const ViewId view_id_2 = view_pair.first.second;
    // FIXME:
    // Use the numbers of point matches as weights.
    //const int num_verified_matches = view_pair.second.num_verified_matches;

    if (constrained_views.find(view_id_1) != constrained_views.end() &&
        constrained_views.find(view_id_2) != constrained_views.end()) {
      const Eigen::Vector3d R_12 = view_pair.second.rotation_2;
      const Eigen::Vector3d relative_rotation_error =
      ComputeRelativeRotationError(view_pair.second.rotation_2,
          constrained_views.at(view_id_1), constrained_views.at(view_id_2));

      // Measure error based on relative rotation angle.
      const double relative_rotation_angle = relative_rotation_error.norm();
      CHECK_LE(relative_rotation_angle, M_PI);
      const double error = relative_rotation_angle / M_PI * 180.0;

      // Accumulate error and count view graph edges for each view.
      const int index_1 = FindOrDie(view_id_to_index, view_id_1);
      const int index_2 = FindOrDie(view_id_to_index, view_id_1);
      accumulated_errors(index_1) += error;
      accumulated_errors(index_2) += error;
      view_count_in_pairs(index_1) += 1;
      view_count_in_pairs(index_2) += 1;
    }
  }

  // Calculate average of errors.
  const Eigen::VectorXd avg_errors =
      accumulated_errors.cwiseQuotient(view_count_in_pairs.cast<double>());

  std::unordered_map<ViewId, double> view_errors;
  view_errors.reserve(num_constrained_views);
  for (const auto& view : constrained_views) {
    const int index = FindOrDie(view_id_to_index, view.first);
    // Skip view with zero view graph edge count.
    if (view_count_in_pairs[index] > 0) {
      view_errors[view.first] = avg_errors[index];
    }
  }

  // FIXME:
  // Remove this.
  // Sort in descending order and report errors.
  std::vector< std::pair<ViewId, double> > sorted_view_errors;
  sorted_view_errors.reserve(view_errors.size());
  for (const auto& view_error: view_errors) {
    sorted_view_errors.emplace_back(view_error.first, view_error.second);
  }
  std::sort(sorted_view_errors.begin(), sorted_view_errors.end(),
            [](const std::pair<ViewId, double>& a,
               const std::pair<ViewId, double>& b) {
                return a.second > b.second; });

  VLOG(3) << "Initial orientation errors:";
  for (const auto& view_error: sorted_view_errors) {
    const std::string& name = reconstruction_->View(view_error.first)->Name();
    const double& error = view_error.second;
    if (error > KErrorThreshold) {
      VLOG(3) << "[" << name << "]:" << " error = " << error << "  - Unset";
    } else {
      VLOG(3) << "[" << name << "]:" << " error = " << error;
    }
  }

  // Unset initial orientation having large errors.
  for (const auto& view_error: view_errors) {
    const ViewId view_id = view_error.first;
    const double error = view_error.second;
    if (error > KErrorThreshold) {
      reconstruction_->MutableView(view_id)->RemoveInitialOrientation();
    }
  }
}

}  // namespace theia