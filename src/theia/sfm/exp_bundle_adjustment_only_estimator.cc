// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'global_reconstruction_estimator.cc'

#include "theia/sfm/exp_bundle_adjustment_only_estimator.h"

#include <Eigen/Core>
#include <memory>
#include <sstream>  // NOLINT

#include "theia/sfm/bundle_adjustment/bundle_adjustment.h"
#include "theia/sfm/estimate_track.h"
#include "theia/sfm/extract_maximally_parallel_rigid_subgraph.h"
#include "theia/sfm/filter_view_graph_cycles_by_rotation.h"
#include "theia/sfm/filter_view_pairs_from_orientation.h"
#include "theia/sfm/filter_view_pairs_from_relative_translation.h"
#include "theia/sfm/reconstruction.h"
#include "theia/sfm/reconstruction_estimator_options.h"
#include "theia/sfm/reconstruction_estimator_utils.h"
#include "theia/sfm/set_camera_intrinsics_from_priors.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/view_graph/orientations_from_maximum_spanning_tree.h"
#include "theia/sfm/view_graph/remove_disconnected_view_pairs.h"
#include "theia/sfm/view_graph/view_graph.h"
#include "theia/solvers/sample_consensus_estimator.h"
#include "theia/util/random.h"
#include "theia/util/timer.h"
// @mhsung
#include "theia/sfm/global_pose_estimation/constrained_robust_rotation_estimator.h"

namespace theia {

using Eigen::Vector3d;

namespace {

// All times are given in seconds.
struct ExpBundleAdjustmentOnlyEstimatorTimings {
  double initial_view_graph_filtering_time = 0.0;
  double camera_intrinsics_calibration_time = 0.0;
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

}  // namespace

ExpBundleAdjustmentOnlyEstimator::ExpBundleAdjustmentOnlyEstimator(
    const ReconstructionEstimatorOptions& options) {
  options_ = options;
  options_.nonlinear_position_estimator_options.num_threads =
      options_.num_threads;
  options_.linear_triplet_position_estimator_options.num_threads =
      options_.num_threads;
}

ReconstructionEstimatorSummary ExpBundleAdjustmentOnlyEstimator::Estimate(
    ViewGraph* view_graph, Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  view_graph_ = view_graph;

  // Assume that orientations and positions are already estimated and set in
  // reconstruction.

  ReconstructionEstimatorSummary summary;
  ExpBundleAdjustmentOnlyEstimatorTimings global_estimator_timings;
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

  // Always triangulate once, then retriangulate and remove outliers depending
  // on the reconstruciton estimator options.
  for (int i = 0; i < options_.num_retriangulation_iterations + 1; i++) {
    // Step 8. Triangulate features.
    LOG(INFO) << "Triangulating all features.";
    timer.Reset();
    EstimateStructure();
    summary.triangulation_time += timer.ElapsedTimeInSeconds();

//    // Set all tracks as estimated.
//    for (const auto& track_id : reconstruction_->TrackIds()) {
//      reconstruction_->MutableTrack(track_id)->SetEstimated(true);
//    }

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
      << summary.camera_intrinsics_calibration_time;
  summary.message = string_stream.str();

  return summary;
}

bool ExpBundleAdjustmentOnlyEstimator::FilterInitialViewGraph() {
  // Remove any view pairs that do not have a sufficient number of inliers.
  std::unordered_set<ViewIdPair> view_pairs_to_remove;
  const auto& view_pairs = view_graph_->GetAllEdges();
  for (const auto& view_pair : view_pairs) {
    if (view_pair.second.num_verified_matches <
        options_.min_num_two_view_inliers) {
      view_pairs_to_remove.insert(view_pair.first);
    }
  }
  for (const ViewIdPair view_id_pair : view_pairs_to_remove) {
    view_graph_->RemoveEdge(view_id_pair.first, view_id_pair.second);
  }

  // Only reconstruct the largest connected component.
  RemoveDisconnectedViewPairs(view_graph_);
  return view_graph_->NumEdges() >= 1;
}

void ExpBundleAdjustmentOnlyEstimator::CalibrateCameras() {
  SetCameraIntrinsicsFromPriors(reconstruction_);
}

void ExpBundleAdjustmentOnlyEstimator::EstimateStructure() {
  // Estimate all tracks.
  TrackEstimator::Options triangulation_options;
  triangulation_options.max_acceptable_reprojection_error_pixels =
      options_.triangulation_max_reprojection_error_in_pixels;
  triangulation_options.min_triangulation_angle_degrees =
      options_.min_triangulation_angle_degrees;
  triangulation_options.bundle_adjustment = options_.bundle_adjust_tracks;
  triangulation_options.ba_options = SetBundleAdjustmentOptions(options_, 0);
  triangulation_options.ba_options.num_threads = 1;
  triangulation_options.ba_options.verbose = false;
  triangulation_options.num_threads = options_.num_threads;
  TrackEstimator track_estimator(triangulation_options, reconstruction_);
  const TrackEstimator::Summary summary = track_estimator.EstimateAllTracks();
}

bool ExpBundleAdjustmentOnlyEstimator::BundleAdjustment() {
  int num_estimated_views = 0;
  for (const auto& view_id : reconstruction_->ViewIds()) {
    const View* view = reconstruction_->View(view_id);
    if (view != nullptr && view->IsEstimated()) {
      ++num_estimated_views;
    }
  }

  // Bundle adjustment.
  bundle_adjustment_options_ =
      SetBundleAdjustmentOptions(options_, num_estimated_views);
  const auto& bundle_adjustment_summary =
      BundleAdjustReconstruction(bundle_adjustment_options_, reconstruction_);
  return bundle_adjustment_summary.success;
}

}  // namespace theia
