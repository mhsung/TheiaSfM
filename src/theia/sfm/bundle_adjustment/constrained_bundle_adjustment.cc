// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "theia/sfm/bundle_adjustment/constrained_bundle_adjustment.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>
#include <include/theia/theia.h>

#include "theia/util/map_util.h"
#include "theia/util/timer.h"
#include "theia/sfm/bundle_adjustment/create_loss_function.h"
#include "theia/sfm/bundle_adjustment/bundle_adjustment.h"
#include "theia/sfm/camera/camera.h"
#include "theia/sfm/camera/reprojection_error.h"
#include "theia/sfm/reconstruction.h"
#include "theia/sfm/reconstruction_estimator_utils.h"
#include "theia/sfm/types.h"

// @mhsung
#include "theia/sfm/bundle_adjustment/bundle_object_rotation_error.h"
#include "theia/sfm/bundle_adjustment/bundle_object_position_error.h"
#include "theia/sfm/bundle_adjustment/bundle_consecutive_camera_error.h"


namespace theia {

namespace {

void SetSolverOptions(const BundleAdjustmentOptions& options,
                      ceres::Solver::Options* solver_options) {
  solver_options->linear_solver_type = options.linear_solver_type;
  solver_options->preconditioner_type = options.preconditioner_type;
  solver_options->visibility_clustering_type =
      options.visibility_clustering_type;
  solver_options->logging_type =
      options.verbose ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
  solver_options->num_threads = options.num_threads;
  solver_options->num_linear_solver_threads = options.num_threads;
  solver_options->max_num_iterations = options.max_num_iterations;
  solver_options->max_solver_time_in_seconds =
      options.max_solver_time_in_seconds;
  solver_options->use_inner_iterations = options.use_inner_iterations;
  solver_options->function_tolerance = options.function_tolerance;
  solver_options->gradient_tolerance = options.gradient_tolerance;
  solver_options->parameter_tolerance = options.parameter_tolerance;
  solver_options->max_trust_region_radius = options.max_trust_region_radius;

  // Solver options takes ownership of the ordering so that we can order the BA
  // problem by points and cameras.
  solver_options->linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
}

// Determine which camera intrinsics to optimize by investigating the individual
// bits of intrinsics_to_optimize.
std::vector<int> GetIntrinsicsToOptimize(
    const OptimizeIntrinsicsType& intrinsics_to_optimize) {
  std::vector<int> constant_intrinsics;
  if (intrinsics_to_optimize == OptimizeIntrinsicsType::ALL) {
    return constant_intrinsics;
  }

  if ((intrinsics_to_optimize &
       OptimizeIntrinsicsType::FOCAL_LENGTH) == OptimizeIntrinsicsType::NONE) {
    constant_intrinsics.emplace_back(Camera::FOCAL_LENGTH);
  }
  if ((intrinsics_to_optimize & OptimizeIntrinsicsType::ASPECT_RATIO) ==
      OptimizeIntrinsicsType::NONE) {
    constant_intrinsics.emplace_back(Camera::ASPECT_RATIO);
  }
  if ((intrinsics_to_optimize & OptimizeIntrinsicsType::SKEW) ==
      OptimizeIntrinsicsType::NONE) {
    constant_intrinsics.emplace_back(Camera::SKEW);
  }
  if ((intrinsics_to_optimize & OptimizeIntrinsicsType::PRINCIPAL_POINTS) ==
      OptimizeIntrinsicsType::NONE) {
    constant_intrinsics.emplace_back(Camera::PRINCIPAL_POINT_X);
    constant_intrinsics.emplace_back(Camera::PRINCIPAL_POINT_Y);
  }
  if ((intrinsics_to_optimize & OptimizeIntrinsicsType::RADIAL_DISTORTION) ==
      OptimizeIntrinsicsType::NONE) {
    constant_intrinsics.emplace_back(Camera::RADIAL_DISTORTION_1);
    constant_intrinsics.emplace_back(Camera::RADIAL_DISTORTION_2);
  }
  return constant_intrinsics;
}

// Adds camera intrinsic parameters to the problem while optionally holding some
// intrinsics parameters constant.
void AddCameraIntrinsicsToProblem(const std::vector<int>& constant_intrinsics,
                                  double* camera_intrinsics,
                                  ceres::Problem* problem) {
  if (constant_intrinsics.size() == Camera::kIntrinsicsSize) {
    problem->AddParameterBlock(camera_intrinsics, Camera::kIntrinsicsSize);
    problem->SetParameterBlockConstant(camera_intrinsics);
  } else if (constant_intrinsics.size() > 0) {
    ceres::SubsetParameterization* subset_parameterization =
        new ceres::SubsetParameterization(Camera::kIntrinsicsSize,
                                          constant_intrinsics);
    problem->AddParameterBlock(camera_intrinsics,
                               Camera::kIntrinsicsSize,
                               subset_parameterization);
  } else {
    problem->AddParameterBlock(camera_intrinsics, Camera::kIntrinsicsSize);
  }

  // Set bounds for certain camera parameters to make sure they are reasonable.
  problem->SetParameterLowerBound(camera_intrinsics, Camera::FOCAL_LENGTH, 0.0);
  problem->SetParameterLowerBound(camera_intrinsics, Camera::ASPECT_RATIO, 0.0);
}

// For each camera intrinsics group, choose one view to use as the reference for
// shared camera intrinsics.
std::unordered_map<CameraIntrinsicsGroupId, double*> GetSharedIntrinsicsMap(
    const std::unordered_set<ViewId>& view_ids,
    Reconstruction* reconstruction) {
  std::unordered_map<CameraIntrinsicsGroupId, double*>
      shared_intrinsics_by_group_id;

  // For each view, find its camera intrinsics group and provide a pointer to
  // the shared intrinsics if one has not already been provided.
  for (const ViewId view_id : view_ids) {
    View* view = CHECK_NOTNULL(reconstruction->MutableView(view_id));
    if (!view->IsEstimated()) {
      continue;
    }


    // Find the camera intrinsics group for this view.
    const CameraIntrinsicsGroupId intrinsics_group_id =
        reconstruction->CameraIntrinsicsGroupIdFromViewId(view_id);

    // Provide a pointer to the shared camera intrinsics if this group does not
    // have one.
    if (!ContainsKey(shared_intrinsics_by_group_id, intrinsics_group_id)) {
      shared_intrinsics_by_group_id[intrinsics_group_id] =
          view->MutableCamera()->mutable_intrinsics();
    }
  }
  return shared_intrinsics_by_group_id;
}

// Copy the output of the shared intrinsic camera parameters to all other
// cameras (including ones that were not optimized during BA) that share these
// intrinsics.
void CopySharedIntrinsicsToViews(
    const std::unordered_map<CameraIntrinsicsGroupId, double*>&
    shared_intrinsics_by_group_id,
    Reconstruction* reconstruction) {
  for (const auto& shared_intrinsics : shared_intrinsics_by_group_id) {
    const double* shared_intrinsics_for_group = shared_intrinsics.second;
    // Get all views in this intrinsics group.
    const auto& views_in_intrinsics_group =
        reconstruction->GetViewsInCameraIntrinsicGroup(shared_intrinsics.first);
    // For all views in this group, copy the shared intrinsics into the view's
    // intrinsics.
    for (const ViewId view_id : views_in_intrinsics_group) {
      Camera* camera = reconstruction->MutableView(view_id)->MutableCamera();
      double* intrinsics = camera->mutable_intrinsics();
      if (intrinsics == shared_intrinsics_for_group) {
        continue;
      } else {
        std::copy(shared_intrinsics_for_group,
                  shared_intrinsics_for_group + Camera::kIntrinsicsSize,
                  intrinsics);
      }
    }
  }
}

}  // namespace

// Bundle adjust the entire model.
BundleAdjustmentSummary ConstrainedBundleAdjustPartialReconstruction(
    const BundleAdjustmentOptions& options,
    const std::unordered_set<ViewId>& view_ids,
    const std::unordered_set<TrackId>& track_ids,
    Reconstruction* reconstruction,
    BundleObjectConstraints* object_constraints) {
  CHECK_NOTNULL(reconstruction);
  CHECK_NOTNULL(object_constraints);

  BundleAdjustmentSummary summary;
  static const int kTrackSize = 4;

  // Start setup timer.
  Timer timer;

  // Get the loss function that will be used for BA.
  ceres::Problem::Options problem_options;
  std::unique_ptr<ceres::LossFunction> loss_function =
      CreateLossFunction(options.loss_function_type, options.robust_loss_width);
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  // Set solver options.
  ceres::Solver::Options solver_options;
  SetSolverOptions(options, &solver_options);
  ceres::ParameterBlockOrdering* parameter_ordering =
      solver_options.linear_solver_ordering.get();

  // Obtain which params will be constant during optimization.
  const std::vector<int> constant_intrinsics =
      GetIntrinsicsToOptimize(options.intrinsics_to_optimize);

  // Get pointers to the shared camera intrinsics.
  std::unordered_map<CameraIntrinsicsGroupId, double*>
      shared_intrinsics_by_group_id =
          GetSharedIntrinsicsMap(view_ids, reconstruction);
  // Add all camera intrinsics to the problem.
  for (auto& shared_intrinsics : shared_intrinsics_by_group_id) {
    // This function will add all camera parameters to the problem and will keep
    // the intrinsic params constant if desired.
    AddCameraIntrinsicsToProblem(constant_intrinsics,
                                 shared_intrinsics.second,
                                 &problem);
  }


  // Per recommendation of Ceres documentation we group the parameters by points
  // (group 0) and camera parameters (group 1) so that the points are eliminated
  // first then the cameras.
  for (const ViewId view_id : view_ids) {
    View* view = CHECK_NOTNULL(reconstruction->MutableView(view_id));
    // Only optimize estimated views.
    if (!view->IsEstimated()) {
      continue;
    }

    // Add the camera extrinsic parameters to the problem.
    Camera* camera = view->MutableCamera();
    problem.AddParameterBlock(camera->mutable_extrinsics(),
                              Camera::kExtrinsicsSize);

    // Get a pointer to the shared camera intrinsics.
    const CameraIntrinsicsGroupId intrinsics_group_id =
        reconstruction->CameraIntrinsicsGroupIdFromViewId(view_id);
    double* shared_intrinsics =
        FindOrDie(shared_intrinsics_by_group_id, intrinsics_group_id);

    // Add camera parameters to groups 1 and 2. The extrinsics *must* belong to
    // group 2. This is because inner iterations uses a reverse ordering of
    // elimination and the Schur-based solvers require the first group to be an
    // independent set. Since the intrinsics may be shared, they are not
    // guaranteed to form an independent set and so we must use the extrinsics
    // in group 2.
    parameter_ordering->AddElementToGroup(camera->mutable_extrinsics(), 2);
    parameter_ordering->AddElementToGroup(shared_intrinsics, 1);

    // Add residuals for all tracks in the view.
    for (const TrackId track_id : view->TrackIds()) {
      const Feature* feature = CHECK_NOTNULL(view->GetFeature(track_id));
      Track* track = CHECK_NOTNULL(reconstruction->MutableTrack(track_id));
      // Only consider tracks with an estimated 3d point.
      if (!track->IsEstimated()) {
        continue;
      }

      problem.AddResidualBlock(
          ReprojectionError::Create(*feature),
          loss_function.get(),
          camera->mutable_extrinsics(),
          shared_intrinsics,
          track->MutablePoint()->data());
      // Add the point to group 0.
      parameter_ordering->AddElementToGroup(track->MutablePoint()->data(), 0);
      problem.SetParameterBlockConstant(track->MutablePoint()->data());
    }
  }

  // @mhsung
  // Object-view orientations.
  for (const auto& object : object_constraints->object_view_orientations_) {
    const ObjectId object_id = object.first;
    // FIXME:
    // Why does this happen?
    if (!ContainsKey(*object_constraints->object_orientations_, object_id)) {
      (*object_constraints->object_orientations_)[object_id] =
          Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d* object_orientation = FindOrNull(
        *object_constraints->object_orientations_, object_id);
    CHECK(object_orientation != nullptr);

    for (const auto& orientation : object.second) {
      const ViewId view_id = orientation.first;
      View* view = reconstruction->MutableView(view_id);
      if (view == nullptr || !view->IsEstimated()) {
        continue;
      }

      Camera* camera = view->MutableCamera();
      double* view_orientation = &(camera->mutable_extrinsics()[
          Camera::ExternalParametersIndex::ORIENTATION]);
      problem.AddResidualBlock(
          BundleObjectPositionError::Create(
              orientation.second, object_constraints->orientation_weight_),
          loss_function.get(),
          camera->mutable_extrinsics(),
          object_orientation->data());
    }

    problem.AddParameterBlock(object_orientation->data(), 3);
    parameter_ordering->AddElementToGroup(object_orientation->data(), 3);
  }

  // @mhsung
  // Object-view positions.
  object_constraints->object_positions_->clear();
  for (const auto& object :
      object_constraints->view_object_position_directions_) {
    const ObjectId object_id = object.first;
    // FIXME:
    // Why does this happen?
    if (!ContainsKey(*object_constraints->object_positions_, object_id)) {
      (*object_constraints->object_positions_)[object_id] =
          Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d* object_position = FindOrNull(
        *object_constraints->object_positions_, object_id);
    CHECK(object_position != nullptr);

    for (const auto& position_direction : object.second) {
      const ViewId view_id = position_direction.first;
      View* view = reconstruction->MutableView(view_id);
      if (view == nullptr || !view->IsEstimated()) {
        continue;
      }

      Camera* camera = view->MutableCamera();
      problem.AddResidualBlock(
          BundleObjectPositionError::Create(
              position_direction.second, object_constraints->position_weight_),
          loss_function.get(),
          camera->mutable_extrinsics(),
          object_position->data());
    }

    problem.AddParameterBlock(object_position->data(), 3);
    parameter_ordering->AddElementToGroup(object_position->data(), 3);
  }

  // @mhsung
  // Consecutive camera constraints.
  if (object_constraints->consecutive_camera_weight_ > 0.0) {
    // Sort View IDs.
    std::vector<ViewId> sorted_view_ids;
    sorted_view_ids.reserve(reconstruction->NumViews());
    for (const auto& view_id : reconstruction->ViewIds()) {
      sorted_view_ids.push_back(view_id);
    }
    std::sort(sorted_view_ids.begin(), sorted_view_ids.end());

    for (int i = 1; i < sorted_view_ids.size() - 1; i++) {
      const ViewId prev_view_id = sorted_view_ids[i - 1];
      const ViewId curr_view_id = sorted_view_ids[i];
      const ViewId next_view_id = sorted_view_ids[i + 1];

      // IMPORTANT NOTE:
      // Consider view IDs as frames.
      if (curr_view_id - prev_view_id >
          object_constraints->consecutive_camera_range_ ||
          next_view_id - curr_view_id >
          object_constraints->consecutive_camera_range_) {
        continue;
      }

      View* prev_view = reconstruction->MutableView(prev_view_id);
      View* curr_view = reconstruction->MutableView(curr_view_id);
      View* next_view = reconstruction->MutableView(next_view_id);
      if (prev_view == nullptr || !prev_view->IsEstimated() ||
          curr_view == nullptr || !curr_view->IsEstimated() ||
          next_view == nullptr || !next_view->IsEstimated()) {
        continue;
      }

      Camera* prev_camera = prev_view->MutableCamera();
      Camera* curr_camera = curr_view->MutableCamera();
      Camera* next_camera = next_view->MutableCamera();
      problem.AddResidualBlock(
          BundleConsecutiveCameraError::Create(
              object_constraints->consecutive_camera_weight_),
          loss_function.get(),
          prev_camera->mutable_extrinsics(),
          curr_camera->mutable_extrinsics(),
          next_camera->mutable_extrinsics());
    }
  }


  // The previous loop gives us residuals for all tracks in all the views that
  // we want to optimize. However, the tracks should still be constrained by
  // *all* views that observe it, not just the ones we want to optimize. Here,
  // we add in any views that were not part of the first loop and we keep them
  // constant during the optimization.
  for (const TrackId track_id : track_ids) {
    Track* track = CHECK_NOTNULL(reconstruction->MutableTrack(track_id));
    if (!track->IsEstimated()) {
      continue;
    }

    problem.AddParameterBlock(track->MutablePoint()->data(), kTrackSize);
    parameter_ordering->AddElementToGroup(track->MutablePoint()->data(), 0);
    problem.SetParameterBlockVariable(track->MutablePoint()->data());

    const auto& observed_view_ids = track->ViewIds();
    for (const ViewId view_id : observed_view_ids) {
      View* view = CHECK_NOTNULL(reconstruction->MutableView(view_id));
      // Only optimize estimated views that have not already been added.
      if (ContainsKey(view_ids, view_id) || !view->IsEstimated()) {
        continue;
      }

      const Feature* feature = CHECK_NOTNULL(view->GetFeature(track_id));
      Camera* camera = view->MutableCamera();
      const CameraIntrinsicsGroupId intrinsics_group_id =
        reconstruction->CameraIntrinsicsGroupIdFromViewId(view_id);
      const bool variable_shared_intrinsics =
          ContainsKey(shared_intrinsics_by_group_id, intrinsics_group_id);

      // To properly add the intrinsics, we need to know if the shared
      // intrinsics have already been added to the problem. If they have, then
      // we will keep the parameter block corresponding to the intrinsics
      // variable. If the shared intrinsics have not already been added in the
      // previous loop then they should remain constant.
      double* shared_intrinsics;
      if (variable_shared_intrinsics) {
        shared_intrinsics =
            FindOrDie(shared_intrinsics_by_group_id, intrinsics_group_id);
      } else {
        shared_intrinsics = camera->mutable_intrinsics();
      }
      problem.AddResidualBlock(
          ReprojectionError::Create(*feature),
          loss_function.get(),
          camera->mutable_extrinsics(),
          shared_intrinsics,
          track->MutablePoint()->data());

      // Add camera parameters to groups 1 and 2.
      parameter_ordering->AddElementToGroup(camera->mutable_extrinsics(), 2);
      parameter_ordering->AddElementToGroup(shared_intrinsics, 1);
      // Any camera that reaches this point was not part of the first loop, so
      // we do not want to optimize it.
      problem.SetParameterBlockConstant(camera->mutable_extrinsics());
      // Only set the parameter block to constant if the shared intrinsics are
      // not shared with cameras that are being optimized.
      if (!variable_shared_intrinsics) {
        problem.SetParameterBlockConstant(shared_intrinsics);
      }
    }
  }

  // NOTE: cmsweeney found a thread on the Ceres Solver email group that
  // indicated using the reverse BA order (i.e., using cameras then points) is a
  // good idea for inner iterations.
  if (solver_options.use_inner_iterations) {
    solver_options.inner_iteration_ordering.reset(
        new ceres::ParameterBlockOrdering(*parameter_ordering));
    solver_options.inner_iteration_ordering->Reverse();
  }

  // Solve the problem.
  const double internal_setup_time = timer.ElapsedTimeInSeconds();
  ceres::Solver::Summary solver_summary;
  ceres::Solve(solver_options, &problem, &solver_summary);
  LOG_IF(INFO, options.verbose) << solver_summary.FullReport();

  // Copy the shared intrinsics to all views that share those intrinsics.
  CopySharedIntrinsicsToViews(shared_intrinsics_by_group_id, reconstruction);

  // Set the BundleAdjustmentSummary.
  summary.setup_time_in_seconds =
      internal_setup_time + solver_summary.preprocessor_time_in_seconds;
  summary.solve_time_in_seconds = solver_summary.total_time_in_seconds;
  summary.initial_cost = solver_summary.initial_cost;
  summary.final_cost = solver_summary.final_cost;
  // This only indicates whether the optimization was successfully run and makes
  // no guarantees on the quality or convergence.
  summary.success = solver_summary.IsSolutionUsable();

  return summary;
}

// Bundle adjust the specified views and all tracks observed by those views.
BundleAdjustmentSummary ConstrainedBundleAdjustReconstruction(
    const BundleAdjustmentOptions& options,
    Reconstruction* reconstruction,
    BundleObjectConstraints* object_constraints) {
  CHECK_NOTNULL(reconstruction);
  CHECK_NOTNULL(object_constraints);

  const auto& view_ids = reconstruction->ViewIds();
  const auto& track_ids = reconstruction->TrackIds();
  const std::unordered_set<ViewId> view_ids_set(view_ids.begin(),
                                                view_ids.end());
  const std::unordered_set<TrackId> track_ids_set(track_ids.begin(),
                                                  track_ids.end());
  return ConstrainedBundleAdjustPartialReconstruction(
      options, view_ids_set, track_ids_set, reconstruction, object_constraints);
}

}  // namespace theia
