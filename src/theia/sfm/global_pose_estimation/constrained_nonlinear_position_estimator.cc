// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'nonlinear_position_estimator.cc'

#include "theia/sfm/global_pose_estimation/constrained_nonlinear_position_estimator.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "theia/sfm/global_pose_estimation/pairwise_translation_error.h"
#include "theia/sfm/reconstruction.h"
#include "theia/sfm/types.h"
#include "theia/util/map_util.h"
// @mhsung
#include "theia/util/random.h"
#include "theia/util/util.h"

namespace theia {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;

}  // namespace

bool ConstrainedNonlinearPositionEstimator::SetObjectViewConstraints(
    const std::unordered_map<ViewId, Eigen::Vector3d>& view_orientations,
    const std::unordered_map<ObjectId, ViewObjectPositionDirections>&
    view_object_constraints,
    const std::unordered_map<ObjectId, ViewObjectPositionDirectionWeights>*
    view_object_constraint_weights) {
  int num_object_view_pairs = 0;
  view_object_constraints_.clear();
  view_object_constraint_weights_.clear();

  bool use_per_constraint_weights = true;
  if (!view_object_constraint_weights) {
    LOG(INFO) << "Position direction weights were not given. Use default: "
              << constraint_weight_multiplier_;
    use_per_constraint_weights = false;
  } else {
    LOG(INFO) << "Use per-constraint position direction weights.";
  }

  for (const auto& object : view_object_constraints) {
    const ObjectId object_id = object.first;

    // Check whether the constrained views exist in the view orientation list.
    for (const auto& constraint : object.second) {
      const ViewId view_id = constraint.first;

      if (ContainsKey(view_orientations, constraint.first)) {
        view_object_constraints_[object_id].emplace(constraint);

        // Set weights.
        if (use_per_constraint_weights) {
          const auto& object_weights =
              FindOrNull(*view_object_constraint_weights, object_id);
          CHECK(object_weights) << "Position direction weights for object "
                                << object_id << " do not exist.";
          const double* weight = FindOrNull(*object_weights, view_id);
          CHECK(weight) << "Position direction Weight for object " << object_id
                        << " and view " << view_id << "pair does not exist.";

          const double scaled_weight =
              constraint_weight_multiplier_ * (*weight);
          view_object_constraint_weights_[object_id].emplace(
              view_id, scaled_weight);
        } else {
          // Use default weight.
          view_object_constraint_weights_[object_id].emplace(
              view_id, constraint_weight_multiplier_);
        }

        ++num_object_view_pairs;
      } else {
        LOG(WARNING) << "View " << view_id << " does not exist.";
      }
    }
  }

  return (num_object_view_pairs > 0);
}

ConstrainedNonlinearPositionEstimator::ConstrainedNonlinearPositionEstimator(
    const NonlinearPositionEstimator::Options& options,
    const Reconstruction& reconstruction,
    const double constraint_weight_multiplier)
    : NonlinearPositionEstimator(options, reconstruction),
      constraint_weight_multiplier_(constraint_weight_multiplier) {
}

bool ConstrainedNonlinearPositionEstimator::EstimatePositions(
    const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
    const std::unordered_map<ViewId, Vector3d>& view_orientations,
    const std::unordered_map<ObjectId, ViewObjectPositionDirections>&
    view_object_constraints,
    std::unordered_map<ViewId, Vector3d>* view_positions,
    std::unordered_map<ViewId, Vector3d>* object_positions,
    const std::unordered_map<ObjectId, ViewObjectPositionDirectionWeights>*
    view_object_constraint_weights,
    bool randomly_initialize) {
  CHECK_NOTNULL(view_positions);
  CHECK_NOTNULL(object_positions);
  if (view_pairs.empty() || view_orientations.empty()) {
    VLOG(2) << "Number of view_pairs = " << view_pairs.size()
            << " Number of orientations = " << view_orientations.size();
    return false;
  }

  triangulated_points_.clear();
  problem_.reset(new ceres::Problem());
  view_pairs_ = &view_pairs;

  // @mhsung
  CHECK(SetObjectViewConstraints(view_orientations, view_object_constraints,
                                 view_object_constraint_weights))
  << "No initial position direction is given. Re-run with 'NONLINEAR' option.";

  // Iterative schur is only used if the problem is large enough, otherwise
  // sparse schur is used.
  static const int kMinNumCamerasForIterativeSolve = 1000;

  // Initialize positions to be random.
  // @mhsung
  if (randomly_initialize) {
    InitializeRandomPositions(view_orientations, view_positions,
                              object_positions);
  }

  // Add the constraints to the problem.
  AddCameraToCameraConstraints(view_orientations, view_positions);

  // @mhsung
  AddCameraToObjectConstraints(
    view_orientations, view_positions, object_positions);

  if (options_.min_num_points_per_view > 0) {
    AddPointToCameraConstraints(view_orientations, view_positions);

    // @mhsung
    AddCamerasAndPointsToParameterGroups(view_positions, object_positions);
  }

  // @mhsung
  if (options_.consecutive_camera_position_constraint_weight > 0.0) {
    AddConsecutiveCameraConstraints(view_positions);
  }

  // @mhsung
  // Fix the first object frame as the origin.
  if (randomly_initialize) {
    object_positions->begin()->second.setZero();
  }
  problem_->SetParameterBlockConstant(object_positions->begin()->second.data());

  // Set the solver options.
  ceres::Solver::Summary summary;
  solver_options_.num_threads = options_.num_threads;
  solver_options_.num_linear_solver_threads = options_.num_threads;
  solver_options_.max_num_iterations = options_.max_num_iterations;

  // Choose the type of linear solver. For sufficiently large problems, we want
  // to use iterative methods (e.g., Conjugate Gradient or Iterative Schur);
  // however, we only want to use a Schur solver if 3D points are used in the
  // optimization.
  if (view_positions->size() > kMinNumCamerasForIterativeSolve) {
    if (options_.min_num_points_per_view > 0) {
      solver_options_.linear_solver_type = ceres::ITERATIVE_SCHUR;
      solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;
    } else {
      solver_options_.linear_solver_type = ceres::CGNR;
      solver_options_.preconditioner_type = ceres::JACOBI;
    }
  } else {
    if (options_.min_num_points_per_view > 0) {
      solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
      solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
  }

  ceres::Solve(solver_options_, problem_.get(), &summary);
  LOG(INFO) << summary.FullReport();
  return summary.IsSolutionUsable();
}

void ConstrainedNonlinearPositionEstimator::InitializeRandomPositions(
    const std::unordered_map<ViewId, Vector3d>& view_orientations,
    std::unordered_map<ViewId, Vector3d>* view_positions,
    std::unordered_map<ViewId, Vector3d>* object_positions) {
  NonlinearPositionEstimator::InitializeRandomPositions(
      view_orientations, view_positions);

  object_positions->reserve(view_object_constraints_.size());

  // Random seed is set in
  // 'NonlinearPositionEstimator::InitializeRandomPositions()'.
  for (const auto& object : view_object_constraints_) {
    (*object_positions)[object.first] = 100.0 * RandVector3d();
  }
}

void ConstrainedNonlinearPositionEstimator::AddCameraToObjectConstraints(
  const std::unordered_map<ViewId, Vector3d>& view_orientations,
  std::unordered_map<ViewId, Vector3d>* view_positions,
  std::unordered_map<ViewId, Vector3d>* object_positions) {
  int num_object_to_camera_constraints = 0;

  for (const auto& object : view_object_constraints_) {
    const ObjectId object_id = object.first;
    Vector3d* object_position = FindOrNull(*object_positions, object_id);
    CHECK(object_position);

    for (const auto& position_direction : object.second) {
      const ViewId view_id = position_direction.first;
      Vector3d* view_position = FindOrNull(*view_positions, view_id);
      if (view_position == nullptr) {
        continue;
      }

      // Rotate the relative translation so that it is aligned to the global
      // orientation frame.
      const Vector3d translation_direction = GetRotatedTranslation(
          FindOrDie(view_orientations, view_id), position_direction.second);

      const double weight = FindOrDie(FindOrDie(
          view_object_constraint_weights_, object_id), view_id);
      ceres::CostFunction* cost_function =
          PairwiseTranslationError::Create(translation_direction, weight);

      problem_->AddResidualBlock(cost_function,
                                 new ceres::HuberLoss(
                                     options_.robust_loss_width),
                                 view_position->data(),
                                 object_position->data());

      ++num_object_to_camera_constraints;
    }
  }

  VLOG(2) << num_object_to_camera_constraints << " camera to camera "
      "constraints were added to the position estimation problem.";
}

void ConstrainedNonlinearPositionEstimator
::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<ViewId, Vector3d>* view_positions,
    std::unordered_map<ViewId, Vector3d>* object_positions) {
  NonlinearPositionEstimator::AddCamerasAndPointsToParameterGroups
      (view_positions);

  ceres::ParameterBlockOrdering* parameter_ordering =
      solver_options_.linear_solver_ordering.get();

  // Add camera parameters to group 1.
  for (auto& position : *object_positions) {
    parameter_ordering->AddElementToGroup(position.second.data(), 1);
  }
}

}  // namespace theia
