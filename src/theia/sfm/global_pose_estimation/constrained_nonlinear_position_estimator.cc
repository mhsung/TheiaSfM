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

#include "theia/sfm/global_pose_estimation/single_translation_error.h"
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

Vector3d GetRotatedTranslation(const Vector3d& rotation_angle_axis,
                               const Vector3d& translation) {
  Matrix3d rotation;
  ceres::AngleAxisToRotationMatrix(
      rotation_angle_axis.data(),
      ceres::ColumnMajorAdapter3x3(rotation.data()));
  return rotation.transpose() * translation;
}

Vector3d GetRotatedFeatureRay(const Camera& camera,
                              const Vector3d& orientation,
                              const Feature& feature) {
  Camera temp_camera = camera;
  temp_camera.SetOrientationFromAngleAxis(orientation);
  // Get the image ray rotated into the world reference frame.
  return camera.PixelToUnitDepthRay(feature).normalized();
}

// Sorts the pairs such that the number of views (i.e. the int) is sorted in
// descending order.
bool CompareViewsPerTrack(const std::pair<TrackId, int>& t1,
                          const std::pair<TrackId, int>& t2) {
  return t1.second > t2.second;
}

}  // namespace

ConstrainedNonlinearPositionEstimator::ConstrainedNonlinearPositionEstimator(
    const NonlinearPositionEstimator::Options& options,
    const Reconstruction& reconstruction,
    const double constraint_weight)
    : NonlinearPositionEstimator(options, reconstruction),
      constraint_weight_(constraint_weight) {
}

bool ConstrainedNonlinearPositionEstimator::EstimatePositions(
    const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
    const std::unordered_map<ViewId, Vector3d>& orientations,
    const ObjectViewPositionDirections& object_view_constraints,
    std::unordered_map<ViewId, Vector3d>* positions) {
  CHECK_NOTNULL(positions);
  if (view_pairs.empty() || orientations.empty()) {
    VLOG(2) << "Number of view_pairs = " << view_pairs.size()
            << " Number of orientations = " << orientations.size();
    return false;
  }

  // @mhsung
  // Use 'NonlinearPositionEstimator' if no constraint is given.
  CHECK(!object_view_constraints.empty());

  triangulated_points_.clear();
  problem_.reset(new ceres::Problem());
  view_pairs_ = &view_pairs;

  // Iterative schur is only used if the problem is large enough, otherwise
  // sparse schur is used.
  static const int kMinNumCamerasForIterativeSolve = 1000;

  // Initialize positions to be random.
  // @mhsung
  InitializeRandomPositions(
      orientations, object_view_constraints, positions);

  // Add the constraints to the problem.
  AddCameraToCameraConstraints(orientations, positions);
  if (options_.min_num_points_per_view > 0) {
    AddPointToCameraConstraints(orientations, positions);
    AddCamerasAndPointsToParameterGroups(positions);
  }

  // @mhsung
  AddSingleCameraConstraints(
      orientations, object_view_constraints, positions);

  // @mhsung
  // If we have constraints, all view are used without fixing one frame as
  // the origin.
//  positions->begin()->second.setZero();
//  problem_->SetParameterBlockConstant(positions->begin()->second.data());

  // Set the solver options.
  ceres::Solver::Summary summary;
  solver_options_.num_threads = options_.num_threads;
  solver_options_.num_linear_solver_threads = options_.num_threads;
  solver_options_.max_num_iterations = options_.max_num_iterations;

  // Choose the type of linear solver. For sufficiently large problems, we want
  // to use iterative methods (e.g., Conjugate Gradient or Iterative Schur);
  // however, we only want to use a Schur solver if 3D points are used in the
  // optimization.
  if (positions->size() > kMinNumCamerasForIterativeSolve) {
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
    const std::unordered_map<ViewId, Vector3d>& orientations,
    const std::unordered_map<ViewId, Eigen::Vector3d>&
    constrained_position_dirs,
    std::unordered_map<ViewId, Vector3d>* positions) {
  std::unordered_set<ViewId> constrained_positions;
  constrained_positions.reserve(orientations.size());
  for (const auto& view_pair : *view_pairs_) {
    constrained_positions.insert(view_pair.first.first);
    constrained_positions.insert(view_pair.first.second);
  }

  positions->reserve(orientations.size());

  // @mhsung
  // Use fixed seed.
  InitRandomGenerator();
  for (const auto& orientation : orientations) {
    if (ContainsKey(constrained_positions, orientation.first)) {
      // Use initial position directions if given.
      const Eigen::Vector3d* init_position_dir =
          FindOrNull(constrained_position_dirs, orientation.first);

      if (init_position_dir != nullptr) {
        (*positions)[orientation.first] = 100.0 * (*init_position_dir);
      } else {
        (*positions)[orientation.first] =
            100.0 * RandVector3d().normalized();
      }
    }
  }
}

void ConstrainedNonlinearPositionEstimator::AddSingleCameraConstraints(
    const std::unordered_map<ViewId, Vector3d>& orientations,
    const std::unordered_map<ViewId, Vector3d>& constrained_position_dirs,
    std::unordered_map<ViewId, Vector3d>* positions) {
  for (auto& position : *positions) {
    const ViewId view_id = position.first;
    const View* view = reconstruction_.View(view_id);
    const Vector3d* position_direction =
        FindOrNull(constrained_position_dirs, view_id);
    if (view == nullptr || position_direction == nullptr) {
      continue;
    }

    ceres::CostFunction* cost_function =
        SingleTranslationError::Create(*position_direction, constraint_weight_);

    problem_->AddResidualBlock(cost_function,
                               new ceres::HuberLoss(options_.robust_loss_width),
                               position.second.data());
  }

  VLOG(2) << problem_->NumResidualBlocks() << " camera to camera constraints "
      "were added to the position "
      "estimation problem.";
}

}  // namespace theia
