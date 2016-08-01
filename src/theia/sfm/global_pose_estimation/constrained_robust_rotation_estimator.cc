// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "theia/sfm/global_pose_estimation/constrained_robust_rotation_estimator.h"

#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>

#include "theia/math/l1_solver.h"
#include "theia/math/matrix/sparse_cholesky_llt.h"
#include "theia/math/util.h"
#include "theia/sfm/types.h"
#include "theia/util/hash.h"
#include "theia/util/map_util.h"

namespace theia {
namespace {

// Computes the absolute rotation error from the global rotations to the
// absolute rotation. The error is returned in angle axis form.
Eigen::Vector3d ComputeAbsoluteRotationError(
    const Eigen::Vector3d& absolute_rotation,
    const Eigen::Vector3d& rotation) {
  Eigen::Matrix3d absolute_rotation_matrix, rotation_matrix;
  ceres::AngleAxisToRotationMatrix(
      absolute_rotation.data(),
      ceres::ColumnMajorAdapter3x3(absolute_rotation_matrix.data()));
  ceres::AngleAxisToRotationMatrix(
      rotation.data(), ceres::ColumnMajorAdapter3x3(rotation_matrix.data()));

  // Compute the absolute rotation error.
  const Eigen::Matrix3d absolute_rotation_matrix_error =
      rotation_matrix * absolute_rotation_matrix.transpose();
  Eigen::Vector3d absolute_rotation_error;
  ceres::RotationMatrixToAngleAxis(
      ceres::ColumnMajorAdapter3x3(absolute_rotation_matrix_error.data()),
      absolute_rotation_error.data());
  return absolute_rotation_error;
}

}  // namespace

bool ConstrainedRobustRotationEstimator::EstimateRotations(
    const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
    const std::unordered_map<ViewId, Eigen::Vector3d>& constrained_views,
    std::unordered_map<ViewId, Eigen::Vector3d>* global_orientations) {
  view_pairs_ = &view_pairs;
  constrained_views_ = &constrained_views;
  global_orientations_ = global_orientations;

  // Use 'RobustRotationEstimator' if no constraint is given.
  CHECK(!constrained_views_->empty());

  // Check whether all constrained views exist in the given list.
  for (const auto& view : *constrained_views_) {
    FindOrDie(*global_orientations, view.first);
  }

  // If we have constraints, all view are used without fixing one frame as
  // the identity rotation.
  int index = 0;
  view_id_to_index_.reserve(global_orientations->size());
  for (const auto& orientation : *global_orientations) {
    view_id_to_index_[orientation.first] = index;
    ++index;
  }

  Eigen::SparseMatrix<double> sparse_mat;
  SetupLinearSystem();

  if (!SolveL1Regression()) {
    LOG(ERROR) << "Could not solve the L1 regression step.";
    return false;
  }

  if (!SolveIRLS()) {
    LOG(ERROR) << "Could not solve the least squares error step.";
    return false;
  }

  return true;
}

// Set up the sparse linear system.
void ConstrainedRobustRotationEstimator::SetupLinearSystem() {
  const int num_variables = global_orientations_->size() * 3;
  const int num_equations =
      (view_pairs_->size() + constrained_views_->size()) * 3;

  // The rotation change is one less than the number of global rotations because
  // we keep one rotation constant.
  rotation_change_.resize(num_variables);
  relative_rotation_error_.resize(num_equations);
  sparse_matrix_.resize(num_equations, num_variables);

  // For each relative rotation constraint, add an entry to the sparse
  // matrix. We use the first order approximation of angle axis such that:
  // R_ij = R_j - R_i. This makes the sparse matrix just a bunch of identity
  // matrices.
  int rotation_error_index = 0;
  std::vector<Eigen::Triplet<double> > triplet_list;
  for (const auto& view_pair : *view_pairs_) {
    const int view1_index =
        FindOrDie(view_id_to_index_, view_pair.first.first);
    if (view1_index != kConstantRotationIndex) {
      triplet_list.emplace_back(3 * rotation_error_index,
                                3 * view1_index,
                                -1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 1,
                                3 * view1_index + 1,
                                -1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 2,
                                3 * view1_index + 2,
                                -1.0);
    }

    const int view2_index =
        FindOrDie(view_id_to_index_, view_pair.first.second);
    if (view2_index != kConstantRotationIndex) {
      triplet_list.emplace_back(3 * rotation_error_index + 0,
                                3 * view2_index + 0,
                                1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 1,
                                3 * view2_index + 1,
                                1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 2,
                                3 * view2_index + 2,
                                1.0);
    }

    ++rotation_error_index;
  }

  // Add constraints.
  for (const auto& view : *constrained_views_) {
    const int view_index = FindOrDie(view_id_to_index_, view.first);
    if (view_index != kConstantRotationIndex) {
      triplet_list.emplace_back(3 * rotation_error_index + 0,
                                3 * view_index + 0,
                                constraint_weight_);
      triplet_list.emplace_back(3 * rotation_error_index + 1,
                                3 * view_index + 1,
                                constraint_weight_);
      triplet_list.emplace_back(3 * rotation_error_index + 2,
                                3 * view_index + 2,
                                constraint_weight_);
    }

    ++rotation_error_index;
  }

  sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

// Computes the relative rotation error based on the current global
// orientation estimates.
void ConstrainedRobustRotationEstimator::ComputeRotationError() {
  RobustRotationEstimator::ComputeRotationError();

  int rotation_error_index = view_pairs_->size();

  // Add constraints.
  for (const auto& view : *constrained_views_) {
    relative_rotation_error_.segment<3>(3 * rotation_error_index) =
        constraint_weight_ * ComputeAbsoluteRotationError(
            view.second, FindOrDie(*global_orientations_, view.first));
    ++rotation_error_index;
  }
}

}  // namespace theia
