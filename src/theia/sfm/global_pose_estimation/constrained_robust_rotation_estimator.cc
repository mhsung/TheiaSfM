// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'robust_rotation_estimator.cc'

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
    const std::unordered_map<ViewId, Eigen::Vector3d>& constrained_orientations,
    std::unordered_map<ViewId, Eigen::Vector3d>* global_orientations) {
  view_pairs_ = &view_pairs;
  constrained_orientations_ = &constrained_orientations;
  global_orientations_ = global_orientations;

  // @mhsung
  // Use 'RobustRotationEstimator' if no constraint is given.
  CHECK(!constrained_orientations_->empty());

  // Check whether all constrained views exist in the given list.
  for (const auto& orientation : *constrained_orientations_) {
    FindOrDie(*global_orientations, orientation.first);
  }

  // @mhsung
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

  // @mhsung
  // Do not use IRLS.
  // FIXME:
  // Fix IRLS to use weights properly.
//  if (!SolveIRLS()) {
//    LOG(ERROR) << "Could not solve the least squares error step.";
//    return false;
//  }

  return true;
}

// Set up the sparse linear system.
void ConstrainedRobustRotationEstimator::SetupLinearSystem() {
  const int num_variables = static_cast<int>(global_orientations_->size() * 3);
  const int num_equations = static_cast<int>(
      (view_pairs_->size() + constrained_orientations_->size()) * 3);

  // The rotation change is one less than the number of global rotations because
  // we keep one rotation constant.
  rotation_change_.resize(num_variables);
  relative_rotation_error_.resize(num_equations);
  sparse_matrix_.resize(num_equations, num_variables);

  // @mhsung.
  std::vector<Eigen::Triplet<double> > triplet_list;
  FillLinearSystemTripletList(&triplet_list);

  sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());

  weight_vector_ = Eigen::VectorXd::Ones(num_equations);
  // Set weights for constraints.
  weight_vector_.tail(constrained_orientations_->size() * 3).setConstant(
      constraint_weight_);
}

void ConstrainedRobustRotationEstimator::FillLinearSystemTripletList(
    std::vector<Eigen::Triplet<double> >* triplet_list) {
  RobustRotationEstimator::FillLinearSystemTripletList(triplet_list);

  int rotation_error_index = static_cast<int>(view_pairs_->size());

  // Add constraints.
  for (const auto& orientation : *constrained_orientations_) {
    const int view_index = FindOrDie(view_id_to_index_, orientation.first);
    if (view_index != kConstantRotationIndex) {
      triplet_list->emplace_back(3 * rotation_error_index + 0,
                                 3 * view_index + 0,
                                 1.0);
      triplet_list->emplace_back(3 * rotation_error_index + 1,
                                 3 * view_index + 1,
                                 1.0);
      triplet_list->emplace_back(3 * rotation_error_index + 2,
                                 3 * view_index + 2,
                                 1.0);
    }

    ++rotation_error_index;
  }
}

// Computes the relative rotation error based on the current global
// orientation estimates.
void ConstrainedRobustRotationEstimator::ComputeRotationError() {
  RobustRotationEstimator::ComputeRotationError();

  int rotation_error_index = static_cast<int>(view_pairs_->size());

  // Add constraints.
  for (const auto& orientation : *constrained_orientations_) {
    relative_rotation_error_.segment<3>(3 * rotation_error_index) =
        ComputeAbsoluteRotationError(
            orientation.second,
            FindOrDie(*global_orientations_, orientation.first));
    ++rotation_error_index;
  }
}

bool ConstrainedRobustRotationEstimator::SolveL1Regression() {
  static const double kConvergenceThreshold = 1e-3;

  L1Solver<Eigen::SparseMatrix<double> >::Options options;
  options.max_num_iterations = 5;
  // @mhsung
  // Set weight.
  L1Solver<Eigen::SparseMatrix<double> > l1_solver(
      options, weight_vector_.asDiagonal() * sparse_matrix_);

  rotation_change_.setZero();
  for (int i = 0; i < options_.max_num_l1_iterations; i++) {
    ComputeRotationError();
    // @mhsung
    // Set weight.
    l1_solver.Solve(weight_vector_.asDiagonal() * relative_rotation_error_,
                    &rotation_change_);
    UpdateGlobalRotations();

    if (relative_rotation_error_.norm() < kConvergenceThreshold) {
      break;
    }
    options.max_num_iterations *= 2;
    l1_solver.SetMaxIterations(options.max_num_iterations);
  }
  return true;
}

bool ConstrainedRobustRotationEstimator::SolveIRLS() {
  static const double kConvergenceThreshold = 1e-3;
  // This is the point where the Huber-like cost function switches from L1 to
  // L2.
  static const double kSigma = DegToRad(5.0);

  // Set up the linear solver and analyze the sparsity pattern of the
  // system. Since the sparsity pattern will not change with each linear solve
  // this can help speed up the solution time.
  SparseCholeskyLLt linear_solver;
  linear_solver.AnalyzePattern(sparse_matrix_.transpose() * sparse_matrix_);
  if (linear_solver.Info() != Eigen::Success) {
    LOG(ERROR) << "Cholesky decomposition failed.";
    return false;
  }

  VLOG(2) << "Iteration   Error           Delta";
  const std::string row_format = "  % 4d     % 4.4e     % 4.4e";

  Eigen::ArrayXd errors, weights;
  Eigen::SparseMatrix<double> at_weight;
  for (int i = 0; i < options_.max_num_irls_iterations; i++) {
    const Eigen::VectorXd prev_rotation_change = rotation_change_;
    ComputeRotationError();

    // Compute the weights for each error term.
    errors =
        (sparse_matrix_ * rotation_change_ - relative_rotation_error_).array();
    weights = kSigma / (errors.square() + kSigma * kSigma).square();

    // Update the factorization for the weighted values.
    at_weight =
        sparse_matrix_.transpose() * weights.matrix().asDiagonal();
    linear_solver.Factorize(at_weight * sparse_matrix_);
    if (linear_solver.Info() != Eigen::Success) {
      LOG(ERROR) << "Failed to factorize the least squares system.";
      return false;
    }

    // Solve the least squares problem.
    rotation_change_ =
        linear_solver.Solve(at_weight * relative_rotation_error_);
    if (linear_solver.Info() != Eigen::Success) {
      LOG(ERROR) << "Failed to solve the least squares system.";
      return false;
    }

    UpdateGlobalRotations();

    // Log some statistics for the output.
    const double rotation_change_sq_norm =
        (prev_rotation_change - rotation_change_).squaredNorm();
    VLOG(2) << StringPrintf(row_format.c_str(), i, errors.square().sum(),
                            rotation_change_sq_norm);
    if (rotation_change_sq_norm < kConvergenceThreshold) {
      VLOG(1) << "IRLS Converged in " << i + 1 << " iterations.";
      break;
    }
  }
  return true;
}

}  // namespace theia
