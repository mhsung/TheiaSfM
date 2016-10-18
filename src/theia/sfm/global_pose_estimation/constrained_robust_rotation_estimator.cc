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

bool ConstrainedRobustRotationEstimator::SetObjectViewConstraints(
    const std::unordered_map<ViewId, Eigen::Vector3d>& view_orientations,
    const std::unordered_map<ObjectId, ObjectViewOrientations>&
    object_view_constraints,
    const std::unordered_map<ObjectId, ObjectViewOrientationWeights>*
    object_view_constraint_weights) {
  int num_object_view_pairs = 0;
  object_view_constraints_.clear();
  object_view_constraint_weights_.clear();

  bool use_per_constraint_weight = true;
  if (!object_view_constraint_weights ||
      object_view_constraint_weights->empty()) {
    LOG(INFO) << "Orientation weights were not given. Use default: "
              << constraint_default_weight_;
    use_per_constraint_weight = false;
  } else {
    LOG(INFO) << "Use per-constraint orientation weights.";
  }

  for (const auto& object : object_view_constraints) {
    const ObjectId object_id = object.first;

    // Check whether the constrained views exist in the view orientation list.
    for (const auto& constraint : object.second) {
      const ViewId view_id = constraint.first;

      if (ContainsKey(view_orientations, view_id)) {
        object_view_constraints_[object_id].emplace(constraint);

        // Set weights.
        if (use_per_constraint_weight) {
          const auto& object_weights =
              FindOrNull(*object_view_constraint_weights, object_id);
          CHECK(object_weights) << "Orientation weights for object "
                                << object_id << " do not exist.";
          const double* weight = FindOrNull(*object_weights, view_id);
          CHECK(weight) << "Orientation Weight for object " << object_id
                        << " and view " << view_id << "pair does not exist.";
          object_view_constraint_weights_[object_id].emplace(view_id, *weight);
        } else {
          // Use default weight.
          object_view_constraint_weights_[object_id].emplace(
              view_id, constraint_default_weight_);
        }

        ++num_object_view_pairs;
      } else {
        LOG(WARNING) << "View " << view_id << " does not exist.";
      }
    }
  }

  return (num_object_view_pairs > 0);
}

bool ConstrainedRobustRotationEstimator::EstimateRotations(
    const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
    const std::unordered_map<ObjectId, ObjectViewOrientations>&
    object_view_constraints,
    std::unordered_map<ViewId, Eigen::Vector3d>* global_view_orientations,
    std::unordered_map<ObjectId, Eigen::Vector3d>* global_object_orientations,
    const std::unordered_map<ObjectId, ObjectViewOrientationWeights>*
    object_view_constraint_weights) {
  CHECK_NOTNULL(global_view_orientations);
  CHECK_NOTNULL(global_object_orientations);

  view_pairs_ = &view_pairs;
  global_view_orientations_ = global_view_orientations;
  global_object_orientations_ = global_object_orientations;

  // @mhsung
  CHECK(SetObjectViewConstraints(
      *global_view_orientations, object_view_constraints,
      object_view_constraint_weights))
  << "No initial orientation is given. Re-run with 'ROBUST_L1L2' option.";

  // @mhsung
  // Fix the orientation of the first object by assigning -1 index.
  int index = -1;
  object_id_to_index_.reserve(global_object_orientations_->size());
  for (const auto& object : *global_object_orientations_) {
    object_id_to_index_[object.first] = index;
    ++index;
  }

  view_id_to_index_.reserve(global_view_orientations_->size());
  for (const auto& orientation : *global_view_orientations_) {
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
  // if (!SolveIRLS()) {
  //   LOG(ERROR) << "Could not solve the least squares error step.";
  //   return false;
  // }

  return true;
}

// Set up the sparse linear system.
void ConstrainedRobustRotationEstimator::SetupLinearSystem() {
  const size_t num_views = global_view_orientations_->size();
  const size_t num_view_pairs = view_pairs_->size();

  const size_t num_objects = object_view_constraints_.size();
  size_t num_object_view_pairs = 0;
  for (const auto& object : object_view_constraints_) {
    num_object_view_pairs += object.second.size();
  }

  // The rotation change is one less than the number of global rotations because
  // we keep one rotation constant.
  const int num_variables = static_cast<int>((num_views + num_objects - 1) * 3);
  const int num_equations = static_cast<int>(
      (num_view_pairs + num_object_view_pairs) * 3);

  rotation_change_.resize(num_variables);
  relative_rotation_error_.resize(num_equations);
  sparse_matrix_.resize(num_equations, num_variables);
  weight_vector_ = Eigen::VectorXd::Ones(num_equations);

  // @mhsung.
  std::vector<Eigen::Triplet<double> > triplet_list;
  FillLinearSystemTripletList(&triplet_list);

  sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

void ConstrainedRobustRotationEstimator::FillLinearSystemTripletList(
    std::vector<Eigen::Triplet<double> >* triplet_list) {
  RobustRotationEstimator::FillLinearSystemTripletList(triplet_list);

  int rotation_error_index = static_cast<int>(view_pairs_->size());

  // Add constraints.
  for (const auto& object : object_view_constraints_) {
    const ObjectId object_id = object.first;
    const int object_index = FindOrDie(object_id_to_index_, object_id);

    for (const auto& orientation : object.second) {
      if (object_index != kConstantRotationIndex) {
        triplet_list->emplace_back(3 * rotation_error_index + 0,
                                   3 * object_index + 0,
                                   -1.0);
        triplet_list->emplace_back(3 * rotation_error_index + 1,
                                   3 * object_index + 1,
                                   -1.0);
        triplet_list->emplace_back(3 * rotation_error_index + 2,
                                   3 * object_index + 2,
                                   -1.0);
      }

      const ViewId view_id = orientation.first;
      const int view_index = FindOrDie(view_id_to_index_, view_id);
      CHECK (view_index != kConstantRotationIndex);
      triplet_list->emplace_back(3 * rotation_error_index + 0,
                                 3 * view_index + 0,
                                 1.0);
      triplet_list->emplace_back(3 * rotation_error_index + 1,
                                 3 * view_index + 1,
                                 1.0);
      triplet_list->emplace_back(3 * rotation_error_index + 2,
                                 3 * view_index + 2,
                                 1.0);

      // Set weights.
      const double weight = FindOrDie(FindOrDie(
          object_view_constraint_weights_, object_id), view_id);
      weight_vector_(3 * rotation_error_index + 0) = weight;
      weight_vector_(3 * rotation_error_index + 1) = weight;
      weight_vector_(3 * rotation_error_index + 2) = weight;

      ++rotation_error_index;
    }
  }
}

// Computes the relative rotation error based on the current global
// orientation estimates.
void ConstrainedRobustRotationEstimator::ComputeRotationError() {
  RobustRotationEstimator::ComputeRotationError();

  int rotation_error_index = static_cast<int>(view_pairs_->size());

  // Add constraints.
  for (const auto& object : object_view_constraints_) {
    for (const auto& orientation : object.second) {
      relative_rotation_error_.segment<3>(3 * rotation_error_index) =
          ComputeRelativeRotationError(
              orientation.second,
              FindOrDie(*global_object_orientations_, object.first),
              FindOrDie(*global_view_orientations_, orientation.first));

      ++rotation_error_index;
    }
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

// Update the global orientations using the current value in the
// rotation_change.
void ConstrainedRobustRotationEstimator::UpdateGlobalRotations() {
  RobustRotationEstimator::UpdateGlobalRotations();

  for (auto& rotation : *global_object_orientations_) {
    const int object_index = FindOrDie(object_id_to_index_, rotation.first);
    if (object_index == kConstantRotationIndex) {
      continue;
    }

    // Apply the rotation change to the global orientation.
    const Eigen::Vector3d& rotation_change =
        rotation_change_.segment<3>(3 * object_index);
    ApplyRotation(rotation_change, &rotation.second);
  }
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
