// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'robust_rotation_estimator.h'

#ifndef THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_ROBUST_ROTATION_ESTIMATOR_H_
#define THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_ROBUST_ROTATION_ESTIMATOR_H_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <list>
#include <unordered_map>

#include "theia/sfm/global_pose_estimation/robust_rotation_estimator.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/types.h"
#include "theia/util/hash.h"
// @mhsung
#include "theia/sfm/object_view_constraint.h"

namespace theia {

class ConstrainedRobustRotationEstimator : public RobustRotationEstimator {
 public:
  explicit ConstrainedRobustRotationEstimator(
      const Options& options, const double constraint_weight_multiplier)
    : RobustRotationEstimator(options),
      constraint_weight_multiplier_(constraint_weight_multiplier) {}

  // Estimates the global orientations of all views based on an initial
  // guess. Returns true on successful estimation and false otherwise.
  bool EstimateRotations(
      const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
      const std::unordered_map<ObjectId, ObjectViewOrientations>&
      object_view_constraints,
      std::unordered_map<ViewId, Eigen::Vector3d>* global_view_orientations,
      std::unordered_map<ObjectId, Eigen::Vector3d>*
      global_object_orientations,
      const std::unordered_map<ObjectId, ObjectViewOrientationWeights>*
      object_view_constraint_weights = nullptr);

 protected:
  // @mhsung
  bool SetObjectViewConstraints(
      const std::unordered_map<ViewId, Eigen::Vector3d>& view_orientations,
      const std::unordered_map<ObjectId, ObjectViewOrientations>&
      object_view_constraints,
      const std::unordered_map<ObjectId, ObjectViewOrientationWeights>*
      object_view_constraint_weights = nullptr);

  // Sets up the sparse linear system such that dR_ij = dR_j - dR_i. This is the
  // first-order approximation of the angle-axis rotations. This should only be
  // called once.
  virtual void SetupLinearSystem();

  // @mhsung.
  virtual void FillLinearSystemTripletList(
      std::vector<Eigen::Triplet<double> >* triplet_list);

  // Computes the relative rotation error based on the current global
  // orientation estimates.
  virtual void ComputeRotationError();

  // Performs the L1 robust loss minimization.
  virtual bool SolveL1Regression();

  // Performs the iteratively reweighted least squares.
  virtual bool SolveIRLS();

  // Updates the global rotations based on the current rotation change.
  virtual void UpdateGlobalRotations();

  Eigen::VectorXd weight_vector_;

  // Multiplied by per object-view weights if provided, or used as a constant
  // weight.
  const double constraint_weight_multiplier_;

  std::unordered_map<ObjectId, ObjectViewOrientations> object_view_constraints_;

  std::unordered_map<ObjectId, ObjectViewOrientationWeights>
      object_view_constraint_weights_;

  // The global orientation estimates for each object.
  std::unordered_map<ObjectId, Eigen::Vector3d>* global_object_orientations_;

  // Map of ViewIds to the corresponding positions of the object's
  // orientation in the linear system.
  std::unordered_map<ObjectId, int> object_id_to_index_;
};

}  // namespace theia

#endif  // THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_ROBUST_ROTATION_ESTIMATOR_H_
