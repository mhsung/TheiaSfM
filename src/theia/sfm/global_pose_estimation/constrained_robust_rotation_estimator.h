// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_ROBUST_ROTATION_ESTIMATOR_H_
#define THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_ROBUST_ROTATION_ESTIMATOR_H_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>

#include "theia/sfm/global_pose_estimation/robust_rotation_estimator.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/types.h"
#include "theia/util/hash.h"

namespace theia {

class ConstrainedRobustRotationEstimator : public RobustRotationEstimator {
 public:
  explicit ConstrainedRobustRotationEstimator(
      const Options& options, const double constraint_weight)
    : RobustRotationEstimator(options),
      constraint_weight_(constraint_weight) {}

  // Estimates the global orientations of all views based on an initial
  // guess. Returns true on successful estimation and false otherwise.
  bool EstimateRotations(
      const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
      const std::unordered_map<ViewId, Eigen::Vector3d>& constrained_views,
      std::unordered_map<ViewId, Eigen::Vector3d>* global_orientations);

 protected:
  // Sets up the sparse linear system such that dR_ij = dR_j - dR_i. This is the
  // first-order approximation of the angle-axis rotations. This should only be
  // called once.
  virtual void SetupLinearSystem();

  // Computes the relative rotation error based on the current global
  // orientation estimates.
  virtual void ComputeRotationError();

  const double constraint_weight_;

  const std::unordered_map<ViewId, Eigen::Vector3d>* constrained_views_;
};

}  // namespace theia

#endif  // THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_ROBUST_ROTATION_ESTIMATOR_H_