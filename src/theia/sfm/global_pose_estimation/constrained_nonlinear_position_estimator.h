// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'nonlinear_position_estimator.h'

#ifndef THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_NONLINEAR_POSITION_ESTIMATOR_H_
#define THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_NONLINEAR_POSITION_ESTIMATOR_H_

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <memory>
#include <unordered_map>
#include <vector>

#include "theia/util/util.h"
#include "theia/sfm/global_pose_estimation/nonlinear_position_estimator.h"
#include "theia/sfm/reconstruction.h"
#include "theia/sfm/types.h"
#include "theia/sfm/view_triplet.h"
// @mhsung
#include "theia/sfm/object_view_constraint.h"

namespace theia {

class ConstrainedNonlinearPositionEstimator
    : public NonlinearPositionEstimator {
 public:
  ConstrainedNonlinearPositionEstimator(
      const NonlinearPositionEstimator::Options& options,
      const Reconstruction& reconstruction,
      const double constraint_weight);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  bool EstimatePositions(
      const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
      const std::unordered_map<ViewId, Eigen::Vector3d>& view_orientation,
      const std::unordered_map<ObjectId, ObjectViewPositionDirections>&
          object_view_constraints,
      std::unordered_map<ViewId, Eigen::Vector3d>* view_positions,
      std::unordered_map<ViewId, Eigen::Vector3d>* object_positions);

 private:
  // Initialize all cameras to be random.
  void InitializeRandomPositions(
      const std::unordered_map<ViewId, Eigen::Vector3d>& view_orientations,
      std::unordered_map<ViewId, Eigen::Vector3d>* view_positions,
      std::unordered_map<ViewId, Eigen::Vector3d>* object_positions);

  // Creates camera to camera constraints from relative translations.
  void AddObjectToCameraConstraints(
      const std::unordered_map<ViewId, Eigen::Vector3d>& view_orientations,
      std::unordered_map<ViewId, Eigen::Vector3d>* view_positions,
      std::unordered_map<ViewId, Eigen::Vector3d>* object_positions);

  void AddCamerasAndPointsToParameterGroups(
      std::unordered_map<ViewId, Eigen::Vector3d>* view_positions,
      std::unordered_map<ViewId, Eigen::Vector3d>* object_positions);

  friend class EstimatePositionsNonlinearTest;

  // FIXME:
  // Remove constant weight and use weight vector.
  const double constraint_weight_;

  const std::unordered_map<ObjectId, ObjectViewPositionDirections>*
      object_view_constraints_;

  DISALLOW_COPY_AND_ASSIGN(ConstrainedNonlinearPositionEstimator);
};

}  // namespace theia

#endif  // THEIA_SFM_GLOBAL_POSE_ESTIMATION_CONSTRAINED_NONLINEAR_POSITION_ESTIMATOR_H_
