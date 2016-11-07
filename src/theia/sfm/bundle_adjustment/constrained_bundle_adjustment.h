// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_BUNDLE_ADJUSTMENT_CONSTRAINED_BUNDLE_ADJUSTMENT_H_
#define THEIA_SFM_BUNDLE_ADJUSTMENT_CONSTRAINED_BUNDLE_ADJUSTMENT_H_

#include <ceres/ceres.h>
#include <unordered_set>

#include "theia/sfm/bundle_adjustment/bundle_adjustment.h"
#include "theia/sfm/bundle_adjustment/create_loss_function.h"
#include "theia/sfm/types.h"
#include "theia/util/enable_enum_bitmask_operators.h"
// @mhsung
#include "theia/sfm/object_view_constraint.h"


namespace theia {

class Reconstruction;

struct BundleObjectConstraints {
  BundleObjectConstraints(
      const std::unordered_map<ObjectId, ObjectViewOrientations>&
      object_view_orientations,
      const std::unordered_map<ObjectId, ViewObjectPositionDirections>&
      view_object_position_directions,
      std::unordered_map<ViewId, Eigen::Vector3d>* object_orientations,
      std::unordered_map<ViewId, Eigen::Vector3d>* object_positions,
      const double orientation_weight,
      const double position_weight)
  : object_view_orientations_(object_view_orientations),
    view_object_position_directions_(view_object_position_directions),
    object_orientations_(object_orientations),
    object_positions_(object_positions),
    orientation_weight_(orientation_weight),
    position_weight_(position_weight) {
    CHECK_NOTNULL(object_orientations_);
    CHECK_NOTNULL(object_positions_);
  }

  const std::unordered_map<ObjectId, ObjectViewOrientations>&
      object_view_orientations_;
  const std::unordered_map<ObjectId, ViewObjectPositionDirections>&
      view_object_position_directions_;
  std::unordered_map<ViewId, Eigen::Vector3d>* object_orientations_;
  std::unordered_map<ViewId, Eigen::Vector3d>* object_positions_;

  // FIXME:
  // Allow to use per object-view pair weight.
  const double orientation_weight_;
  const double position_weight_;
};

// Bundle adjust all views and tracks in the reconstruction.
BundleAdjustmentSummary ConstrainedBundleAdjustReconstruction(
    const BundleAdjustmentOptions& options,
    Reconstruction* reconstruction,
    BundleObjectConstraints* object_constraints);

// Bundle adjust the specified views and all tracks observed by those views.
BundleAdjustmentSummary ConstrainedBundleAdjustPartialReconstruction(
    const BundleAdjustmentOptions& options,
    const std::unordered_set<ViewId>& views_to_optimize,
    const std::unordered_set<TrackId>& tracks_to_optimize,
    Reconstruction* reconstruction,
    BundleObjectConstraints* object_constraints);

}  // namespace theia

#endif  // THEIA_SFM_BUNDLE_ADJUSTMENT_CONSTRAINED_BUNDLE_ADJUSTMENT_H_
