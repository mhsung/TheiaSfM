// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_OBJECT_VIEW_CONSTRAINT_H_
#define THEIA_SFM_OBJECT_VIEW_CONSTRAINT_H_

#include <Eigen/Core>
#include <unordered_map>

#include "theia/sfm/types.h"


namespace theia {

typedef uint32_t ObjectId;

// View orientations w.r.t object orientations.
typedef std::unordered_map<ViewId, Eigen::Vector3d> ObjectViewOrientations;
typedef std::unordered_map<ViewId, double> ObjectViewOrientationWeights;

// Object positions w.r.t view positions.
typedef std::unordered_map<ViewId, Eigen::Vector3d>
    ViewObjectPositionDirections;
typedef std::unordered_map<ViewId, double>
    ViewObjectPositionDirectionWeights;

}  // namespace theia


#endif  // THEIA_SFM_OBJECT_VIEW_CONSTRAINT_H_
