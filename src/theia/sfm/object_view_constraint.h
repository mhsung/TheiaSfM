// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_OBJECT_VIEW_CONSTRAINT_H_
#define THEIA_SFM_OBJECT_VIEW_CONSTRAINT_H_

#include <Eigen/Core>
#include <unordered_map>

#include "theia/sfm/types.h"


namespace theia {

typedef uint32_t ObjectId;

// View orientations w.r.t object coordinates.
typedef std::unordered_map<ViewId, Eigen::Vector3d> ObjectViewOrientations;

// View position directions w.r.t object coordinates.
typedef std::unordered_map<ViewId, Eigen::Vector3d>
    ObjectViewPositionDirections;

}  // namespace theia


#endif  // THEIA_SFM_OBJECT_VIEW_CONSTRAINT_H_
