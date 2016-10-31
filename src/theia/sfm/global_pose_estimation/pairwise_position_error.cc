// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "theia/sfm/global_pose_estimation/pairwise_position_error.h"

#include <ceres/ceres.h>
#include <glog/logging.h>

namespace theia {

PairwisePositionError::PairwisePositionError(const double weight)
    : weight_(weight) {
  CHECK_GT(weight_, 0);
}

ceres::CostFunction* PairwisePositionError::Create(const double weight) {
  //return (new ceres::AutoDiffCostFunction<PairwisePositionError, 3, 3, 3, 3>(
  return (new ceres::AutoDiffCostFunction<PairwisePositionError, 3, 3, 3>(
        new PairwisePositionError(weight)));
}

}  // namespace theia
