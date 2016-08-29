// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'pairwise_translation_error.cc'

#include "theia/sfm/global_pose_estimation/single_translation_error.h"

#include <ceres/ceres.h>
#include <glog/logging.h>

namespace theia {

SingleTranslationError::SingleTranslationError(
    const Eigen::Vector3d& translation_direction, const double weight)
    : translation_direction_(translation_direction), weight_(weight) {
  CHECK_GT(weight_, 0);
}

ceres::CostFunction* SingleTranslationError::Create(
    const Eigen::Vector3d& translation_direction, const double weight) {
  return (new ceres::AutoDiffCostFunction<SingleTranslationError, 3, 3>(
      new SingleTranslationError(translation_direction, weight)));
}

}  // namespace theia
