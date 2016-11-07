// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_CONSECUTIVE_CAMERA_ERROR_H_
#define THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_CONSECUTIVE_CAMERA_ERROR_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include "theia/sfm/feature.h"
#include "theia/sfm/camera/camera.h"
#include "theia/sfm/camera/project_point_to_image.h"

namespace theia {

struct BundleConsecutiveCameraError {
 public:
  explicit BundleConsecutiveCameraError(const double weight)
      : weight_(weight) {
    CHECK_GT(weight_, 0);
  }

  template<typename T> bool operator()(const T* prev_camera_extrinsics,
                      const T* curr_camera_extrinsics,
                      const T* next_camera_extrinsics,
                      T* residuals) const {
    const T* prev_position = &(prev_camera_extrinsics[
        Camera::ExternalParametersIndex::POSITION]);
    const T* curr_position = &(curr_camera_extrinsics[
        Camera::ExternalParametersIndex::POSITION]);
    const T* next_position = &(next_camera_extrinsics[
        Camera::ExternalParametersIndex::POSITION]);

    // Compute first derivative error.
    // residuals[0] = T(weight_) * (next_position[0] - curr_position[0]);
    // residuals[1] = T(weight_) * (next_position[1] - curr_position[1]);
    // residuals[2] = T(weight_) * (next_position[2] - curr_position[2]);

    // Compute second derivative error.
    residuals[0] = T(weight_) * ((next_position[0] - curr_position[0]) -
                                 (curr_position[0] - prev_position[0]));
    residuals[1] = T(weight_) * ((next_position[1] - curr_position[1]) -
                                 (curr_position[1] - prev_position[1]));
    residuals[2] = T(weight_) * ((next_position[2] - curr_position[2]) -
                                 (curr_position[2] - prev_position[2]));

    return true;
  }

  static ceres::CostFunction* Create(const double weight) {
    return new ceres::AutoDiffCostFunction<
        BundleConsecutiveCameraError, Camera::kExtrinsicsSize,
        Camera::kExtrinsicsSize, Camera::kExtrinsicsSize, 3>(
        new BundleConsecutiveCameraError(weight));
  }

private:
  const double weight_;
};

}  // namespace theia

#endif  // THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_CONSECUTIVE_CAMERA_ERROR_H_
