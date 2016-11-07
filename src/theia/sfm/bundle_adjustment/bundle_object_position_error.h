// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_OBJECT_POSITION_ERROR_H_
#define THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_OBJECT_POSITION_ERROR_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include "theia/sfm/feature.h"
#include "theia/sfm/camera/camera.h"
#include "theia/sfm/camera/project_point_to_image.h"

namespace theia {

struct BundleObjectPositionError {
 public:
  explicit BundleObjectPositionError(
      const Eigen::Vector3d view_object_position_direction,
      const double weight)
      : view_object_position_direction_(view_object_position_direction),
        weight_(weight) {
    CHECK_GT(weight_, 0);
  }

  template<typename T> bool operator()(const T* camera_extrinsics,
                      const T* object_position,
                      T* residuals) const {
    const T* view_position = &(camera_extrinsics[
        Camera::ExternalParametersIndex::POSITION]);
    const T* view_orientation = &(camera_extrinsics[
        Camera::ExternalParametersIndex::ORIENTATION]);

    Eigen::Matrix<T, 3, 3> view_rotation_mat;
    ceres::AngleAxisToRotationMatrix(
        view_orientation,
        ceres::ColumnMajorAdapter3x3(view_rotation_mat.data()));

    const Eigen::Matrix<T, 3, 1> translation_direction =
        view_rotation_mat.transpose() *
            view_object_position_direction_.cast<T>();

    const T kNormTolerance = T(1e-12);

    T translation[3];
    translation[0] = object_position[0] - view_position[0];
    translation[1] = object_position[1] - view_position[1];
    translation[2] = object_position[2] - view_position[2];
    T norm =
        sqrt(translation[0] * translation[0] + translation[1] * translation[1] +
             translation[2] * translation[2]);

    // If the norm is very small then the positions are very close together. In
    // this case, avoid dividing by a tiny number which will cause the weight of
    // the residual term to potentially skyrocket.
    if (T(norm) < kNormTolerance) {
      norm = T(1.0);
    }

    residuals[0] =
        T(weight_) * (translation[0] / norm - translation_direction[0]);
    residuals[1] =
        T(weight_) * (translation[1] / norm - translation_direction[1]);
    residuals[2] =
        T(weight_) * (translation[2] / norm - translation_direction[2]);
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d view_object_position_direction,
      const double weight) {
    return new ceres::AutoDiffCostFunction<
        BundleObjectPositionError, 3, Camera::kExtrinsicsSize, 3>(
        new BundleObjectPositionError(
            view_object_position_direction, weight));
  }

private:
  const Eigen::Vector3d view_object_position_direction_;
  const double weight_;
};

}  // namespace theia

#endif  // THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_OBJECT_POSITION_ERROR_H_
