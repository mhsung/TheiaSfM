// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_OBJECT_ROTATION_ERROR_H_
#define THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_OBJECT_ROTATION_ERROR_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include "theia/sfm/feature.h"
#include "theia/sfm/camera/camera.h"
#include "theia/sfm/camera/project_point_to_image.h"

namespace theia {

struct BundleObjectRotationError {
 public:
  explicit BundleObjectRotationError(
      const Eigen::Vector3d object_view_orientation,
      const double weight)
      : object_view_orientation_(object_view_orientation),
        weight_(weight) {
    CHECK_GT(weight_, 0);
  }

  template<typename T> bool operator()(const T* camera_extrinsics,
                      const T* object_orientation,
                      T* residuals) const {
    // const T* view_position = &(camera_extrinsics[
    //     Camera::ExternalParametersIndex::POSITION]);
    const T* view_orientation = &(camera_extrinsics[
        Camera::ExternalParametersIndex::ORIENTATION]);

    T object_view_orientation[3];
    object_view_orientation[0] = T(object_view_orientation_[0]);
    object_view_orientation[1] = T(object_view_orientation_[1]);
    object_view_orientation[2] = T(object_view_orientation_[2]);

    // Convert angle axis rotations to rotation matrices.
    Eigen::Matrix<T, 3, 3> object_orientation_mat, view_orientation_mat;
    ceres::AngleAxisToRotationMatrix(
        object_orientation, ceres::ColumnMajorAdapter3x3(
            object_orientation_mat.data()));
    ceres::AngleAxisToRotationMatrix(
        view_orientation, ceres::ColumnMajorAdapter3x3(
            view_orientation_mat.data()));

    // Compute the loop rotation from the two global rotations.
    const Eigen::Matrix<T, 3, 3> loop_rotation_mat =
        view_orientation_mat * object_orientation_mat.transpose();
    Eigen::Matrix<T, 3, 1> loop_rotation;
    ceres::RotationMatrixToAngleAxis(
        ceres::ColumnMajorAdapter3x3(loop_rotation_mat.data()),
        loop_rotation.data());
    residuals[0] = T(weight_) * (loop_rotation(0) - object_view_orientation[0]);
    residuals[1] = T(weight_) * (loop_rotation(1) - object_view_orientation[1]);
    residuals[2] = T(weight_) * (loop_rotation(2) - object_view_orientation[2]);

    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d view_object_position_direction,
      const double weight) {
    return new ceres::AutoDiffCostFunction<
        BundleObjectRotationError, 3, Camera::kExtrinsicsSize, 3>(
        new BundleObjectRotationError(
            view_object_position_direction, weight));
  }

private:
  const Eigen::Vector3d object_view_orientation_;
  const double weight_;
};

}  // namespace theia

#endif  // THEIA_SFM_BUNDLE_ADJUSTMENT_BUNDLE_OBJECT_ROTATION_ERROR_H_
