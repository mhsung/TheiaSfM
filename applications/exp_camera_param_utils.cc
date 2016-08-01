// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_camera_param_utils.h"

#include <glog/logging.h>

#include "unsupported/Eigen/MatrixFunctions"


Eigen::Affine3d ComputeModelviewFromTheiaCamera(
    const theia::Camera& camera) {
  Eigen::Affine3d theia_modelview = Eigen::Affine3d::Identity();
  theia_modelview.pretranslate(-camera.GetPosition());
  theia_modelview.prerotate(camera.GetOrientationAsRotationMatrix());

  // (X, Y, Z) -> (X, -Y, -Z).
  Eigen::Matrix3d axes_converter = Eigen::Matrix3d::Identity();
  axes_converter(0, 0) = 1.0;
  axes_converter(1, 1) = -1.0;
  axes_converter(2, 2) = -1.0;

  const Eigen::Affine3d modelview = axes_converter * theia_modelview;
  return modelview;
}

Eigen::Matrix3d ComputeModelviewRotationFromTheiaCamera(
    const Eigen::Matrix3d& theia_camera_rotation) {
  // (X, Y, Z) -> (X, -Y, -Z).
  Eigen::Matrix3d axes_converter = Eigen::Matrix3d::Identity();
  axes_converter(0, 0) = 1.0;
  axes_converter(1, 1) = -1.0;
  axes_converter(2, 2) = -1.0;

  const Eigen::Matrix3d modelview_rotation =
      axes_converter * theia_camera_rotation;
  return modelview_rotation;
}

Eigen::Matrix3d ComputeTheiaCameraRotationFromModelview(
    const Eigen::Matrix3d& modelview_rotation) {
  // (X, Y, Z) -> (X, -Y, -Z).
  Eigen::Matrix3d inverse_axes_converter = Eigen::Matrix3d::Identity();
  inverse_axes_converter(0, 0) = 1.0;
  inverse_axes_converter(1, 1) = -1.0;
  inverse_axes_converter(2, 2) = -1.0;

  const Eigen::Matrix3d theia_camera_rotation =
      inverse_axes_converter * modelview_rotation;
  return theia_camera_rotation;
}

Eigen::Vector3d ComputeCameraParamsFromModelviewRotation(
    const Eigen::Matrix3d& modelview_rotation) {

  // Default rotation R_d.
  // R = R_z R_x R_y R_d.
  const Eigen::AngleAxisd default_rotation(
      -0.5 * M_PI, Eigen::Vector3d::UnitY());

  // R R_d^T = R_z R_x R_y.
  const Eigen::Matrix3d rotation =
      modelview_rotation * default_rotation.toRotationMatrix().transpose();


  // ZXY Euler angles.
  Eigen::Vector3d angles = rotation.eulerAngles(2, 0, 1);

  Eigen::Vector3d camera_params;
  // Y-axis (Azimuth, [R_y | 0])
  // Note: Change the sign.
  camera_params[0] = -angles[2] / M_PI * 180.0;

  // X-axis (Elevation, [R_x | 0])
  camera_params[1] = +angles[1] / M_PI * 180.0;

  // Z-axis (Theta, [R_z | 0])
  // Note: Change the sign.
  camera_params[2] = -angles[0] / M_PI * 180.0;


  // Elevation must be in range [-90, 90].
  while(camera_params[1] < -90.0) camera_params[1] += 360.0;

  if (camera_params[1] > 270.0) {
    camera_params[1] = 360.0 - camera_params[1];
  } else if (camera_params[1] > 90.0) {
    camera_params[1] = 180.0 - camera_params[1];
    camera_params[0] += 180.0;
    camera_params[2] += 180.0;
  }

  // Azimuth and theta must be in range [0, 360).
  while(camera_params[0] >= 360.0) camera_params[0] -= 360.0;
  while(camera_params[0] < 0.0) camera_params[0] += 360.0;

  while(camera_params[2] >= 360.0) camera_params[2] -= 360.0;
  while(camera_params[2] < 0.0) camera_params[2] += 360.0;

  return camera_params;
}

Eigen::Affine3d ComputeModelviewFromCameraParams(
    const Eigen::Vector3d& camera_params) {
  const double kTranslationDistance = 1.5;

  const double azimuth_deg = camera_params[0];
  const double elevation_deg = camera_params[1];
  const double theta_deg = camera_params[2];

  Eigen::Affine3d modelview(Eigen::Affine3d::Identity());

  // Default rotation R_d.
  // R = R_z R_x R_y R_d.
  const Eigen::AngleAxisd default_rotation(
      -0.5 * M_PI, Eigen::Vector3d::UnitY());

  modelview.prerotate(default_rotation);

  // Y-axis (Azimuth, [R_y | 0])
  // Note: Change the sign.
  const double azimuth = (double) -azimuth_deg / 180.0 * M_PI;
  modelview.prerotate(Eigen::AngleAxisd(azimuth, Eigen::Vector3d::UnitY()));

  // X-axis (Elevation, [R_x | 0])
  const double elevation = (double) elevation_deg / 180.0 * M_PI;
  modelview.prerotate(
      Eigen::AngleAxisd(elevation, Eigen::Vector3d::UnitX()));

  // Translation ([I | t])
  modelview.pretranslate(Eigen::Vector3d(0, 0, -kTranslationDistance));

  // Z-axis (Theta, [R_z | 0])
  // Note: Change the sign.
  const double theta = (double) -theta_deg / 180.0 * M_PI;
  modelview.prerotate(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));

  // T = [R_z | 0] [I | t] [R_x | 0] [R_y | 0] [R_d | 0].
  return modelview;
}

Eigen::Vector3d ComputeCameraParamsFromTheiaCamera(
    const theia::Camera& camera) {
  return ComputeCameraParamsFromModelviewRotation(
      ComputeModelviewFromTheiaCamera(camera).rotation());
}

Eigen::Vector3d ComputeCameraParamsFromTheiaCameraRotation(
    const Eigen::Matrix3d& theia_camera_rotation) {
  return ComputeCameraParamsFromModelviewRotation(
      ComputeModelviewRotationFromTheiaCamera(theia_camera_rotation));
}

Eigen::Matrix3d ComputeTheiaCameraRotationFromCameraParams(
    const Eigen::Vector3d& camera_params) {
  return ComputeTheiaCameraRotationFromModelview(
      ComputeModelviewFromCameraParams(camera_params).rotation());
}

Eigen::Matrix3d ComputeAverageRotation(
    const std::vector<Eigen::Matrix3d>& R_list) {
  const double kErrorThreshold = 1.0E-8;
  const int kMaxIter = 30;

  const int num_Rs = R_list.size();
  CHECK_GT(num_Rs, 0);

  Eigen::Matrix3d avg_R = Eigen::Matrix3d::Identity();

  // Minimize under the geodesic metric.
  // Hartley et al., L1 rotation averaging using the Weiszfeld algorithm,
  // CVPR 2011.
  // Algorithm 1.
  for (int iter = 0; iter < kMaxIter; ++iter) {
    Eigen::Matrix3d r = Eigen::Matrix3d::Zero();
    for (const Eigen::Matrix3d diff_R : R_list) {
      r += (avg_R.transpose() * diff_R).log();
    }
    r = r / (double) num_Rs;
    avg_R = avg_R * r.exp();

    const double error = r.norm();
    VLOG(3) << "[" << iter << "] error = " << error;
    if (error < kErrorThreshold) break;
  }

  VLOG(3) << "Done.";
  return avg_R;
}
