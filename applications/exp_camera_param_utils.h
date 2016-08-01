// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <theia/theia.h>
#include <Eigen/Geometry>
#include <vector>


// 'Modelview': OpenGL modelview matrix.
// 'CameraParams': Azimuth, elevation, and theta.
//    Used in RenderForCNN (ICCV 2015')


// Modelview <-> Theia camera.
Eigen::Affine3d ComputeModelviewFromTheiaCamera(
    const theia::Camera& camera);
Eigen::Matrix3d ComputeModelviewRotationFromTheiaCamera(
    const Eigen::Matrix3d& theia_camera_rotation);
Eigen::Matrix3d ComputeTheiaCameraRotationFromModelview(
    const Eigen::Matrix3d& modelview_rotation);

// Camera parameter <-> Modelview.
Eigen::Vector3d ComputeCameraParamsFromModelviewRotation(
    const Eigen::Matrix3d& modelview_rotation);
Eigen::Affine3d ComputeModelviewFromCameraParams(
    const Eigen::Vector3d& camera_params);

// Camera parameter <-> Theia camera.
Eigen::Vector3d ComputeCameraParamsFromTheiaCamera(
    const theia::Camera& camera);
Eigen::Vector3d ComputeCameraParamsFromTheiaCameraRotation(
    const Eigen::Matrix3d& theia_camera_rotation);
Eigen::Matrix3d ComputeTheiaCameraRotationFromCameraParams(
    const Eigen::Vector3d& camera_params);

// Single rotation averaging.
// Compute avg_R that minimizes \sum_i \log (avg_R^T R_i).
Eigen::Matrix3d ComputeAverageRotation(
    const std::vector<Eigen::Matrix3d>& R_list);
