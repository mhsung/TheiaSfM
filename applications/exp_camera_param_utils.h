#include <Eigen/Core>
#include <theia/theia.h>
#include <Eigen/Geometry>
#include <vector>


// NOTE:
// Camera orientation is modelview rotation.


// Theia camera -> OpenGL modelview.
Eigen::Affine3d ComputeModelviewFromTheiaCamera(
    const theia::Camera& camera);

// Camera parameter <-> Modelview.
Eigen::Vector3d ComputeCameraParamsFromModelview(
    const Eigen::Matrix3d& modelview_rotation);
Eigen::Affine3d ComputeModelviewFromCameraParams(
    const Eigen::Vector3d& camera_params);

// Single rotation averaging.
// Compute avg_R that minimizes \sum_i \log (avg_R^T R_i).
Eigen::Matrix3d ComputeAverageRotation(
    const std::vector<Eigen::Matrix3d>& R_list);
