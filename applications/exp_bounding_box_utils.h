// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <theia/theia.h>
#include <string>
#include <unordered_map>


void ReadBoundingBoxes(
    const std::string& bounding_box_dir,
    std::unordered_map<std::string, Eigen::Vector4d>* bounding_boxes);

// Compute camera to object direction in camera coordinates.
Eigen::Vector3d ComputeCameraToObjectDirections(
    const Eigen::Vector4d& bounding_box,
    const theia::CameraIntrinsicsPrior& intrinsic);
