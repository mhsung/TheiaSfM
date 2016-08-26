// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <theia/theia.h>
#include <string>
#include <unordered_map>


void ReadBoundingBoxes(
    const std::string& bounding_box_dir,
    std::unordered_map<std::string, Eigen::Vector4d>* bounding_boxes);

void ComputeCameraToObjectDirections(
    const std::unordered_map<std::string, Eigen::Vector4d>& bounding_boxes,
    const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>&
    intrinsics,
    std::unordered_map<std::string, Eigen::Vector3d>* cam_coord_cam_to_objs);
