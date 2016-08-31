// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <theia/theia.h>

#include <string>
#include <unordered_map>


namespace theia {

// Bounding box format:
// xmin, ymin, xmax, ymax.

typedef std::unordered_map<std::string, Eigen::Vector4d> ViewBoundingBoxes;

bool IsPointIncludedInBoundingBox(
    const Eigen::Vector2d& point, const Eigen::Vector4d& bounding_box);

void ReadBoundingBoxes(
    const std::string& bounding_box_dir,
    std::unordered_map<std::string, Eigen::Vector4d>* bounding_boxes);

void ReadMultipleBoundingBoxes(
    const std::string& bounding_box_dir,
    const std::vector<std::string>& image_filenames,
    std::unordered_map<theia::ViewId, ViewBoundingBoxes>* view_bounding_boxes);

bool WriteBoundingBox(
    const std::string& bounding_box_file, const Eigen::Vector4d& bounding_box);

// Compute camera to object direction in camera coordinates.
Eigen::Vector3d ComputeCameraToObjectDirections(
    const Eigen::Vector4d& bounding_box,
    const theia::CameraIntrinsicsPrior& intrinsic);
}