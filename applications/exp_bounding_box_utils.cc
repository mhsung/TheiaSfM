// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_bounding_box_utils.h"

#include <fstream>
#include <iostream>
#include <glog/logging.h>
#include <stlplus3/file_system.hpp>


void ReadBoundingBoxes(
    const std::string& bounding_box_dir,
    std::unordered_map<std::string, Eigen::Vector4d>* bounding_boxes) {
  CHECK_NE(bounding_box_dir, "");
  CHECK_NOTNULL(bounding_boxes);
  bounding_boxes->clear();

  for(const auto& filename : stlplus::folder_files(bounding_box_dir)) {
    const std::string basename = stlplus::basename_part(filename);
    const std::string filepath = bounding_box_dir + "/" + filename;
    if (!theia::FileExists(filepath)) {
      LOG(WARNING) << "File does not exist: '" << filepath << "'";
      continue;
    }

    // Read camera parameters.
    std::ifstream file(filepath);
    Eigen::Vector4d bounding_box;
    for(int i = 0; i < 4; ++i) file >> bounding_box[i];
    file.close();

    (*bounding_boxes)[basename] = bounding_box;
    VLOG(3) << "Loaded '" << filepath << "'.";
  }
}

void ComputeCameraToObjectDirections(
    const std::unordered_map<std::string, Eigen::Vector4d>& bounding_boxes,
    const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>&
    intrinsics,
    std::unordered_map<std::string, Eigen::Vector3d>* cam_coord_cam_to_objs) {
  CHECK_NOTNULL(cam_coord_cam_to_objs);
  cam_coord_cam_to_objs->clear();
  cam_coord_cam_to_objs->reserve(bounding_boxes.size());

  for (const auto& view : bounding_boxes) {
    const std::string view_name = view.first;
    const Eigen::Vector4d bounding_box = view.second;
    const theia::CameraIntrinsicsPrior* intrinsic =
        theia::FindOrNull(intrinsics, view_name);
    if (intrinsic == nullptr) continue;

    // Compute object center on image.
    Eigen::Vector2d image_center;
    image_center << 0.5 * (bounding_box[2] - bounding_box[0]),
        0.5 * (bounding_box[3] - bounding_box[1]);

    // Compute ray passing through object center.
    static const bool kSetFocalLengthFromMedianFOV = false;
    theia::Camera camera;
    SetCameraIntrinsicsFromPriors(
        *intrinsic, kSetFocalLengthFromMedianFOV, &camera);
    const Eigen::Vector3d cam_coord_cam_to_obj =
        camera.PixelToNormalizedCoordinates(image_center).normalized();
    (*cam_coord_cam_to_objs)[view_name] = cam_coord_cam_to_obj;
  }
}
