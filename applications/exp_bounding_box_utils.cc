// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_bounding_box_utils.h"

#include <fstream>
#include <iostream>
#include <glog/logging.h>
#include <stlplus3/file_system.hpp>


namespace theia {

bool IsPointIncludedInBoundingBox(
    const Eigen::Vector2d& point, const Eigen::Vector4d& bounding_box) {
  return (point[0] >= bounding_box[0] && point[0] <= bounding_box[2] &&
          point[1] >= bounding_box[1] && point[1] <= bounding_box[3]);
}

void ReadBoundingBoxes(
    const std::string& bounding_box_dir,
    std::unordered_map<std::string, Eigen::Vector4d>* bounding_boxes) {
  CHECK_NE(bounding_box_dir, "");
  CHECK_NOTNULL(bounding_boxes)->clear();

  for(const auto& filename : stlplus::folder_files(bounding_box_dir)) {
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

    const std::string basename = stlplus::basename_part(filename);
    (*bounding_boxes)[basename] = bounding_box;
    VLOG(3) << "Loaded '" << filepath << "'.";
  }
}

void ReadMultipleBoundingBoxes(
    const std::string& bounding_box_dir,
    const std::vector<std::string>& image_filenames,
    std::unordered_map<theia::ViewId, ViewBoundingBoxes>* view_bounding_boxes) {
  CHECK_NE(bounding_box_dir, "");
  CHECK_NOTNULL(view_bounding_boxes)->clear();

  for (theia::ViewId view_id = 0; view_id < image_filenames.size(); ++view_id) {
    const std::string image_basename =
        stlplus::basename_part(image_filenames[view_id]);
    const std::string view_dir = bounding_box_dir + "/" + image_basename;

    (*view_bounding_boxes)[view_id].clear();
    for(const auto& filename : stlplus::folder_files(view_dir)) {
      const std::string filepath = view_dir + "/" + filename;
      if (!theia::FileExists(filepath)) {
        LOG(WARNING) << "File does not exist: '" << filepath << "'";
        continue;
      }

      // NOTE:
      // Assume that bounding file has '_bbox' postfix.
      const std::string basename = stlplus::basename_part(filename);
      const std::string postfix = "_bbox";
      const size_t keylen = postfix.length();
      const size_t strlen = basename.length();
      if(strlen < keylen ||
          basename.rfind(postfix.c_str(), strlen - keylen, keylen) ==
              std::string::npos) {
        continue;
      }

      // Read camera parameters.
      std::ifstream file(filepath);
      Eigen::Vector4d bounding_box;
      for(int i = 0; i < 4; ++i) file >> bounding_box[i];
      file.close();

      (*view_bounding_boxes)[view_id][basename] = bounding_box;
      VLOG(3) << "Loaded '" << filepath << "'.";
    }
  }
}

bool WriteBoundingBox(
    const std::string& bounding_box_file, const Eigen::Vector4d& bounding_box) {
  std::ofstream file(bounding_box_file);
  if (!file.good()) return false;
  for(int i = 0; i < 4; ++i) file << bounding_box[i] << " ";
  file.close();
  VLOG(3) << "Saved '" << bounding_box_file << "'.";
  return true;
}

Eigen::Vector3d ComputeCameraToObjectDirections(
    const Eigen::Vector4d& bounding_box,
    const theia::CameraIntrinsicsPrior& intrinsic) {
  // Compute object center on image.
  const Eigen::Vector2d bbox_center(
    0.5 * (bounding_box[0] + bounding_box[2]),
    0.5 * (bounding_box[1] + bounding_box[3]));

  // Compute ray passing through object center.
  static const bool kSetFocalLengthFromMedianFOV = false;
  theia::Camera camera;
  SetCameraIntrinsicsFromPriors(
      intrinsic, kSetFocalLengthFromMedianFOV, &camera);
  const Eigen::Vector3d cam_coord_cam_to_obj_dir =
      camera.PixelToNormalizedCoordinates(bbox_center).normalized();
  return cam_coord_cam_to_obj_dir;
}
}