// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>
#include <theia/theia.h>

#include "exp_bounding_box_utils.h"
#include "exp_camera_param_utils.h"
#include "exp_camera_param_io.h"


// Input/output files.
DEFINE_string(calibration_filepath, "", "");
DEFINE_string(input_bounding_box_filepath, "", "");
DEFINE_string(input_orientation_data_type, "",
              "'param', 'pose', 'modelview', or " "'reconstruction'");
DEFINE_string(input_orientation_filepath, "", "");
DEFINE_string(output_ply_filepath, "output.ply", "");


bool WritePlyFile(const std::string& ply_file,
                  const std::vector<Eigen::Vector3d>& points) {
  CHECK_GT(ply_file.length(), 0);

  // Return false if the file cannot be opened for writing.
  std::ofstream ply_writer(ply_file, std::ofstream::out);
  if (!ply_writer.is_open()) {
    LOG(ERROR) << "Could not open the file: " << ply_file
               << " for writing a PLY file.";
    return false;
  }

  ply_writer << "ply"
             << '\n' << "format ascii 1.0"
             << '\n' << "element vertex " << points.size()
             << '\n' << "property float x"
             << '\n' << "property float y"
             << '\n' << "property float z"
             << '\n' << "end_header" << std::endl;

  for (int i = 0; i < points.size(); i++) {
    ply_writer << points[i].transpose() << "\n";
  }

  return true;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  std::unordered_map<std::string, Eigen::Vector4d> bounding_boxes;
  ReadBoundingBoxes(FLAGS_input_bounding_box_filepath, &bounding_boxes);

  std::unordered_map<std::string, CameraIntrinsicsPrior>
      intrinsics_with_filenames;
  CHECK(theia::ReadCalibration(
      FLAGS_calibration_filepath, &intrinsics_with_filenames));

  // NOTE:
  // Remove extensions in view names.
  std::unordered_map<std::string, CameraIntrinsicsPrior> intrinsics;
  intrinsics.reserve(intrinsics_with_filenames.size());
  for (const auto& view : intrinsics_with_filenames) {
    const std::string basename = stlplus::basename_part(view.first);
    intrinsics[basename] = view.second;
  }

  std::unordered_map<std::string, Eigen::Matrix3d> orientations;
  CHECK(ReadOrientations(
      FLAGS_input_orientation_data_type,
      FLAGS_input_orientation_filepath, &orientations));

  std::unordered_map<std::string, Eigen::Vector3d> cam_coord_cam_to_objs;
  ComputeCameraToObjectDirections(
      bounding_boxes, intrinsics, &cam_coord_cam_to_objs);

  // Compute direction of camera from object.
  std::vector<Eigen::Vector3d> world_coord_obj_to_cams;
  world_coord_obj_to_cams.reserve(cam_coord_cam_to_objs.size());

  for (const auto& view : cam_coord_cam_to_objs) {
    const std::string view_name = view.first;
    const Eigen::Vector3d cam_coord_cam_to_obj = view.second;
    const Eigen::Matrix3d* orientation =
        theia::FindOrNull(orientations, view_name);
    if (orientation == nullptr) continue;

    const Eigen::Vector3d world_coord_obj_to_cam =
        (*orientation).inverse() * (-cam_coord_cam_to_obj);
    world_coord_obj_to_cams.push_back(world_coord_obj_to_cam);
  }
  CHECK(WritePlyFile(FLAGS_output_ply_filepath, world_coord_obj_to_cams));
}