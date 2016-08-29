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
DEFINE_string(input_reconstruction_filepath, "",
              "Use reconstruction file by default if it is given.");
DEFINE_string(calibration_filepath, "",
              "Use camera calibrations, bounding boxes and orientations if "
                  "reconstruction file is not given.");
DEFINE_string(input_bounding_boxes_filepath, "",
              "Use camera calibrations, bounding boxes and orientations if "
                  "reconstruction file is not given.");
DEFINE_string(input_orientations_data_type, "",
              "'param', 'pose', 'modelview', or " "'reconstruction'");
DEFINE_string(input_orientations_filepath, "",
              "Use camera calibrations, bounding boxes and orientations if "
                  "reconstruction file is not given.");
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

  std::vector<Eigen::Vector3d> world_coord_obj_to_cam_dirs;

  if (FLAGS_input_reconstruction_filepath != "") {
    // Use reconstruction file by default if it is given.
    Reconstruction reconstruction;
    CHECK(ReadReconstruction(
        FLAGS_input_reconstruction_filepath, &reconstruction))
    << "Could not read reconstruction file: '"
    << FLAGS_input_reconstruction_filepath << "'.";

    world_coord_obj_to_cam_dirs.clear();
    world_coord_obj_to_cam_dirs.reserve(reconstruction.NumViews());
    for (const auto& view_id : reconstruction.ViewIds()) {
      const View* view = reconstruction.View(view_id);
      CHECK(view != nullptr);
      const Camera& camera = view->Camera();
      world_coord_obj_to_cam_dirs.push_back(camera.GetPosition());
    }
  } else {
    // Use camera calibrations, bounding boxes and orientations if
    // reconstruction file is not given.
    std::unordered_map<std::string, CameraIntrinsicsPrior>
        intrinsics_with_filenames;
    CHECK(theia::ReadCalibration(
        FLAGS_calibration_filepath, &intrinsics_with_filenames))
    << "Could not read calibration file: '"
    << FLAGS_calibration_filepath << "'.";

    // NOTE:
    // Remove extensions in view names.
    std::unordered_map<std::string, CameraIntrinsicsPrior> intrinsics;
    intrinsics.reserve(intrinsics_with_filenames.size());
    for (const auto& view : intrinsics_with_filenames) {
      const std::string basename = stlplus::basename_part(view.first);
      intrinsics[basename] = view.second;
    }

    // Read orientations and bounding boxes.
    std::unordered_map<std::string, Eigen::Matrix3d> orientations;
    CHECK(ReadOrientations(
        FLAGS_input_orientations_data_type,
        FLAGS_input_orientations_filepath, &orientations))
    << "Could not read orientation file: '"
    << FLAGS_input_orientations_filepath << "'.";

    std::unordered_map<std::string, Eigen::Vector4d> bounding_boxes;
    ReadBoundingBoxes(FLAGS_input_bounding_boxes_filepath, &bounding_boxes);

    // Compute object to camera direction in world coordinates.
    world_coord_obj_to_cam_dirs.clear();
    world_coord_obj_to_cam_dirs.reserve(bounding_boxes.size());
    for (const auto& view : bounding_boxes) {
      const std::string view_name = view.first;
      const Eigen::Vector4d bounding_box = view.second;
      const theia::CameraIntrinsicsPrior* intrinsic =
          theia::FindOrNull(intrinsics, view_name);
      const Eigen::Matrix3d* orientation =
          theia::FindOrNull(orientations, view_name);
      if (intrinsic == nullptr || orientation == nullptr) continue;

      const Eigen::Vector3d cam_coord_cam_to_obj_dir =
          ComputeCameraToObjectDirections(bounding_box, *intrinsic);
      const Eigen::Vector3d world_coord_obj_to_cam_dir =
          (*orientation).inverse() * (-cam_coord_cam_to_obj_dir);
      world_coord_obj_to_cam_dirs.push_back(world_coord_obj_to_cam_dir);
    }
  }

  CHECK(WritePlyFile(FLAGS_output_ply_filepath, world_coord_obj_to_cam_dirs));
}
