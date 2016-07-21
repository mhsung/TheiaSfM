#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>
#include <fstream>
#include <iostream>
#include <stlplus3/file_system.hpp>
#include <string>
#include <vector>

#include "exp_camera_param_utils.h"
#include "exp_camera_param_io.h"


// Input/output files.
DEFINE_string(images, "", "Wildcard of images to reconstruct.");
DEFINE_string(reference_reconstruction_file, "",
              "Reference reconstruction file in binary format.");
DEFINE_string(target_input_type, "", "");
DEFINE_string(target_filepath, "", "");
DEFINE_string(output_relative_target_camera_param_dir, "", "");


void ReadImageNames(
    const std::string& image_wildcard_filepath,
    std::vector<std::string>* image_names) {
  CHECK_NOTNULL(image_names);

  std::vector<std::string> image_files;
  CHECK(theia::GetFilepathsFromWildcard(
      image_wildcard_filepath, &image_files))
  << "Could not find images that matched the filepath: "
  << image_wildcard_filepath
  << ". NOTE that the ~ filepath is not supported.";
  CHECK(!image_files.empty());

  image_names->clear();
  image_names->reserve(image_files.size());

  for (const std::string& image_file : image_files) {
    std::string image_filename;
    CHECK(theia::GetFilenameFromFilepath(image_file, true, &image_filename));
    image_names->push_back(image_filename);
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  // Load images.
  std::vector<std::string> image_names;
  ReadImageNames(FLAGS_images, &image_names);

  // Load the reference ref_reconstruction.
  theia::Reconstruction ref_reconstruction;
  CHECK(theia::ReadReconstruction(
      FLAGS_reference_reconstruction_file, &ref_reconstruction))
      << "Could not read reconstruction file: '"
      << FLAGS_reference_reconstruction_file << "'.";

  std::unordered_map<theia::ViewId, Eigen::Matrix3d> ref_rotations;
  GetRotationsFromReconstruction(ref_reconstruction, &ref_rotations);


  // Load target data.
  std::unordered_map<theia::ViewId, Eigen::Matrix3d> est_rotations;

  if (FLAGS_target_input_type == "param") {
    ReadRotationsFromCameraParams(
        FLAGS_target_filepath, image_names, ref_reconstruction, &est_rotations);
  }
  else if (FLAGS_target_input_type == "modelview") {
    ReadRotationsFromModelviews(
        FLAGS_target_filepath, image_names, ref_reconstruction, &est_rotations);
  }
  else if (FLAGS_target_input_type == "reconstruction") {
    ReadRotationsFromReconstruction(
        FLAGS_target_filepath, image_names, ref_reconstruction, &est_rotations);
  }
  else {
    CHECK(false) << "'target_input_type' must be either "
        << "'param', 'modelview', or 'reconstruction'.";
  }


//  // Sync target (estimated) rotations with reference.
//  std::unordered_map<theia::ViewId, Eigen::Matrix3d> rel_est_rotations;
//  SyncRotationLists(ref_rotations, est_rotations, &rel_est_rotations);

  std::unordered_map<theia::ViewId, Eigen::Matrix3d> rel_ref_rotations;
  ComputeRelativeRotationFromFirstFrame(ref_rotations, &rel_ref_rotations);
  std::unordered_map<theia::ViewId, Eigen::Matrix3d> rel_est_rotations;
  ComputeRelativeRotationFromFirstFrame(est_rotations, &rel_est_rotations);


  // Save relative target camera parameters.
  if (FLAGS_output_relative_target_camera_param_dir != "") {
    if (!stlplus::folder_exists(
        FLAGS_output_relative_target_camera_param_dir)) {
      CHECK(stlplus::folder_create(
          FLAGS_output_relative_target_camera_param_dir));
    }
    CHECK(stlplus::folder_writable(
        FLAGS_output_relative_target_camera_param_dir));

    for (const auto& rel_est_R_pair : rel_est_rotations) {
      const theia::ViewId view_id = rel_est_R_pair.first;
      const Eigen::Matrix3d rel_est_R = rel_est_R_pair.second;
      const Eigen::Vector3d rel_est_camera_params
          = ComputeCameraParamsFromModelview(rel_est_R);

      const theia::View* view = ref_reconstruction.View(view_id);
      const std::string basename = stlplus::basename_part(view->Name());
      const std::string output_filepath =
          FLAGS_output_relative_target_camera_param_dir +
          "/" + basename + ".txt";
      std::cout << "Saving '" << output_filepath << "'... ";

      std::ofstream output_file(output_filepath);
      output_file
          << rel_est_camera_params[0] << " "
          << rel_est_camera_params[1] << " "
          << rel_est_camera_params[2] << std::endl;
      output_file.close();

      std::cout << "Done." << std::endl;
    }
  }
}
