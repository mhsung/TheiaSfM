#include "exp_camera_param_io.h"

#include <fstream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <stlplus3/file_system.hpp>

#include "exp_camera_param_utils.h"
#include "unsupported/Eigen/MatrixFunctions"


void ReadRotationsFromCameraParams(
    const std::string& camera_param_dir,
    const std::vector<std::string>& image_names,
    const theia::Reconstruction& ref_reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations) {
  CHECK_NE(camera_param_dir, "");
  CHECK_NOTNULL(modelview_rotations);

  for (const std::string& image_name : image_names) {
    const std::string basename = stlplus::basename_part(image_name);

    std::string camera_file = camera_param_dir + "/" + basename + ".txt";
    if (!theia::FileExists(camera_file)) {
      LOG(WARNING) << "File does not exist: '" << basename << "'";
      continue;
    }

    // Read camera pose parameters.
    std::ifstream file(camera_file);
    Eigen::Vector3i camera_params;
    for(int i = 0; i < 3; ++i) file >> camera_params[i];
    file.close();

    const Eigen::Affine3d modelview =
        ComputeModelviewFromCameraParams(camera_params.cast<double>());

    // DEBUG.
    /*
    const Eigen::Vector3d test_camera_param =
        ComputeCameraParamsFromModelview(modelview.rotation());
    for (int i = 0; i < 3; i++) {
      CHECK_EQ(camera_params[i], static_cast<int>(std::round
          (test_camera_param[i])) % 360);
    }
    */

    const theia::ViewId view_id =
        ref_reconstruction.ViewIdFromName(image_name);
    if (view_id != theia::kInvalidViewId) {
      modelview_rotations->emplace(view_id, modelview.rotation());
    }
  }
}

void ReadRotationsFromModelviews(
    const std::string& modelview_dir,
    const std::vector<std::string>& image_names,
    const theia::Reconstruction& ref_reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations) {
  CHECK_NE(modelview_dir, "");
  CHECK_NOTNULL(modelview_rotations);

  for (const std::string& image_name : image_names) {
    const std::string basename = stlplus::basename_part(image_name);

    std::string modelview_file = modelview_dir + "/" + basename + ".txt";
    if (!theia::FileExists(modelview_file)) {
      LOG(WARNING) << "File does not exist: '" << basename << "'";
      continue;
    }

    // Read camera pose parameters.
    std::ifstream file(modelview_file);
    double values[16];
    for(int i = 0; i < 16; ++i) file >> values[i];
    file.close();

    const Eigen::Affine3d modelview =
        Eigen::Affine3d(Eigen::Matrix4d::Map(values));

    const theia::ViewId view_id =
        ref_reconstruction.ViewIdFromName(image_name);
    if (view_id != theia::kInvalidViewId) {
      modelview_rotations->emplace(view_id, modelview.rotation());
    }
  }
}

void ReadRotationsFromReconstruction(
    const std::string& target_reconstruction_filepath,
    const std::vector<std::string>& image_names,
    const theia::Reconstruction& ref_reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations) {
  CHECK_NOTNULL(modelview_rotations);

  theia::Reconstruction target_reconstruction;
  CHECK(theia::ReadReconstruction(
      target_reconstruction_filepath, &target_reconstruction))
  << "Could not read reconstruction file: '"
  << target_reconstruction_filepath << "'.";

  for (const std::string& image_name : image_names) {
    const theia::ViewId target_view_id =
        target_reconstruction.ViewIdFromName(image_name);
    if (target_view_id == theia::kInvalidViewId) {
      continue;
    }

    const theia::View* view = target_reconstruction.View(target_view_id);
    const theia::Camera& camera = view->Camera();
    const Eigen::Affine3d modelview = ComputeModelviewFromTheiaCamera(camera);

    // NOTE:
    // Use reference reconstruction for view ID.
    const theia::ViewId ref_view_id =
        ref_reconstruction.ViewIdFromName(image_name);
    if (ref_view_id != theia::kInvalidViewId) {
      modelview_rotations->emplace(ref_view_id, modelview.rotation());
    }
  }
}

void GetRotationsFromReconstruction(
    const theia::Reconstruction& reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations) {
  CHECK_NOTNULL(modelview_rotations);

  for (const theia::ViewId view_id : reconstruction.ViewIds()) {
    const theia::View* view = reconstruction.View(view_id);
    const theia::Camera& camera = view->Camera();
    const Eigen::Affine3d modelview = ComputeModelviewFromTheiaCamera(camera);
    modelview_rotations->emplace(view_id, modelview.rotation());
  }
}

void ComputeRelativeRotationFromFirstFrame(
    const std::unordered_map<theia::ViewId, Eigen::Matrix3d>& rotations,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* relative_rotations) {
  CHECK_NOTNULL(relative_rotations);

  const theia::ViewId kIdZero = 0;
  if (rotations.find(kIdZero) == rotations.end()) {
    CHECK(false) << "The first frame does not exist.";
  }
  const Eigen::Matrix3d first_R = rotations.at(kIdZero);

  relative_rotations->clear();
  for (auto& R_pair : rotations) {
    const theia::ViewId view_id = R_pair.first;
    const Eigen::Matrix3d R = R_pair.second;
    relative_rotations->emplace(view_id, R * first_R.transpose());
  }
}

void SyncRotationLists(
    const std::unordered_map<theia::ViewId, Eigen::Matrix3d>& ref_rotations,
    const std::unordered_map<theia::ViewId, Eigen::Matrix3d>& est_rotations,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* synced_est_rotations) {
  CHECK_NOTNULL(synced_est_rotations);

  // Find global_R that makes ref_R_i = est_R_i global_R.
  std::vector<Eigen::Matrix3d> diff_R_list;
  diff_R_list.reserve(est_rotations.size());

  for (const auto& est_R_pair : est_rotations) {
    const theia::ViewId view_id = est_R_pair.first;
    const Eigen::Matrix3d est_R = est_R_pair.second;

    if (ref_rotations.find(view_id) != ref_rotations.end()) {
      const Eigen::Matrix3d ref_R = ref_rotations.at(view_id);
      const Eigen::Matrix3d diff_R = est_R.transpose() * ref_R;
      diff_R_list.push_back(diff_R);
    }
  }

  CHECK(!diff_R_list.empty());
  const Eigen::Matrix3d global_R = ComputeAverageRotation(diff_R_list);

  // Post-multiply transpose of global_R.
  synced_est_rotations->clear();
  for (auto& est_R_pair : est_rotations) {
    const theia::ViewId view_id = est_R_pair.first;
    const Eigen::Matrix3d est_R = est_R_pair.second;
    synced_est_rotations->emplace(view_id, est_R * global_R);
  }
}

void WriteFovy(const theia::Reconstruction& reconstruction) {
  // Compute fovy for each view.
  for (const theia::ViewId view_id : reconstruction.ViewIds()) {
    const theia::View* view = reconstruction.View(view_id);
    const theia::Camera& camera = view->Camera();

    // fovy.
    const int kImageHeight = 1080;

    const double f = 2.0 * camera.FocalLength() / (double) kImageHeight;
    std::cout << "f: " << f << std::endl;

    // f = 1 / tan(fovx / 2).
    // fovx = 2 * atan(1 / f).
    const double fovy = (2.0 * std::atan(1.0 / f)) / M_PI * 180.0;
    std::cout << "View ID: " << view_id << ", "
              << "Fovy: " << fovy << std::endl;
  }
}

void WriteModelviews(
    const theia::Reconstruction& reconstruction,
    const std::string& output_dir) {
  CHECK_NE(output_dir, "");
  if (!stlplus::folder_exists(output_dir)) {
    CHECK(stlplus::folder_create(output_dir));
  }
  CHECK(stlplus::folder_writable(output_dir));

  // Save OpenGL modelview matrix for each view.
  for (const theia::ViewId view_id : reconstruction.ViewIds()) {
    const theia::View* view = reconstruction.View(view_id);
    const theia::Camera& camera = view->Camera();

    // Save OpenGL modelview matrix.
    const Eigen::Affine3d modelview =
        ComputeModelviewFromTheiaCamera(camera);

    const std::string basename = stlplus::basename_part(view->Name());
    const std::string modelview_filename = output_dir + "/" + basename + ".txt";
    std::cout << "Saving '" << modelview_filename << "'... ";

    std::ofstream file(modelview_filename);
    // Column-wise
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        file << modelview.matrix()(j, i) << std::endl;
    file.close();
    std::cout << "Done." << std::endl;
  }
}
