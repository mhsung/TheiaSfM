// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_camera_param_io.h"

#include <fstream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <stlplus3/file_system.hpp>

#include "exp_camera_param_utils.h"
#include "unsupported/Eigen/MatrixFunctions"


bool ReadOrientations(
    const std::string& data_type, const std::string& filepath,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations) {
  if (data_type == "param") {
    ReadOrientationsFromCameraParams(filepath, orientations);
  }
  else if (data_type == "pose") {
    ReadOrientationsFromCameraMatrices(filepath, orientations);
  }
  else if (data_type == "modelview") {
    ReadOrientationsFromModelviews(filepath, orientations);
  }
  else if (data_type == "reconstruction") {
    ReadOrientationsFromReconstruction(filepath, orientations);
  }
  else {
    LOG(WARNING) << "Data type must be either 'param', 'modelview', or "
                 << "'reconstruction' (Current: " << data_type << ")";
    return false;
  }
  return true;
}

void ReadOrientationsFromCameraParams(
    const std::string& camera_param_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations) {
  CHECK_NE(camera_param_dir, "");
  CHECK_NOTNULL(orientations);
  orientations->clear();

  for(const auto& filename : stlplus::folder_files(camera_param_dir)) {
    const std::string basename = stlplus::basename_part(filename);
    const std::string filepath = camera_param_dir + "/" + filename;
    if (!FileExists(filepath)) {
      LOG(WARNING) << "File does not exist: '" << filepath << "'";
      continue;
    }

    // Read camera parameters.
    std::ifstream file(filepath);
    Eigen::Vector3i camera_params;
    for(int i = 0; i < 3; ++i) file >> camera_params[i];
    file.close();

    const Eigen::Matrix3d orientation =
        ComputeTheiaCameraRotationFromCameraParams(
            camera_params.cast<double>());

//    // DEBUG.
//    const Eigen::Vector3d test_camera_param =
//        ComputeCameraParamsFromTheiaCameraRotation(orientation);
//    for (int i = 0; i < 3; i++) {
//      CHECK_EQ(camera_params[i], static_cast<int>(std::round
//          (test_camera_param[i])) % 360);
//    }
//    //

    (*orientations)[basename] = orientation;
    VLOG(3) << "Loaded '" << filepath << "'.";
  }
}

void ReadOrientationsFromCameraMatrices(
    const std::string& camera_matrix_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations) {
  CHECK_NE(camera_matrix_dir, "");
  CHECK_NOTNULL(orientations);
  orientations->clear();

  for(const auto& filename : stlplus::folder_files(camera_matrix_dir)) {
    const std::string basename = stlplus::basename_part(filename);
    const std::string filepath = camera_matrix_dir + "/" + filename;
    if (!FileExists(filepath)) {
      LOG(WARNING) << "File does not exist: '" << filepath << "'";
      continue;
    }

    // Read modelview matrix.
    Eigen::Matrix4d camera_matrix;
    CHECK(ReadEigenMatrixFromCSV(filepath, &camera_matrix));
    const Eigen::Affine3d camera_pose(camera_matrix);
    const Eigen::Matrix3d orientation = camera_pose.inverse().rotation();
    (*orientations)[basename] = orientation;
    VLOG(3) << "Loaded '" << filepath << "'.";
  }
}

void ReadOrientationsFromModelviews(
    const std::string& modelview_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations) {
  CHECK_NE(modelview_dir, "");
  CHECK_NOTNULL(orientations);
  orientations->clear();

  for(const auto& filename : stlplus::folder_files(modelview_dir)) {
    const std::string basename = stlplus::basename_part(filename);
    const std::string filepath = modelview_dir + "/" + filename;
    if (!FileExists(filepath)) {
      LOG(WARNING) << "File does not exist: '" << filepath << "'";
      continue;
    }

    // Read modelview matrix.
    Eigen::Matrix4d modelview_matrix;
    CHECK(ReadEigenMatrixFromCSV(filepath, &modelview_matrix));
    const Eigen::Affine3d modelview(modelview_matrix);
    const Eigen::Matrix3d orientation =
        ComputeTheiaCameraRotationFromModelview(modelview.rotation());
    (*orientations)[basename] = orientation;
    VLOG(3) << "Loaded '" << filepath << "'.";
  }
}

void ReadOrientationsFromReconstruction(
    const std::string& reconstruction_filepath,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations) {
  CHECK_NE(reconstruction_filepath, "");
  CHECK_NOTNULL(orientations);
  orientations->clear();

  Reconstruction reconstruction;
  CHECK(ReadReconstruction(reconstruction_filepath, &reconstruction))
  << "Could not read reconstruction file: '"
  << reconstruction_filepath << "'.";

  for (const auto& view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    const std::string basename = stlplus::basename_part(view->Name());
    const Camera& camera = view->Camera();
    const Eigen::Matrix3d orientation =
        camera.GetOrientationAsRotationMatrix();
    (*orientations)[basename] = orientation;
    VLOG(3) << "Loaded '" << basename << "'.";
  }
}

void WriteOrientationsAsCameraParams(
    const std::string& camera_param_dir,
    const std::unordered_map<std::string, Eigen::Matrix3d>& orientations) {
  CHECK_NE(camera_param_dir, "");

  // Empty directory.
  if (stlplus::folder_exists(camera_param_dir)) {
    CHECK(stlplus::folder_delete(camera_param_dir, true));
  }
  CHECK(stlplus::folder_create(camera_param_dir));
  CHECK(stlplus::folder_writable(camera_param_dir));

  for (const auto& orientation : orientations) {
    const Eigen::Vector3d camera_params =
        ComputeCameraParamsFromTheiaCameraRotation(orientation.second);

//    // DEBUG.
//    const double kErrorThreshold = 1.E-3;
//    const Eigen::Matrix3d test_theia_rotation_matrix =
//        ComputeTheiaCameraRotationFromCameraParams(camera_params);
//    CHECK(test_theia_rotation_matrix.isApprox(
//        orientation.second, kErrorThreshold));
//    //

    const std::string basename = orientation.first;
    const std::string filepath = camera_param_dir + "/" + basename + ".txt";

    // Write camera parameters.
    std::ofstream file(filepath);
    const double* values = camera_params.data();
    for(int i = 0; i < 3; ++i) file << values[i] << " ";
    file.close();
    VLOG(3) << "Saved '" << filepath << "'.";
  }
}

void WriteOrientationsAsCameraParams(
    const std::string& camera_param_dir, const Reconstruction& reconstruction) {
  CHECK_NE(camera_param_dir, "");

  // Empty directory.
  if (stlplus::folder_exists(camera_param_dir)) {
    CHECK(stlplus::folder_delete(camera_param_dir, true));
  }
  CHECK(stlplus::folder_create(camera_param_dir));
  CHECK(stlplus::folder_writable(camera_param_dir));

  for (const ViewId view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    const Camera& camera = view->Camera();
    const Eigen::Vector3d camera_params =
        ComputeCameraParamsFromTheiaCamera(camera);

    const std::string basename = stlplus::basename_part(view->Name());
    const std::string filepath = camera_param_dir + "/" + basename + ".txt";

    // Write camera parameters.
    std::ofstream file(filepath);
    const double* values = camera_params.data();
    for(int i = 0; i < 3; ++i) file << values[i] << " ";
    file.close();
    VLOG(3) << "Saved '" << filepath << "'.";
  }
}

void WriteModelviews(
    const std::string& modelview_dir, const Reconstruction& reconstruction) {
  CHECK_NE(modelview_dir, "");

  // Empty directory.
  if (stlplus::folder_exists(modelview_dir)) {
    CHECK(stlplus::folder_delete(modelview_dir, true));
  }
  CHECK(stlplus::folder_create(modelview_dir));
  CHECK(stlplus::folder_writable(modelview_dir));

  for (const ViewId view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    const Camera& camera = view->Camera();
    const Eigen::Affine3d modelview = ComputeModelviewFromTheiaCamera(camera);

    const std::string basename = stlplus::basename_part(view->Name());
    const std::string filepath = modelview_dir + "/" + basename + ".txt";

    // Write modelview matrix.
    CHECK(WriteEigenMatrixToCSV(filepath, modelview.matrix()));
    VLOG(3) << "Saved '" << filepath << "'.";
  }
}

void PrintFovy(const int image_height, const Reconstruction& reconstruction) {
  // Compute fovy for each view.
  for (const ViewId view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    const Camera& camera = view->Camera();

    const double f = 2.0 * camera.FocalLength() / (double) image_height;
    std::cout << "f: " << f << std::endl;

    // f = 1 / tan(fovx / 2).
    // fovx = 2 * atan(1 / f).
    const double fovy = (2.0 * std::atan(1.0 / f)) / M_PI * 180.0;
    std::cout << "View ID: " << view_id << ", "
              << "Fovy: " << fovy << std::endl;
  }
}

void GetOrientationsFromReconstruction(
    const Reconstruction& reconstruction,
    std::unordered_map<ViewId, Eigen::Matrix3d>* orientations) {
  CHECK_NOTNULL(orientations);

  for (const ViewId view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    const Camera& camera = view->Camera();
    (*orientations)[view_id] = camera.GetOrientationAsRotationMatrix();
  }
}

void GetOrientationsFromReconstruction(
    const Reconstruction& reconstruction,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations) {
  CHECK_NOTNULL(orientations);

  for (const ViewId view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    const Camera& camera = view->Camera();
    const std::string basename = stlplus::basename_part(view->Name());
    (*orientations)[basename] = camera.GetOrientationAsRotationMatrix();
  }
}

void MapOrientationsToViewIds(
    const Reconstruction& reconstruction,
    const std::unordered_map<std::string, Eigen::Matrix3d>& name_orientations,
    std::unordered_map<ViewId, Eigen::Matrix3d>* id_orientations) {
  CHECK_NOTNULL(id_orientations);
  id_orientations->clear();
  id_orientations->reserve(name_orientations.size());

  for(const auto& name_orientation : name_orientations) {
    // FIXME:
    // Now we simply assume that all view name has '.png' extension.
    const auto& view_name = name_orientation.first + ".png";
    const ViewId view_id = reconstruction.ViewIdFromName(view_name);
    if (view_id == kInvalidViewId) {
      LOG(WARNING) << "Invalid view name: '" << view_name << "'";
    } else {
      const auto& orientation = name_orientation.second;
      (*id_orientations)[view_id] = orientation;
    }
  }
}

void MapOrientationsToViewNames(
    const Reconstruction& reconstruction,
    const std::unordered_map<ViewId, Eigen::Matrix3d>& id_orientations,
    std::unordered_map<std::string, Eigen::Matrix3d>* name_orientations) {
  CHECK_NOTNULL(name_orientations);
  name_orientations->clear();
  name_orientations->reserve(id_orientations.size());

  for(const auto& id_orientation : id_orientations) {
    const ViewId view_id = id_orientation.first;
    const View* view = reconstruction.View(view_id);
    if (view == nullptr) {
      LOG(WARNING) << "Invalid view ID: '" << view_id << "'";
    } else {
      const std::string basename = stlplus::basename_part(view->Name());
      const auto& orientation = id_orientation.second;
      (*name_orientations)[basename] = orientation;
    }
  }
}

void ComputeRelativeOrientationsFromFirstFrame(
    const std::unordered_map<ViewId, Eigen::Matrix3d>& orientations,
    std::unordered_map<ViewId, Eigen::Matrix3d>* relative_orientations) {
  CHECK_NOTNULL(relative_orientations);

  const ViewId kIdZero = 0;
  if (orientations.find(kIdZero) == orientations.end()) {
    CHECK(false) << "The first frame does not exist.";
  }
  const Eigen::Matrix3d first_R = orientations.at(kIdZero);

  relative_orientations->clear();
  relative_orientations->reserve(orientations.size());
  for (auto& R_pair : orientations) {
    const ViewId view_id = R_pair.first;
    const Eigen::Matrix3d R = R_pair.second;
    (*relative_orientations)[view_id] = R * first_R.transpose();
  }
}

void SyncOrientationSequences(
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    reference_orientations,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    estimated_orientations,
    std::unordered_map<std::string, Eigen::Matrix3d>*
    synced_estimated_orientations) {
  CHECK_NOTNULL(synced_estimated_orientations);

  synced_estimated_orientations->clear();
  if (estimated_orientations.empty()) return;
  synced_estimated_orientations->reserve(estimated_orientations.size());

  // Find global_R that makes ref_R_i = est_R_i global_R.
  // => est_R_i^T ref_R_i = global_R.
  // => Find global_R that minimizes d(est_R_i^T ref_R_i, global_R),
  // where d(*,*) is a distance metric of rotation matrices.
  std::vector<Eigen::Matrix3d> diff_R_list;
  diff_R_list.reserve(estimated_orientations.size());

  for (const auto& est_R_pair : estimated_orientations) {
    const std::string view_name = est_R_pair.first;
    const Eigen::Matrix3d est_R = est_R_pair.second;

    if (reference_orientations.find(view_name) !=
        reference_orientations.end()) {
      const Eigen::Matrix3d ref_R = reference_orientations.at(view_name);
      const Eigen::Matrix3d diff_R = est_R.transpose() * ref_R;
      diff_R_list.push_back(diff_R);
    }
  }

  CHECK(!diff_R_list.empty());
  const Eigen::Matrix3d global_R = ComputeAverageRotation(diff_R_list);

  // Post-multiply transpose of global_R.
  for (auto& est_R_pair : estimated_orientations) {
    const std::string view_name = est_R_pair.first;
    const Eigen::Matrix3d est_R = est_R_pair.second;
    (*synced_estimated_orientations)[view_name] = est_R * global_R;
  }

  // (Optional) Report angle difference before/after sync.
  double sum_before_angles = 0.0, sum_after_angles = 0.0;
  for (auto& est_R_pair : estimated_orientations) {
    const std::string view_name = est_R_pair.first;
    const Eigen::Matrix3d est_R = est_R_pair.second;

    if (reference_orientations.find(view_name) !=
        reference_orientations.end()) {
      const Eigen::Matrix3d ref_R = reference_orientations.at(view_name);
      const Eigen::AngleAxisd R1(ref_R.transpose() * est_R);
      const Eigen::AngleAxisd R2(ref_R.transpose() * est_R * global_R);
      sum_before_angles += (R1.angle() / M_PI * 180.0);
      sum_after_angles += (R2.angle() / M_PI * 180.0);
    }
  }

  VLOG(3) << "Angle difference (before): "
          << sum_before_angles / estimated_orientations.size();
  VLOG(3) << "Angle difference (after): "
          << sum_after_angles / estimated_orientations.size();
}
