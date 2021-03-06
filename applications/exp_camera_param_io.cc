// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_camera_param_io.h"

#include <fstream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <sstream>

#include "exp_camera_param_utils.h"
#include "exp_matrix_utils.h"
#include "unsupported/Eigen/MatrixFunctions"


bool ReadModelviews(
    const std::string& data_type, const std::string& filepath,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews) {
  if (data_type == "pose") {
    ReadModelviewsFromCameraMatrices(filepath, modelviews);
  }
  else if (data_type == "modelview") {
    ReadModelviewsFromModelviews(filepath, modelviews);
  }
  else if (data_type == "reconstruction") {
    ReadModelviewsFromReconstruction(filepath, modelviews);
  }
  else {
    LOG(WARNING) << "Data type must be either 'param', 'modelview', or "
                 << "'reconstruction' (Current: " << data_type << ")";
    return false;
  }
  return true;
}

void ReadModelviewsFromCameraMatrices(
    const std::string& camera_matrix_dir,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews) {
  CHECK_NE(camera_matrix_dir, "");
  CHECK_NOTNULL(modelviews);
  modelviews->clear();

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
    const Eigen::Affine3d theia_modelview = camera_pose.inverse();
    const Eigen::Matrix3d axes_converter = GetVisionToOpenGLAxesConverter();
    const Eigen::Affine3d modelview = axes_converter * theia_modelview;
    (*modelviews)[basename] = modelview;
    VLOG(3) << "Loaded '" << filepath << "'.";
  }
}

void ReadModelviewsFromModelviews(
    const std::string& modelview_dir,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews) {
  CHECK_NE(modelview_dir, "");
  CHECK_NOTNULL(modelviews);
  modelviews->clear();

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
    (*modelviews)[basename] = modelview;
    VLOG(3) << "Loaded '" << filepath << "'.";
  }
}

void ReadModelviewsFromReconstruction(
    const std::string& reconstruction_filepath,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews) {
  CHECK_NE(reconstruction_filepath, "");
  CHECK_NOTNULL(modelviews);
  modelviews->clear();

  Reconstruction reconstruction;
  CHECK(ReadReconstruction(reconstruction_filepath, &reconstruction))
  << "Could not read reconstruction file: '"
  << reconstruction_filepath << "'.";

  for (const auto& view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    const std::string basename = stlplus::basename_part(view->Name());
    const Camera& camera = view->Camera();
    const Eigen::Affine3d modelview = ComputeModelviewFromTheiaCamera(camera);
    (*modelviews)[basename] = modelview;
    VLOG(3) << "Loaded '" << basename << "'.";
  }
}

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
    Eigen::Vector3d camera_params;
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
    const Eigen::Affine3d theia_modelview = camera_pose.inverse();
    const Eigen::Matrix3d orientation = theia_modelview.rotation();
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

  // Create directory.
  if (!stlplus::folder_exists(camera_param_dir)) {
    CHECK(stlplus::folder_create(camera_param_dir));
  }
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
    for(int i = 0; i < 3; ++i) file << camera_params[i] << " ";
    file.close();
    VLOG(3) << "Saved '" << filepath << "'.";
  }
}

void WriteOrientationsAsCameraParams(
    const std::string& camera_param_dir, const Reconstruction& reconstruction) {
  CHECK_NE(camera_param_dir, "");

  // Create directory.
  if (stlplus::folder_exists(camera_param_dir)) {
    CHECK(stlplus::folder_create(camera_param_dir));
  }
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
    for(int i = 0; i < 3; ++i) file << camera_params[i] << " ";
    file.close();
    VLOG(3) << "Saved '" << filepath << "'.";
  }
}

void WriteModelviews(
    const std::string& modelview_dir,
    const std::unordered_map<std::string, Eigen::Affine3d>& modelviews) {
  CHECK_NE(modelview_dir, "");

  // Create directory.
  if (!stlplus::folder_exists(modelview_dir)) {
    CHECK(stlplus::folder_create(modelview_dir));
  }
  CHECK(stlplus::folder_writable(modelview_dir));

  for (const auto& modelview : modelviews) {
    const std::string basename = modelview.first;
    const std::string filepath = modelview_dir + "/" + basename + ".txt";

    // Write modelview matrix.
    CHECK(WriteEigenMatrixToCSV(filepath, modelview.second.matrix()));
    VLOG(3) << "Saved '" << filepath << "'.";
  }
}

void WriteModelviews(
    const std::string& modelview_dir, const Reconstruction& reconstruction) {
  CHECK_NE(modelview_dir, "");

  // Create directory.
  if (!stlplus::folder_exists(modelview_dir)) {
    CHECK(stlplus::folder_create(modelview_dir));
  }
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

void SyncOrientationSequences(
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    reference_orientations,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    estimated_orientations,
    std::unordered_map<std::string, Eigen::Matrix3d>*
    synced_estimated_orientations) {
  CHECK_NOTNULL(synced_estimated_orientations)->clear();

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

void SyncOrientationSequencesWithPivot(
    const std::string& pivot_view_name,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    reference_orientations,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    estimated_orientations,
    std::unordered_map<std::string, Eigen::Matrix3d>*
    synced_estimated_orientations) {
  CHECK_NOTNULL(synced_estimated_orientations)->clear();

  if (estimated_orientations.empty()) return;
  synced_estimated_orientations->reserve(estimated_orientations.size());

  const Eigen::Matrix3d ref_R =
      FindOrDie(reference_orientations, pivot_view_name);
  const Eigen::Matrix3d est_R =
      FindOrDie(estimated_orientations, pivot_view_name);
  const Eigen::Matrix3d global_R = est_R.transpose() * ref_R;

  // Post-multiply transpose of global_R.
  for (auto& est_R_pair : estimated_orientations) {
    const std::string view_name = est_R_pair.first;
    const Eigen::Matrix3d est_R = est_R_pair.second;
    (*synced_estimated_orientations)[view_name] = est_R * global_R;
  }
}

void SyncModelviewSequences(
    const std::unordered_map<std::string, Eigen::Affine3d>&
    reference_modelviews,
    const std::unordered_map<std::string, Eigen::Affine3d>&
    estimated_modelviews,
    std::unordered_map<std::string, Eigen::Affine3d>*
    synced_estimated_modelviews) {
  CHECK_NOTNULL(synced_estimated_modelviews)->clear();

  if (estimated_modelviews.empty()) return;
  synced_estimated_modelviews->reserve(estimated_modelviews.size());

  // Find global_M that makes ref_M_i = est_M_i global_M.
  // => [ref_R_i ref_t_i] = [est_R_i est_t_i] [global_R global_t].
  // => [ref_R_i ref_t_i] =
  //    [(est_R_i * global_R) (est_R_i * global_t + est_t_i)].
  // => ref_R_i = est_R_i * global_R and,
  //    ref_t_i = est_R_i * global_t + est_t_i.
  // => global_R = est_R_i^T * ref_R_i,
  //    global_R = est_R_i^T * (ref_t_i - est_t_i).
  std::vector<Eigen::Matrix3d> diff_R_list;
  std::vector<Eigen::Vector3d> diff_t_list;
  diff_R_list.reserve(estimated_modelviews.size());
  diff_t_list.reserve(estimated_modelviews.size());

  for (const auto& est_M_pair : estimated_modelviews) {
    const std::string view_name = est_M_pair.first;
    const Eigen::Affine3d est_M = est_M_pair.second;

    if (reference_modelviews.find(view_name) !=
        reference_modelviews.end()) {
      const Eigen::Affine3d ref_M = reference_modelviews.at(view_name);

      const Eigen::Matrix3d diff_R =
          est_M.rotation().transpose() * ref_M.rotation();
      diff_R_list.push_back(diff_R);

      const Eigen::Vector3d diff_t = est_M.rotation().transpose() *
          (ref_M .translation() - est_M.translation());
      diff_t_list.push_back(diff_t);
    }
  }

  CHECK(!diff_R_list.empty());
  const Eigen::Matrix3d global_R = ComputeAverageRotation(diff_R_list);
  CHECK(!diff_t_list.empty());
  const Eigen::Vector3d global_t = ComputeAverageTranslation(diff_t_list);
  Eigen::Affine3d global_M;
  global_M.rotate(global_R);
  global_M.translate(global_t);

  // DEBUG.
  CHECK_EQ(global_M.translation(), global_t);

  // Post-multiply transpose of global_M.
  for (auto& est_M_pair : estimated_modelviews) {
    const std::string view_name = est_M_pair.first;
    const Eigen::Affine3d est_M = est_M_pair.second;
    (*synced_estimated_modelviews)[view_name] = est_M * global_M;
  }
}

bool ReadSequenceIndices(
    const std::string filepath,
    std::unordered_map<std::string, int>* _seq_indices) {
  CHECK_NOTNULL(_seq_indices);

  std::ifstream file(filepath);
  if (!file.good()) {
    LOG(WARNING) << "Can't read file: '" << filepath << "'.";
    return false;
  }

  _seq_indices->clear();

  std::string line;
  while(std::getline(file, line)) {
    if (line == "") break;
    std::stringstream sstr(line);

    std::string image_name;
    if (!std::getline(sstr, image_name, ',')) {
      LOG(WARNING) << "Wrong file format: '" << line << "'.";
      return false;
    }

    std::string seq_index_str;
    if (!std::getline(sstr, seq_index_str)) {
      LOG(WARNING) << "Wrong file format: '" << line << "'.";
      return false;
    }
    const int seq_index = std::stoi(seq_index_str);

    (*_seq_indices)[image_name] = seq_index;
  }

  file.close();
  return true;
}

std::unique_ptr<Reconstruction> CreateTheiaReconstructionFromModelviews(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>*
  camera_intrinsics_priors) {
  std::unique_ptr<Reconstruction> reconstruction(new Reconstruction());

  for (const auto& modelview : modelviews) {
    const ViewId view_id = reconstruction->AddView(modelview.first);
    View* view = reconstruction->MutableView(view_id);

    // Set camera intrinsics if provieded.
    if (camera_intrinsics_priors) {
      CameraIntrinsicsPrior* camera_intrinsics_prior =
        view->MutableCameraIntrinsicsPrior();

      const CameraIntrinsicsPrior* retrieved_camera_intrinsics_prior =
        FindOrNull(*camera_intrinsics_priors, modelview.first);
      if (retrieved_camera_intrinsics_prior) {
        (*camera_intrinsics_prior) = (*retrieved_camera_intrinsics_prior);
      }
    }

    // Set camera extrinsics.
    Camera* camera = view->MutableCamera();
    ComputeTheiaCameraFromModelview(modelview.second, camera);

    // Set view as estimated.
    view->SetEstimated(true);
  }

  return std::move(reconstruction);
}
