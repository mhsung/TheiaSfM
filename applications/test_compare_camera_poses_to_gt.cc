// Copied from 'build_reconstruction.cc'

#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>
#include <fstream>
#include <iostream>
#include <stlplus3/file_system.hpp>
#include <string>
#include <vector>

#include "unsupported/Eigen/MatrixFunctions"


// Input/output files.
DEFINE_string(images, "", "Wildcard of images to reconstruct.");
DEFINE_string(input_reconstruction_file, "",
              "Input reconstruction file in binary format.");
DEFINE_string(gt_camera_param_directory, "",
              "Directory containing ground truth camera parameters.");
DEFINE_string(gt_modelview_directory, "",
              "Directory containing ground truth modelview matrice.");
DEFINE_string(diff_orientation_directory, "",
              "Directory where orientation difference between final and "
                  "initial camera poses are saved.");
DEFINE_string(diff_param_directory, "",
              "Directory where the camera parameters of orientation difference "
                  "between final and initial camera poses are saved.");

using theia::Reconstruction;


Eigen::Matrix4d GetCameraPose(const theia::Camera& camera) {
  Eigen::Matrix4d camera_pose = Eigen::Matrix4d::Zero();
  camera_pose.block<3, 3>(0, 0) =
      camera.GetOrientationAsRotationMatrix().transpose();
  camera_pose.col(3).head<3>() = camera.GetPosition();
  camera_pose(3, 3) = 1.0;
  return camera_pose;
}

Eigen::Matrix3d GetOrientationFromCameraPose(
    const Eigen::Matrix4d& camera_pose) {
  return camera_pose.block<3, 3>(0, 0).transpose();
}

Eigen::Matrix4d CameraPoseToOpenGLModelview(
    const Eigen::Matrix4d& camera_pose) {
  // (X, Y, Z) -> (X, -Y, -Z).
  Eigen::Matrix4d axes_converter = Eigen::Matrix4d::Identity();
  axes_converter(0, 0) = 1.0;
  axes_converter(1, 1) = -1.0;
  axes_converter(2, 2) = -1.0;

  const Eigen::Matrix4d modelview =
      axes_converter * Eigen::Affine3d(camera_pose).inverse().matrix();
  return modelview;
}

Eigen::Matrix4d OpenGLModelviewToCameraPose(
    const Eigen::Matrix4d& modelview) {
  // (X, Y, Z) -> (X, -Y, -Z).
  Eigen::Matrix4d axes_converter = Eigen::Matrix4d::Identity();
  axes_converter(0, 0) = 1.0;
  axes_converter(1, 1) = -1.0;
  axes_converter(2, 2) = -1.0;

  const Eigen::Matrix4d camera_pose =
      Eigen::Affine3d(modelview).inverse().matrix() * axes_converter;
  return camera_pose;
}

Eigen::Affine3d ComputeModelviewFromCameraParams(
    const Eigen::Vector3d& camera_params) {
  const double azimuth_deg = camera_params[0];
  const double elevation_deg = camera_params[1];
  const double theta_deg = camera_params[2];
  const double translation_distance = 1.5;

  Eigen::Affine3d modelview(Eigen::Affine3d::Identity());

  // Default transformation ([R_d | 0]).
  modelview.prerotate(Eigen::AngleAxisd(-0.5 * M_PI, Eigen::Vector3d::UnitY()));

  // Y-axis (Azimuth, [R_y | 0])
  // Note: Change the sign.
  const double azimuth = (double) -azimuth_deg / 180.0 * M_PI;
  modelview.prerotate(Eigen::AngleAxisd(azimuth, Eigen::Vector3d::UnitY()));

  // X-axis (Elevation, [R_x | 0])
  const double elevation = (double) elevation_deg / 180.0 * M_PI;
  modelview.prerotate(Eigen::AngleAxisd(elevation, Eigen::Vector3d::UnitX()));

  // Translation ([I | t])
  modelview.pretranslate(Eigen::Vector3d(0, 0, -translation_distance));

  // Z-axis (Theta, [R_z | 0])
  // Note: Change the sign.
  const double theta = (double) -theta_deg / 180.0 * M_PI;
  modelview.prerotate(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));

  // T = [R_z | 0] [I | t] [R_x | 0] [R_y | 0] [R_d | 0].
  return modelview;
}

Eigen::Vector3d ComputeParamsFromRotation(
    const Eigen::Matrix3d& rotation) {
  // ZXY Euler angles.
  Eigen::Vector3d angles = rotation.eulerAngles(2, 0, 1);

  Eigen::Vector3d camera_params;
  // Y-axis (Azimuth, [R_y | 0])
  // Note: Change the sign.
  camera_params[0] = -angles[2] / M_PI * 180.0;

  // X-axis (Elevation, [R_x | 0])
  camera_params[1] = +angles[1] / M_PI * 180.0;

  // Z-axis (Theta, [R_z | 0])
  // Note: Change the sign.
  camera_params[2] = -angles[0] / M_PI * 180.0;


  // Elevation must be in range [-90, 90].
  while(camera_params[1] < -90.0) camera_params[1] += 360.0;

  if (camera_params[1] > 270.0) {
    camera_params[1] = 360.0 - camera_params[1];
  } else if (camera_params[1] > 90.0) {
    camera_params[1] = 180.0 - camera_params[1];
    camera_params[0] += 180.0;
    camera_params[2] += 180.0;
  }

  // Azimuth and theta must be in range [0, 360).
  while(camera_params[0] >= 360.0) camera_params[0] -= 360.0;
  while(camera_params[0] < 0.0) camera_params[0] += 360.0;

  while(camera_params[2] >= 360.0) camera_params[2] -= 360.0;
  while(camera_params[2] < 0.0) camera_params[2] += 360.0;

  return camera_params;
}

Eigen::Vector3d ComputeCameraParamsFromModelview(
    const Eigen::Matrix3d& modelview) {
  // R = R_z R_x R_y R_d.
  Eigen::Matrix3d default_rotation =
      Eigen::AngleAxisd(-0.5 * M_PI, Eigen::Vector3d::UnitY())
          .toRotationMatrix();

  // Post-multiply inverse(transpose) of default transformation.
  // R R_d^T = R_z R_x R_y.
  const Eigen::Matrix3d rotation = modelview * default_rotation.transpose();

  return ComputeParamsFromRotation(rotation);
}

Eigen::Matrix3d ComputeOrientationFromCameraParams(
    const Eigen::Vector3d& camera_params) {
  const Eigen::Affine3d modelview = ComputeModelviewFromCameraParams(
      camera_params);

  // Take inverse(transpose) of modelview rotation
  // to get camera pose rotation.
  const Eigen::Matrix3d camera_orientation = modelview.rotation().transpose();
  return camera_orientation;
}

Eigen::Vector3d ComputeCameraParamsFromOrientation(
    const Eigen::Matrix3d& camera_orientation) {
  // Take inverse(transpose) of camera orientation
  // to get modelview rotation.
  const Eigen::Matrix3d modelview_rotation = camera_orientation.transpose();

  const Eigen::Vector3d camera_params =
      ComputeCameraParamsFromModelview(modelview_rotation);

//  // Debug.
//  const Eigen::Affine3d test_modelview = ComputeModelviewFromCameraParams(
//      camera_params);
//  const Eigen::Vector3i test_camera_params = ComputeCameraParamsFromModelview
//      (test_modelview.rotation());
//  CHECK_EQ(camera_params.transpose(), test_camera_params.transpose());

  return camera_params;
}

Eigen::Matrix3d ComputeAverageRotation(const std::vector<Eigen::Matrix3d>& Rs) {
  const int num_Rs = Rs.size();
  CHECK_GT(num_Rs, 0);

  Eigen::Matrix3d avg_R = Eigen::Matrix3d::Identity();

  // Minimize under the geodesic metric.
  // Hartley et al., Minimization under the geodesic metric, CVPR 2011.
  // Algorithm 1.
  const int MAX_ITER = 10;
  for (int iter = 0; iter < MAX_ITER; ++iter) {
    Eigen::Matrix3d r = Eigen::Matrix3d::Zero();
    for (const Eigen::Matrix3d diff_R : Rs) {
      r += (avg_R.transpose() * diff_R).log();
    }
    r = r / (double) num_Rs;
    avg_R = avg_R * r.exp();

    const double error = r.norm();
    std::cout << "[" << iter << "] error = " << error << std::endl;
  }

  std::cout << "Done." << std::endl;
  return avg_R;
}

void ReadOrientationsFromCameraParams(
    const std::string& base_dir,
    const std::vector<std::string>& image_filenames,
    const Reconstruction& reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* camera_orientations) {
  CHECK(camera_orientations != nullptr);

  for (const std::string& image_filename : image_filenames) {
    const std::string base_name = stlplus::basename_part(image_filename);

    std::string camera_file = base_dir + "/" + base_name + ".txt";
    if (!theia::FileExists(camera_file)) {
      LOG(WARNING) << "File does not exist: '" << base_name << "'";
      continue;
    }

    // Read camera pose parameters.
    std::ifstream f(camera_file);
    Eigen::Vector3i camera_params;
    for(int i = 0; i < 3; ++i) f >> camera_params[i];
    f.close();

    const Eigen::Matrix3d camera_orientation =
        ComputeOrientationFromCameraParams(camera_params.cast<double>());

    const theia::ViewId view_id =
        reconstruction.ViewIdFromName(image_filename);
    if (view_id != theia::kInvalidViewId) {
      camera_orientations->emplace(view_id, camera_orientation);
    }
  }
}

void ReadOrientationsFromModelviews(
    const std::string& base_dir,
    const std::vector<std::string>& image_filenames,
    const Reconstruction& reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* camera_orientations) {
  CHECK(camera_orientations != nullptr);

  for (const std::string& image_filename : image_filenames) {
    const std::string base_name = stlplus::basename_part(image_filename);

    std::string modelview_file = base_dir + "/" + base_name + ".txt";
    if (!theia::FileExists(modelview_file)) {
      LOG(WARNING) << "File does not exist: '" << base_name << "'";
      continue;
    }

    // Read camera pose parameters.
    std::ifstream f(modelview_file);
    double values[16];
    for(int i = 0; i < 16; ++i) f >> values[i];
    f.close();

    const Eigen::Matrix4d modelview_matrix = Eigen::Matrix4d::Map(values);
    const Eigen::Matrix4d camera_pose_matrix =
        OpenGLModelviewToCameraPose(modelview_matrix);
    const Eigen::Matrix3d camera_orientation =
        GetOrientationFromCameraPose(camera_pose_matrix);

    const theia::ViewId view_id =
        reconstruction.ViewIdFromName(image_filename);
    if (view_id != theia::kInvalidViewId) {
      camera_orientations->emplace(view_id, camera_orientation);
    }
  }
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  // Load images.
  std::vector<std::string> image_files;
  CHECK(theia::GetFilepathsFromWildcard(FLAGS_images, &image_files))
  << "Could not find images that matched the filepath: " << FLAGS_images
  << ". NOTE that the ~ filepath is not supported.";
  CHECK(!image_files.empty());

  std::vector<std::string> image_filenames;
  image_filenames.reserve(image_files.size());

  for (const std::string& image_file : image_files) {
    std::string image_filename;
    CHECK(theia::GetFilenameFromFilepath(image_file, true, &image_filename));
    image_filenames.push_back(image_filename);
  }


  // Load the reconstuction.
  theia::Reconstruction reconstruction;
  CHECK(theia::ReadReconstruction(
      FLAGS_input_reconstruction_file, &reconstruction))
  << "Could not read Reconstruction file.";


  // Read other results.
  std::unordered_map<theia::ViewId, Eigen::Matrix3d> gt_camera_orientations;

  if (FLAGS_gt_camera_param_directory != "") {
    ReadOrientationsFromCameraParams(
        FLAGS_gt_camera_param_directory, image_filenames, reconstruction,
        &gt_camera_orientations);
  }
  else if (FLAGS_gt_modelview_directory != "") {
    ReadOrientationsFromModelviews(
        FLAGS_gt_modelview_directory, image_filenames, reconstruction,
        &gt_camera_orientations);
  }
  else {
    CHECK(false) << "Either 'gt_camera_param_directory' or "
        "'gt_modelview_directory' must be given.";
  }


  std::cout << "Compute global rotation from SfM result to "
      "ground truth poses..." << std::endl;

  std::vector<Eigen::Matrix3d> diff_Rs;
  diff_Rs.reserve(reconstruction.NumViews());

  for (const theia::ViewId view_id : reconstruction.ViewIds()) {
    if(gt_camera_orientations.find(view_id) == gt_camera_orientations.end()) {
      continue;
    }

    // Compare with other results.
    const theia::View* view = reconstruction.View(view_id);
    const Eigen::Matrix3d gt_R = gt_camera_orientations.at(view_id);
    const Eigen::Matrix3d est_R =
        view->Camera().GetOrientationAsRotationMatrix();
    const Eigen::Matrix3d diff_R = gt_R * est_R.transpose();
    diff_Rs.push_back(diff_R);
  }

  const Eigen::Matrix3d global_est_to_gt_R = ComputeAverageRotation(diff_Rs);


  std::cout << "Compute rotation differences..." << std::endl;

  std::vector<double> diff_angles;
  diff_angles.reserve(reconstruction.NumViews());

  for (const theia::ViewId view_id : reconstruction.ViewIds()) {
    if(gt_camera_orientations.find(view_id) == gt_camera_orientations.end()) {
      continue;
    }

    const theia::View* view = reconstruction.View(view_id);
    const Eigen::Matrix3d gt_R = gt_camera_orientations.at(view_id);
    const Eigen::Matrix3d est_R =
        view->Camera().GetOrientationAsRotationMatrix();

    // Pre-multiply global rotation.
    const Eigen::Matrix3d rotated_est_R = global_est_to_gt_R * est_R;
    const Eigen::AngleAxisd diff_R(gt_R * rotated_est_R.transpose());
    diff_angles.push_back(diff_R.angle() / M_PI * 180.0);

    // Save orientation difference.
    if (FLAGS_diff_orientation_directory != "") {
      const std::string base_name = stlplus::basename_part(view->Name());
      const std::string diff_orientation_filepath =
          FLAGS_diff_orientation_directory + "/" + base_name + ".txt";
      std::cout << "Saving '" << diff_orientation_filepath << "'... ";

      std::ofstream diff_orientation_file(diff_orientation_filepath);
      diff_orientation_file <<
      diff_R.angle() << " " <<
      diff_R.axis()[0] << " " <<
      diff_R.axis()[1] << " " <<
      diff_R.axis()[2] << std::endl;
      diff_orientation_file.close();

      std::cout << "Done." << std::endl;
    }

    // Save orientation difference camera parameters.
    const Eigen::Vector3d diff_params
        = ComputeParamsFromRotation(diff_R.toRotationMatrix());

    if (FLAGS_diff_param_directory != "") {
      const std::string base_name = stlplus::basename_part(view->Name());
      const std::string diff_param_filepath =
          FLAGS_diff_param_directory +
              "/" + base_name + ".txt";
      std::cout << "Saving '" << diff_param_filepath << "'... ";

      std::ofstream diff_param_file(diff_param_filepath);
      diff_param_file <<
      diff_params[0] << " " <<
      diff_params[1] << " " <<
      diff_params[2] << std::endl;
      diff_param_file.close();

      std::cout << "Done." << std::endl;
    }
  }

  const Eigen::VectorXd diff_angle_vec =
      Eigen::VectorXd::Map(diff_angles.data(), diff_angles.size());

  std::nth_element(diff_angles.begin(),
                   diff_angles.begin() + diff_angles.size() / 2,
                   diff_angles.end());

  std::cout << "Min:    " << diff_angle_vec.minCoeff()  << std::endl;
  std::cout << "Max:    " << diff_angle_vec.maxCoeff()  << std::endl;
  std::cout << "Mean:   " << diff_angle_vec.mean()      << std::endl;
  std::cout << "Median: " << diff_angles[diff_angles.size() / 2] << std::endl;
}
