// Copyright (C) 2016 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>

#include <fstream>
#include <iostream>
#include <stlplus3/file_system.hpp>
#include <string>


DEFINE_string(input_reconstruction_file, "",
              "Input reconstruction file in binary format.");
DEFINE_string(output_ply_file, "", "Output ply file.");
DEFINE_string(output_modelview_dir, "",
              "Output directory for modelview matrices.");



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

Eigen::Vector3d ComputeCameraParamsFromModelview(
    const Eigen::Matrix3d& modelview) {
  // R = R_z R_x R_y R_d.
  Eigen::Matrix3d default_rotation =
      Eigen::AngleAxisd(-0.5 * M_PI, Eigen::Vector3d::UnitY())
          .toRotationMatrix();

  // Post-multiply inverse(transpose) of default transformation.
  // R R_d^T = R_z R_x R_y.
  const Eigen::Matrix3d rotation = modelview * default_rotation.transpose();

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

Eigen::Matrix3d ComputeOrientationFromCameraParams(
    const Eigen::Vector3d& camera_params) {
  const Eigen::Affine3d modelview = ComputeModelviewFromCameraParams(
      camera_params);

  // Take inverse(transpose) of modelview rotation
  // to get camera pose rotation.
  const Eigen::Matrix3d camera_rotation = modelview.rotation().transpose();
  return camera_rotation;
}

Eigen::Vector3d ComputeCameraParamsFromOrientation(
    const Eigen::Matrix3d& camera_rotation) {
  // Take inverse(transpose) of camera rotation
  // to get modelview rotation.
  const Eigen::Matrix3d modelview_rotation = camera_rotation.transpose();

  const Eigen::Vector3d camera_params =
      ComputeCameraParamsFromModelview(modelview_rotation);

//  // Debug.
//  const Eigen::Affine3d test_modelview = ComputeModelviewFromCameraParams(
//      camera_params[0], camera_params[1], camera_params[2]);
//  const Eigen::Vector3i test_camera_params = ComputeCameraParamsFromModelview
//      (test_modelview.rotation());
//  CHECK_EQ(camera_params.transpose(), test_camera_params.transpose());

  return camera_params;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  // Load the reconstuction.
  theia::Reconstruction reconstruction;
  CHECK(theia::ReadReconstruction(FLAGS_input_reconstruction_file,
                                  &reconstruction))
      << "Could not read Reconstruction file.";

  // Save a ply file.
  CHECK(theia::WritePlyFile(FLAGS_output_ply_file, reconstruction, 10))
      << "Could not write NVM file.";

  // Compute fovy and save modelview matrices for each view.
  for (const theia::ViewId view_id : reconstruction.ViewIds()) {
    const theia::View* view = reconstruction.View(view_id);
    const theia::Camera& camera = view->Camera();

    const std::string basename_part =
        stlplus::basename_part(view->Name());

    // fovy.
    const int kImageHeight = 1080;

    const double f = 2.0 * camera.FocalLength() / (double) kImageHeight;
    std::cout << "f: " << f << std::endl;

    // f = 1 / tan(fovx / 2).
    // fovx = 2 * atan(1 / f).
    const double fovy = (2.0 * std::atan(1.0 / f)) / M_PI * 180.0;
    std::cout << "fovy: " << fovy << std::endl;


    // Modelview matrix.
    const Eigen::Matrix4d camera_pose_matrix = GetCameraPose(camera);
    const Eigen::Matrix4d modelview_matrix =
        CameraPoseToOpenGLModelview(camera_pose_matrix);

    const std::string modelview_filename =
        FLAGS_output_modelview_dir + "/" + basename_part + ".txt";
    std::ofstream modelview_file(modelview_filename);
    std::cout << "Saving '" << modelview_filename << "'... ";
    // Column-wise
    for(int i = 0; i < 4; ++i)
      for(int j = 0; j < 4; ++j)
        modelview_file << modelview_matrix(j, i) << std::endl;
    modelview_file.close();
    std::cout << "Done." << std::endl;
  }

  return 0;
}
