// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <chrono>  // NOLINT
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "applications/command_line_helpers.h"
#include "applications/exp_camera_param_io.h"
#include "applications/exp_camera_param_utils.h"
#include "applications/exp_neural_net_output_reader.h"

// Input/output files.
DEFINE_string(calibration_file,
              "/Users/msung/Developer/data/ICL-NUIM/lr_kt2/calibration.txt",
              "Calibration file containing image calibration data.");
DEFINE_string(ground_truth_data_type, "reconstruction", "");
DEFINE_string(ground_truth_filepath,
              "/Users/msung/Developer/data/ICL-NUIM/lr_kt2/ground_truth.bin",
              "");
DEFINE_string(bounding_boxes_filepath,
              "/Users/msung/Developer/data/ICL-NUIM/lr_kt2/convnet"
                "/object_bboxes.csv", "");
DEFINE_string(orientations_filepath,
              "/Users/msung/Developer/data/ICL-NUIM/lr_kt2/convnet"
                "/object_orientations_fitted.csv", "");
DEFINE_string(out_fitted_bounding_boxes_filepath,
              "/Users/msung/Developer/data/ICL-NUIM/lr_kt2/convnet"
                "/object_bboxes_test.csv", "");
DEFINE_string(out_fitted_orientations_filepath,
              "/Users/msung/Developer/data/ICL-NUIM/lr_kt2/convnet"
                "/object_orientations_test.csv", "");


std::string PrintMeanMedianHistogram(
  const std::vector<double>& sorted_errors,
  const std::vector<double>& histogram_bins) {
  double mean = 0;
  theia::Histogram<double> histogram(histogram_bins);
  for (const auto& error : sorted_errors) {
    histogram.Add(error);
    mean += error;
  }

  mean /= static_cast<double>(sorted_errors.size());
  const std::string error_msg = theia::StringPrintf(
    "Mean = %lf\nMedian = %lf\nHistogram:\n%s",
    mean,
    sorted_errors[sorted_errors.size() / 2],
    histogram.PrintString().c_str());
  return error_msg;
}

void EvaluateRotations(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes) {
  CHECK_NOTNULL(object_bboxes);
  // @object_bboxes will be updated.
  // FIXME: Make a copy of @object_bboxes and update it.

  for (const auto& object : *object_bboxes) {
    const theia::ObjectId object_id = object.first;
    LOG(INFO) << "Object ID: " << object_id << std::endl;

    const int num_bboxes = object.second.size();
    std::vector<Eigen::Matrix3d> world_to_object_R_list;
    world_to_object_R_list.reserve(num_bboxes);

    // Collect world-to-object rotations.
    for (const auto& bbox : object.second) {
      const std::string basename = stlplus::basename_part(bbox->view_name_);
      const Eigen::Affine3d& modelview = FindOrDie(modelviews, basename);

      // Compute world-to-camera rotation.
      const Eigen::Matrix3d world_to_camera_R =
        ComputeTheiaCameraRotationFromModelview(modelview.rotation());

      // Compute object-to-camera rotation.
      const Eigen::Matrix3d object_to_camera_R =
        ComputeTheiaCameraRotationFromCameraParams(bbox->camera_param_);

      // Compute world-to-object rotation.
      const Eigen::Matrix3d world_to_object_R =
        object_to_camera_R.transpose() * world_to_camera_R;
      world_to_object_R_list.push_back(world_to_object_R);
    }

    // Compute average world-to-object rotation.
    const Eigen::Matrix3d avg_world_to_object_R =
      ComputeAverageRotation(world_to_object_R_list);

    // Compute errors.
    std::vector<double> angle_errors(num_bboxes);
    for (auto& bbox : object.second) {
      const std::string basename = stlplus::basename_part(bbox->view_name_);
      const Eigen::Affine3d& modelview = FindOrDie(modelviews, basename);

      // Compute world-to-camera rotation.
      const Eigen::Matrix3d world_to_camera_R =
        ComputeTheiaCameraRotationFromModelview(modelview.rotation());

      // Compute object-to-camera rotation.
      const Eigen::Matrix3d object_to_camera_R =
        ComputeTheiaCameraRotationFromCameraParams(bbox->camera_param_);

      // Compute fitted object-to-camera rotation.
      const Eigen::Matrix3d fitted_object_to_camera_R =
        world_to_camera_R * avg_world_to_object_R.transpose();

      // Compute angle error.
      const Eigen::AngleAxisd diff_R(
        fitted_object_to_camera_R.transpose() * object_to_camera_R);
      const double angle = diff_R.angle() / M_PI * 180.0;
      angle_errors.push_back(angle);

      // Update camera parameters.
      // FIXME: Make a copy of @object_bboxes and update it.
      const Eigen::Vector3d fitted_camera_param =
        ComputeCameraParamsFromTheiaCameraRotation(fitted_object_to_camera_R);
      bbox->camera_param_ = fitted_camera_param;
    }

    std::sort(angle_errors.begin(), angle_errors.end());
    std::vector<double> histogram_bins = {
      15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0};
    const std::string angle_error_msg =
      PrintMeanMedianHistogram(angle_errors, histogram_bins);
    LOG(INFO) << "Orientation angle errors: \n" << angle_error_msg;
  }
}

void EvaluatePositions(
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>&
  camera_intrinsics_prior,
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes) {
  CHECK_NOTNULL(object_bboxes);
  // @object_bboxes will be updated.
  // FIXME: Make a copy of @object_bboxes and update it.

  static const bool kSetFocalLengthFromMedianFOV = false;

  for (const auto& object : *object_bboxes) {
    const theia::ObjectId object_id = object.first;
    LOG(INFO) << "Object ID: " << object_id << std::endl;

    const int num_bboxes = object.second.size();
    std::vector<Eigen::Matrix3d> world_to_object_R_list;
    world_to_object_R_list.reserve(num_bboxes);

    std::vector<Matrix3x4d> poses;
    std::vector<Eigen::Vector2d> points;
    poses.reserve(num_bboxes);
    points.reserve(num_bboxes);

    // Collect rays.
    for (const auto& bbox : object.second) {
      // Find extrinsics.
      const std::string basename = stlplus::basename_part(bbox->view_name_);
      const Eigen::Affine3d& modelview = FindOrDie(modelviews, basename);
      theia::Camera camera;
      ComputeTheiaCameraFromModelview(modelview, &camera);

      // Find intrinsics.
      const theia::CameraIntrinsicsPrior& intrinsics =
        FindOrDie(camera_intrinsics_prior, bbox->view_name_);
      SetCameraIntrinsicsFromPriors(
        intrinsics, kSetFocalLengthFromMedianFOV, &camera);

      // Compute projection matrix.
      Matrix3x4d pmatrix;
      camera.GetProjectionMatrix(&pmatrix);
      poses.push_back(pmatrix);

      // Compute bounding box center on image.
      Eigen::Vector2d bbox_center(
        0.5 * (bbox->bbox_[0] + bbox->bbox_[2]),
        0.5 * (bbox->bbox_[1] + bbox->bbox_[3]));

      points.push_back(bbox_center);
    }

    // Compute object center in 3D.
    Eigen::Vector4d triangulated_point;
    CHECK(theia::TriangulateNView(poses, points, &triangulated_point));

    // Compute errors.
    std::vector<double> angle_errors(num_bboxes);
    for (const auto& bbox : object.second) {
      // Find extrinsics.
      const std::string basename = stlplus::basename_part(bbox->view_name_);
      const Eigen::Affine3d& modelview = FindOrDie(modelviews, basename);
      theia::Camera camera;
      ComputeTheiaCameraFromModelview(modelview, &camera);

      // Find intrinsics.
      const theia::CameraIntrinsicsPrior& intrinsics =
        FindOrDie(camera_intrinsics_prior, bbox->view_name_);
      SetCameraIntrinsicsFromPriors(
        intrinsics, kSetFocalLengthFromMedianFOV, &camera);

      // Compute bounding box center on image.
      Eigen::Vector2d bbox_center(
        0.5 * (bbox->bbox_[0] + bbox->bbox_[2]),
        0.5 * (bbox->bbox_[1] + bbox->bbox_[3]));

      // Compute fitted bounding box center on image.
      Eigen::Vector2d fitted_bbox_center;
      camera.ProjectPoint(triangulated_point, &fitted_bbox_center);

      // Compute angle error.
      const Eigen::Vector3d ray =
        camera.PixelToUnitDepthRay(bbox_center).normalized();
      const Eigen::Vector3d fitted_ray =
        camera.PixelToUnitDepthRay(fitted_bbox_center).normalized();
      const double angle =
        std::abs(std::acos(ray.dot(fitted_ray))) / M_PI * 180.0;
      angle_errors.push_back(angle);

      // Update camera parameters.
      // FIXME: Make a copy of @object_bboxes and update it.
      const double bbox_half_size_x = 0.5 * (bbox->bbox_[2] - bbox->bbox_[0]);
      CHECK_GE(bbox_half_size_x, 0.0);
      const double bbox_half_size_y = 0.5 * (bbox->bbox_[3] - bbox->bbox_[1]);
      CHECK_GE(bbox_half_size_y, 0.0);

      Eigen::Vector4d fitted_bbox(
        fitted_bbox_center[0] - bbox_half_size_x,
        fitted_bbox_center[1] - bbox_half_size_y,
        fitted_bbox_center[0] + bbox_half_size_x,
        fitted_bbox_center[1] + bbox_half_size_y);
      bbox->bbox_ = fitted_bbox;
    }

    std::sort(angle_errors.begin(), angle_errors.end());
    std::vector<double> histogram_bins = {
      15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0};
    const std::string angle_error_msg =
      PrintMeanMedianHistogram(angle_errors, histogram_bins);
    LOG(INFO) << "Position ray angle errors: \n" << angle_error_msg;
  }
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Read calibration file.
  std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
    camera_intrinsics_prior;
  CHECK(theia::ReadCalibration(
    FLAGS_calibration_file, &camera_intrinsics_prior));

  // Read ground truth modelview matrices.
  std::unordered_map<std::string, Eigen::Affine3d> modelviews;
  CHECK(ReadModelviews(FLAGS_ground_truth_data_type,
                       FLAGS_ground_truth_filepath, &modelviews));

  // Read bounding box information.
  std::unordered_map<ObjectId, DetectedBBoxPtrList> object_bboxes;
  ReadNeuralNetBBoxesAndOrientations(FLAGS_bounding_boxes_filepath,
                                     FLAGS_orientations_filepath,
                                     &object_bboxes);

  EvaluateRotations(modelviews, &object_bboxes);

  EvaluatePositions(camera_intrinsics_prior, modelviews, &object_bboxes);

  CHECK(WriteNeuralNetBBoxes(
    FLAGS_out_fitted_bounding_boxes_filepath, object_bboxes));

  CHECK(WriteNeuralNetOrientations(
    FLAGS_out_fitted_orientations_filepath, object_bboxes));

  return 0;
}
