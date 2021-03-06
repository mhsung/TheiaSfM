// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <ceres/rotation.h>
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
#include "exp_bounding_box_utils.h"
#include "exp_json_utils.h"
#include "exp_matrix_utils.h"
#include "applications/exp_neural_net_output_reader.h"

#include "theia/sfm/global_pose_estimation/constrained_robust_rotation_estimator.h"
#include "theia/sfm/global_pose_estimation/constrained_nonlinear_position_estimator.h"

// Input/output files.
DEFINE_string(calibration_file, "calibration.txt",
              "Calibration file containing image calibration data.");
DEFINE_string(ground_truth_data_type, "reconstruction", "");
DEFINE_string(ground_truth_filepath, "ground_truth.bin", "");
DEFINE_string(bounding_boxes_filepath, "convnet/object_bboxes.csv", "");
DEFINE_string(orientations_filepath,
              "convnet/object_orientations_fitted.csv", "");
DEFINE_string(out_fitted_bounding_boxes_filepath, "", "");
DEFINE_string(out_fitted_orientations_filepath, "", "");
DEFINE_string(out_json_file, "", "");
DEFINE_bool(test_rotation_optimization, false, "");
DEFINE_bool(test_position_optimization, false, "");
DEFINE_string(reconstruction_for_image_list, "",
              "Optional. Use bounding boxes in the images included in the "
                  "reconstruction.");
DEFINE_string(out_top_accuracy_bounding_boxes_filepath, "", "");
DEFINE_string(out_top_accuracy_orientations_filepath, "", "");
DEFINE_double(top_accuracy_proportion, 0.1, "");


const std::vector<double> kAngleHistogramBins = {
    15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0};

void ComputeMeanMedian(
    const std::vector<double>& sorted_errors,
    double* mean_error, double* median_error) {
  CHECK_NOTNULL(mean_error);
  CHECK_NOTNULL(median_error);

  (*mean_error) = std::accumulate(
      sorted_errors.begin(), sorted_errors.end(), 0.0) /
                  static_cast<double>(sorted_errors.size());
  (*median_error) = sorted_errors[sorted_errors.size() / 2];
}

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
    "# data = %d\nMean = %lf\nMedian = %lf\nHistogram:\n%s",
    static_cast<int>(sorted_errors.size()),
    mean,
    sorted_errors[sorted_errors.size() / 2],
    histogram.PrintString().c_str());
  return error_msg;
}

template <typename T>
void Arrange(const std::vector<T>& values,
             const std::vector<size_t>& indices,
             std::vector<T>* arranged_values) {
  CHECK_EQ(values.size(), indices.size());
  CHECK_NOTNULL(arranged_values)->clear();
  arranged_values->resize(values.size());

  for (size_t i = 0; i < indices.size(); i++) {
    const size_t index = indices[i];
    CHECK_LE(index, values.size());
    (*arranged_values)[i] = values[index];
  }
}

template <typename T>
void ArgSort(const std::vector<T>& values,
             std::vector<size_t>* sorted_indices,
             std::vector<T>* sorted_values = nullptr) {
  CHECK_NOTNULL(sorted_indices)->clear();

  const size_t num_values = values.size();
  sorted_indices->resize(num_values);
  std::iota(sorted_indices->begin(), sorted_indices->end(), 0u);
  std::sort(sorted_indices->begin(), sorted_indices->end(),
            [&](int lhs, int rhs) { return values[lhs] < values[rhs]; });

  if (sorted_values != nullptr) {
    Arrange(values, *sorted_indices, sorted_values);
  }
}

template <typename Key, typename Value>
std::unordered_map<Key, Value> CreateUnorderedMap(
    const std::vector<Key>& keys,
    const std::vector<Value>& values) {
  CHECK_EQ(keys.size(), values.size());

  std::unordered_map<Key, Value> unordered_map;
  for (int i = 0; i < keys.size(); i++) {
    unordered_map.emplace(keys[i], values[i]);
  }
  return unordered_map;
}

void GetBBoxOrientations(
  const DetectedBBox& bbox,
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  Eigen::Matrix3d* world_to_camera_R = nullptr,
  Eigen::Matrix3d* object_to_camera_R = nullptr,
  Eigen::Matrix3d* world_to_object_R = nullptr) {

  const std::string basename = stlplus::basename_part(bbox.view_name_);
  const Eigen::Affine3d& modelview = FindOrDie(modelviews, basename);

  // Compute world-to-camera rotation.
  const Eigen::Matrix3d world_to_camera =
    ComputeTheiaCameraRotationFromModelview(modelview.rotation());
  if (world_to_camera_R) (*world_to_camera_R) = world_to_camera;

  // Compute object-to-camera rotation.
  const Eigen::Matrix3d object_to_camera =
    ComputeTheiaCameraRotationFromCameraParams(bbox.camera_param_);
  if (object_to_camera_R) (*object_to_camera_R) = object_to_camera;

  // Compute world-to-object rotation.
  const Eigen::Matrix3d world_to_object =
    object_to_camera.transpose() * world_to_camera;
  if (world_to_object_R) (*world_to_object_R) = world_to_object;
}

std::unique_ptr<theia::Camera> GetBBoxTheiaCamera(
  const DetectedBBox& bbox,
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>&
  camera_intrinsics_priors,
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews) {
  static const bool kSetFocalLengthFromMedianFOV = false;

  std::unique_ptr<theia::Camera> camera(new theia::Camera);

  // Find extrinsics.
  const std::string basename = stlplus::basename_part(bbox.view_name_);
  const Eigen::Affine3d& modelview = FindOrDie(modelviews, basename);
  ComputeTheiaCameraFromModelview(modelview, camera.get());

  // Find intrinsics.
  const theia::CameraIntrinsicsPrior& intrinsics =
    FindOrDie(camera_intrinsics_priors, bbox.view_name_);
  SetCameraIntrinsicsFromPriors(
    intrinsics, kSetFocalLengthFromMedianFOV, camera.get());

  return std::move(camera);
}

void CreateViewNameToIdMap(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  std::unordered_map<std::string, ViewId>* view_name_to_ids) {
  CHECK_NOTNULL(view_name_to_ids)->clear();

  const int num_cameras = modelviews.size();
  view_name_to_ids->reserve(num_cameras);
  ViewId view_id = 0;
  for (const auto& modelview : modelviews) {
    view_name_to_ids->emplace(modelview.first, view_id);
    ++view_id;
  }
}

void GetAllWorldToCameraRotations(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  const std::unordered_map<std::string, ViewId>& view_name_to_ids,
  std::unordered_map<ViewId, Eigen::Vector3d>* world_to_camera_Rs) {
  CHECK_NOTNULL(world_to_camera_Rs)->clear();

  for (const auto& modelview : modelviews) {
    const ViewId view_id = FindOrDie(view_name_to_ids, modelview.first);
    theia::Camera camera;
    ComputeTheiaCameraFromModelview(modelview.second, &camera);

    const Eigen::Matrix3d world_to_camera_R_matrix =
      camera.GetOrientationAsRotationMatrix();
    Eigen::Vector3d world_to_camera_R;
    ceres::RotationMatrixToAngleAxis(
      ceres::ColumnMajorAdapter3x3(world_to_camera_R_matrix.data()),
      world_to_camera_R.data());
    world_to_camera_Rs->emplace(view_id, world_to_camera_R);
  }
}

void GetAllWorldToObjectRotations(
  const std::unordered_map<uint32_t, Eigen::Matrix3d>& object_orientations,
  std::unordered_map<ObjectId, Eigen::Vector3d>* world_to_object_Rs) {
  CHECK_NOTNULL(world_to_object_Rs)->clear();

  for (const auto& object : object_orientations) {
    const ObjectId object_id = object.first;

    const Eigen::Matrix3d world_to_object_R_matrix = object.second;
    Eigen::Vector3d world_to_object_R;
    ceres::RotationMatrixToAngleAxis(
      ceres::ColumnMajorAdapter3x3(world_to_object_R_matrix.data()),
      world_to_object_R.data());
    world_to_object_Rs->emplace(object_id, world_to_object_R);
  }
}

void GetAllObjectToCameraRotations(
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes,
  const std::unordered_map<std::string, ViewId>& view_name_to_ids,
  std::unordered_map<ObjectId, ObjectViewOrientations>*
  object_to_camera_Rs) {
  CHECK_NOTNULL(object_to_camera_Rs)->clear();

  for (const auto& object : object_bboxes) {
    const theia::ObjectId object_id = object.first;

    for (const auto& bbox : object.second) {
      const std::string basename = stlplus::basename_part(bbox->view_name_);
      const ViewId view_id = FindOrDie(view_name_to_ids, basename);

      // Compute object-to-camera rotation.
      const Eigen::Matrix3d object_to_camera_R_matrix =
        ComputeTheiaCameraRotationFromCameraParams(bbox->camera_param_);
      Eigen::Vector3d object_to_camera_R;
      ceres::RotationMatrixToAngleAxis(
        ceres::ColumnMajorAdapter3x3(object_to_camera_R_matrix.data()),
        object_to_camera_R.data());

      (*object_to_camera_Rs)[object_id].emplace(view_id, object_to_camera_R);
    }
  }
}

void GetAllTwoViewInfos(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  const std::unordered_map<std::string, ViewId>& view_name_to_ids,
  std::unordered_map<ViewIdPair, TwoViewInfo>* camera_pairs) {
  CHECK_NOTNULL(camera_pairs)->clear();

  for (const auto& modelview1 : modelviews) {
    const ViewId view1_id = FindOrDie(view_name_to_ids, modelview1.first);

    theia::Camera camera1;
    ComputeTheiaCameraFromModelview(modelview1.second, &camera1);
    const Eigen::Matrix3d R1 = camera1.GetOrientationAsRotationMatrix();
    const Eigen::Vector3d t1 = camera1.GetPosition();

    for (const auto& modelview2 : modelviews) {
      const ViewId view2_id = FindOrDie(view_name_to_ids, modelview2.first);
      if (view1_id >= view2_id) continue;

      theia::Camera camera2;
      ComputeTheiaCameraFromModelview(modelview2.second, &camera2);
      const Eigen::Matrix3d R2 = camera2.GetOrientationAsRotationMatrix();
      const Eigen::Vector3d t2 = camera2.GetPosition();

      // Compute relative transformation.
      const Eigen::Matrix3d R12 = R2 * R1.transpose();
      const Eigen::Vector3d t12 = R1 * (t2 - t1);

      TwoViewInfo two_view_info;
      ceres::RotationMatrixToAngleAxis(
        ceres::ColumnMajorAdapter3x3(R12.data()),
        two_view_info.rotation_2.data());
      two_view_info.position_2 = t12;
      camera_pairs->emplace(std::make_pair(view1_id, view2_id), two_view_info);
    }
  }
}

void GetAllWorldToCameraPositions(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  const std::unordered_map<std::string, ViewId>& view_name_to_ids,
  std::unordered_map<ViewId, Eigen::Vector3d>* world_to_camera_ts) {
  CHECK_NOTNULL(world_to_camera_ts)->clear();

  for (const auto& modelview : modelviews) {
    const ViewId view_id = FindOrDie(view_name_to_ids, modelview.first);
    theia::Camera camera;
    ComputeTheiaCameraFromModelview(modelview.second, &camera);

    const Eigen::Vector3d world_to_camera_t = camera.GetPosition();
    world_to_camera_ts->emplace(view_id, world_to_camera_t);
  }
}

void GetAllCameraToObjectPositionDirections(
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>&
  camera_intrinsics_priors,
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes,
  const std::unordered_map<std::string, ViewId>& view_name_to_ids,
  std::unordered_map<ObjectId, ViewObjectPositionDirections>*
  camera_coord_camera_to_object_t_dirs) {
  CHECK_NOTNULL(camera_coord_camera_to_object_t_dirs)->clear();

  for (const auto& object : object_bboxes) {
    const theia::ObjectId object_id = object.first;

    for (const auto& bbox : object.second) {
      const std::string basename = stlplus::basename_part(bbox->view_name_);
      const ViewId view_id = FindOrDie(view_name_to_ids, basename);

      // Compute camera-to-object position direction.
      const theia::CameraIntrinsicsPrior& intrinsics =
        FindOrDie(camera_intrinsics_priors, bbox->view_name_);
      const Eigen::Vector3d camera_to_object_t_dir =
        ComputeCameraToObjectDirections(bbox->bbox_, intrinsics);

      (*camera_coord_camera_to_object_t_dirs)[object_id].emplace(
        view_id, camera_to_object_t_dir);
    }
  }
}

void EvaluateRotations(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes,
  std::vector<double>* sorted_errors,
  std::vector<uint32_t>* sorted_bbox_ids,
  std::unordered_map<ObjectId, Eigen::Matrix3d>* object_orientations = nullptr,
  JsonFile* out_file = nullptr) {
  CHECK_NOTNULL(object_bboxes);
  CHECK_NOTNULL(sorted_errors);
  // @object_bboxes will be updated.
  // FIXME: Make a copy of @object_bboxes and update it.

  int all_num_bboxes = 0;
  for (const auto& object : *object_bboxes) {
    all_num_bboxes += object.second.size();
  }
  std::vector<double> all_angle_errors;
  std::vector<uint32_t> all_bbox_ids;
  all_angle_errors.reserve(all_num_bboxes);
  all_bbox_ids.reserve(all_num_bboxes);

  for (const auto& object : *object_bboxes) {
    const theia::ObjectId object_id = object.first;
    VLOG(1) << "Object ID: " << object_id << std::endl;

    const int num_bboxes = object.second.size();
    std::vector<Eigen::Matrix3d> world_to_object_Rs;
    world_to_object_Rs.reserve(num_bboxes);

    // Collect world-to-object rotations.
    for (const auto& bbox : object.second) {
      Eigen::Matrix3d world_to_object_R;
      GetBBoxOrientations(*bbox, modelviews, nullptr, nullptr,
                          &world_to_object_R);
      world_to_object_Rs.push_back(world_to_object_R);
    }

    // Compute average world-to-object rotation.
    const Eigen::Matrix3d avg_world_to_object_R =
      ComputeAverageRotation(world_to_object_Rs);

    // Compute errors.
    std::vector<double> object_angle_errors;
    object_angle_errors.reserve(num_bboxes);

    for (const auto& bbox : object.second) {
      Eigen::Matrix3d world_to_camera_R;
      Eigen::Matrix3d object_to_camera_R;
      GetBBoxOrientations(*bbox, modelviews, &world_to_camera_R,
                          &object_to_camera_R, nullptr);

      // Compute fitted object-to-camera rotation.
      const Eigen::Matrix3d fitted_object_to_camera_R =
        world_to_camera_R * avg_world_to_object_R.transpose();

      // Compute angle error.
      const Eigen::AngleAxisd diff_R(
        fitted_object_to_camera_R.transpose() * object_to_camera_R);
      const double angle = diff_R.angle() / M_PI * 180.0;
      object_angle_errors.push_back(angle);
      all_angle_errors.push_back(angle);
      all_bbox_ids.push_back(bbox->bbox_id_);

      // Update camera parameters.
      // FIXME: Make a copy of @object_bboxes and update it.
      const Eigen::Vector3d fitted_camera_param =
        ComputeCameraParamsFromTheiaCameraRotation(fitted_object_to_camera_R);
      bbox->camera_param_ = fitted_camera_param;
    }

    if (object_orientations) {
      object_orientations->emplace(object_id, avg_world_to_object_R);
    }

    std::sort(object_angle_errors.begin(), object_angle_errors.end());
    const std::string angle_error_msg =
      PrintMeanMedianHistogram(object_angle_errors, kAngleHistogramBins);
    VLOG(1) << "Object orientation angle errors: \n" << angle_error_msg;
  }


  //std::sort(all_angle_errors.begin(), all_angle_errors.end());
  std::vector<size_t> sorted_indices;
  ArgSort(all_angle_errors, &sorted_indices, sorted_errors);
  Arrange(all_bbox_ids, sorted_indices, sorted_bbox_ids);

  const std::string angle_error_msg =
      PrintMeanMedianHistogram(*sorted_errors, kAngleHistogramBins);
  LOG(INFO) << "All orientation angle errors: \n" << angle_error_msg;

  if (out_file && out_file->IsOpen()) {
    double all_angle_mean_error, all_angle_median_error;
    ComputeMeanMedian(
        *sorted_errors, &all_angle_mean_error, &all_angle_median_error);
    out_file->WriteElement("mean_convnet_rotation_error", all_angle_mean_error);
    out_file->WriteElement(
        "median_convnet_rotation_error", all_angle_median_error);
  }
}

void EvaluatePositions(
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>&
  camera_intrinsics_priors,
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes,
  std::vector<double>* sorted_errors,
  std::vector<uint32_t>* sorted_bbox_ids,
  std::unordered_map<ObjectId, Eigen::Vector3d>* object_positions = nullptr,
  std::unordered_map<uint32_t, double>* camera_to_object_distances = nullptr,
  JsonFile* out_file = nullptr) {
  CHECK_NOTNULL(object_bboxes);
  CHECK_NOTNULL(sorted_errors);
  // @object_bboxes will be updated.
  // FIXME: Make a copy of @object_bboxes and update it.

  int all_num_bboxes = 0;
  for (const auto& object : *object_bboxes) {
    all_num_bboxes += object.second.size();
  }
  std::vector<double> all_angle_errors;
  std::vector<uint32_t> all_bbox_ids;
  all_angle_errors.reserve(all_num_bboxes);
  all_bbox_ids.reserve(all_num_bboxes);

  for (const auto& object : *object_bboxes) {
    const theia::ObjectId object_id = object.first;
    VLOG(1) << "Object ID: " << object_id << std::endl;

    const int num_bboxes = object.second.size();
    std::vector<Eigen::Matrix3d> world_to_object_R_list;
    world_to_object_R_list.reserve(num_bboxes);

    std::vector<Matrix3x4d> poses;
    std::vector<Eigen::Vector2d> points;
    poses.reserve(num_bboxes);
    points.reserve(num_bboxes);

    // Collect rays.
    for (const auto& bbox : object.second) {
      Matrix3x4d pmatrix;
      const std::unique_ptr<theia::Camera> camera =
        GetBBoxTheiaCamera(*bbox, camera_intrinsics_priors, modelviews);
      camera->GetProjectionMatrix(&pmatrix);
      poses.push_back(pmatrix);
      points.push_back(bbox->bbox_center());
    }

    // Compute object center in 3D.
    Eigen::Vector4d triangulated_point;
    CHECK(theia::TriangulateNView(poses, points, &triangulated_point));
    const Eigen::Vector3d object_position = triangulated_point.hnormalized();

    // Compute errors.
    std::vector<double> object_angle_errors;
    object_angle_errors.reserve(num_bboxes);

    for (const auto& bbox : object.second) {
      // Compute fitted bounding box center on image.
      Eigen::Vector2d fitted_bbox_center;
      const std::unique_ptr<theia::Camera> camera =
        GetBBoxTheiaCamera(*bbox, camera_intrinsics_priors, modelviews);
      camera->ProjectPoint(object_position.homogeneous(), &fitted_bbox_center);

      const Eigen::Vector3d fitted_ray =
        camera->PixelToUnitDepthRay(fitted_bbox_center).normalized();
      const Eigen::Vector3d test_fitted_ray =
        (object_position - camera->GetPosition()).normalized();
      const double kErrorTol = 1.0 - 1.0E-4;
      CHECK_GT(std::abs(fitted_ray.dot(test_fitted_ray)), kErrorTol);

      /*
      // NOTE:
      // Remove the bounding box if the object is behind the camera.
      if (fitted_ray.dot(test_fitted_ray) < 0) {
        VLOG(3) << "Remove the bounding box since the object is behind "
                << "the camera : ("
                << "object ID = " << object_id << ", "
                << "view = " << bbox->view_name_ << ")";
        bbox_it = object.second.erase(bbox_it);
        continue;
      }
      */

      // Compute angle error.
      const Eigen::Vector3d ray =
        camera->PixelToUnitDepthRay(bbox->bbox_center()).normalized();
      const double angle =
        std::abs(std::acos(ray.dot(fitted_ray))) / M_PI * 180.0;
      object_angle_errors.push_back(angle);
      all_angle_errors.push_back(angle);
      all_bbox_ids.push_back(bbox->bbox_id_);

      // Update camera parameters.
      // FIXME: Make a copy of @object_bboxes and update it.
      const double half_size_x = 0.5 * (bbox->bbox_[2] - bbox->bbox_[0]);
      CHECK_GE(half_size_x, 0.0);
      const double half_size_y = 0.5 * (bbox->bbox_[3] - bbox->bbox_[1]);
      CHECK_GE(half_size_y, 0.0);

      Eigen::Vector4d fitted_bbox(
        fitted_bbox_center[0] - half_size_x,
        fitted_bbox_center[1] - half_size_y,
        fitted_bbox_center[0] + half_size_x,
        fitted_bbox_center[1] + half_size_y);
      bbox->bbox_ = fitted_bbox;

      // Compute camera to object distance.
      if (camera_to_object_distances) {
        const double distance =
          (object_position - camera->GetPosition()).norm();
        camera_to_object_distances->emplace(bbox->bbox_id_, distance);
      }
    }

    if (object_positions) {
      object_positions->emplace(object_id, object_position);
    }

    std::sort(object_angle_errors.begin(), object_angle_errors.end());
    const std::string angle_error_msg =
      PrintMeanMedianHistogram(object_angle_errors, kAngleHistogramBins);
    VLOG(1) << "Object position ray angle errors: \n" << angle_error_msg;
  }


  //std::sort(all_angle_errors.begin(), all_angle_errors.end());
  std::vector<size_t> sorted_indices;
  ArgSort(all_angle_errors, &sorted_indices, sorted_errors);
  Arrange(all_bbox_ids, sorted_indices, sorted_bbox_ids);

  const std::string angle_error_msg =
      PrintMeanMedianHistogram(*sorted_errors, kAngleHistogramBins);
  LOG(INFO) << "All position angle errors: \n" << angle_error_msg;

  if (out_file && out_file->IsOpen()) {
    double all_angle_mean_error, all_angle_median_error;
    ComputeMeanMedian(
        *sorted_errors, &all_angle_mean_error, &all_angle_median_error);
    out_file->WriteElement("mean_convnet_position_error", all_angle_mean_error);
    out_file->WriteElement(
        "median_convnet_position_error", all_angle_median_error);
  }
}

bool TestRotationOptimization(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  const std::unordered_map<ObjectId, Eigen::Matrix3d>&
  object_orientations,
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes) {

  // Map view names to view IDs.
  std::unordered_map<std::string, ViewId> view_name_to_ids;
  CreateViewNameToIdMap(modelviews, &view_name_to_ids);

  // Compute camera orientations.
  std::unordered_map<ViewId, Eigen::Vector3d> world_to_camera_Rs;
  GetAllWorldToCameraRotations(
    modelviews, view_name_to_ids, &world_to_camera_Rs);

  // Compute object orientations.
  std::unordered_map<ObjectId, Eigen::Vector3d> world_to_object_Rs;
  GetAllWorldToObjectRotations(object_orientations, &world_to_object_Rs);

  // Compute object-to-camera-orientations.
  std::unordered_map<ObjectId, ObjectViewOrientations>
    object_to_camera_Rs;
  GetAllObjectToCameraRotations(
    object_bboxes, view_name_to_ids, &object_to_camera_Rs);

  // Compute camera pair relative orientations.
  std::unordered_map<ViewIdPair, TwoViewInfo> camera_pairs;
  GetAllTwoViewInfos(modelviews, view_name_to_ids, &camera_pairs);

  // Run optimization.
  const double kWeight = 1.0E4;
  RobustRotationEstimator::Options robust_rotation_estimator_options;
  std::unique_ptr<ConstrainedRobustRotationEstimator>
    constrained_rotation_estimator(new ConstrainedRobustRotationEstimator(
    robust_rotation_estimator_options, kWeight));

  // Keep ground truth.
  const std::unordered_map<ViewId, Eigen::Vector3d> gt_world_to_object_Rs =
    world_to_object_Rs;

  // 'world_to_camera_Rs' and 'world_to_object_Rs' are updated.
  // Use default weight.
  const bool ret = constrained_rotation_estimator->EstimateRotations(
    camera_pairs, object_to_camera_Rs,
    &world_to_camera_Rs, &world_to_object_Rs, nullptr);
  if (!ret) return false;

  // Check object orientation differences.
  for (const auto& object : world_to_object_Rs) {
    const ObjectId object_id = object.first;
    const auto& gt_world_to_object_R =
      FindOrDie(gt_world_to_object_Rs, object_id);
    const auto& pred_world_to_object_R = object.second;

    const Eigen::IOFormat csv_format(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ",");
    LOG(INFO) << "Object ID: " << object_id;
    LOG(INFO) << " - Ground truth orientation: "
              << gt_world_to_object_R.transpose().format(csv_format);
    LOG(INFO) << " - Predicted orientation:    "
              << pred_world_to_object_R.transpose().format(csv_format);
  }

  return true;
}

bool TestPositionOptimization(
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>&
  camera_intrinsics_priors,
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  const std::unordered_map<ObjectId, Eigen::Vector3d>&
  object_positions,
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes) {

  // Map view names to view IDs.
  std::unordered_map<std::string, ViewId> view_name_to_ids;
  CreateViewNameToIdMap(modelviews, &view_name_to_ids);

  // Compute camera orientations.
  std::unordered_map<ViewId, Eigen::Vector3d> world_to_camera_Rs;
  GetAllWorldToCameraRotations(
    modelviews, view_name_to_ids, &world_to_camera_Rs);

  // Compute camera positions.
  std::unordered_map<ViewId, Eigen::Vector3d> world_to_camera_ts;
  GetAllWorldToCameraPositions(
    modelviews, view_name_to_ids, &world_to_camera_ts);

  // Copy object positions.
  std::unordered_map<ObjectId, Eigen::Vector3d> world_to_object_ts =
    object_positions;

  // Compute object-to-camera-orientations.
  std::unordered_map<ObjectId, ViewObjectPositionDirections>
    camera_coord_camera_to_object_t_dirs;
  GetAllCameraToObjectPositionDirections(
    camera_intrinsics_priors, object_bboxes, view_name_to_ids,
    &camera_coord_camera_to_object_t_dirs);

  // Compute camera pair relative orientations.
  std::unordered_map<ViewIdPair, TwoViewInfo> camera_pairs;
  GetAllTwoViewInfos(modelviews, view_name_to_ids, &camera_pairs);

  // Run optimization.
  const double kWeight = 1.0E4;
  Reconstruction reconstruction;
  NonlinearPositionEstimator::Options nonlinear_position_estimator_options;
  // No feature track is used.
  nonlinear_position_estimator_options.min_num_points_per_view = 0;
  std::unique_ptr<ConstrainedNonlinearPositionEstimator>
    constrained_position_estimator(new ConstrainedNonlinearPositionEstimator(
      nonlinear_position_estimator_options, reconstruction, kWeight));

  // Keep ground truth.
  const std::unordered_map<ViewId, Eigen::Vector3d> gt_world_to_object_ts =
    world_to_object_ts;

  // 'world_to_camera_ts' and 'world_to_object_ts' are updated.
  // Use default weight.
  const bool kRandomlyInitialize = false;
  const bool ret = constrained_position_estimator->EstimatePositions(
    camera_pairs, world_to_camera_Rs, camera_coord_camera_to_object_t_dirs,
    &world_to_camera_ts, &world_to_object_ts, nullptr, kRandomlyInitialize);
  if (!ret) return false;

  // Check object position differences.
  for (const auto& object : world_to_object_ts) {
    const ObjectId object_id = object.first;
    const auto& gt_world_to_object_t =
      FindOrDie(gt_world_to_object_ts, object_id);
    const auto& pred_world_to_object_t = object.second;

    const Eigen::IOFormat csv_format(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ",");
    LOG(INFO) << "Object ID: " << object_id;
    LOG(INFO) << " - Ground truth position: "
              << gt_world_to_object_t.transpose().format(csv_format);
    LOG(INFO) << " - Predicted position:    "
              << pred_world_to_object_t.transpose().format(csv_format);
  }

  // Check camera-to-object direction differences.
  std::vector<double> angle_errors;

  for (const auto& object : camera_coord_camera_to_object_t_dirs) {
    const ObjectId object_id = object.first;
    for (const auto& view : object.second) {
      const ViewId view_id = view.first;

      Eigen::Matrix3d camera_R;
      ceres::AngleAxisToRotationMatrix(
        FindOrDie(world_to_camera_Rs, view_id).data(),
        ceres::ColumnMajorAdapter3x3(camera_R.data()));

      const Eigen::Vector3d gt_camera_to_object_t_dir =
        camera_R.transpose() * view.second.normalized();

      const Eigen::Vector3d camera_to_object_t_dir =
        (FindOrDie(world_to_object_ts, object_id) -
          FindOrDie(world_to_camera_ts, view_id)).normalized();

      double dot_prod = gt_camera_to_object_t_dir.dot(camera_to_object_t_dir);
      dot_prod = std::min(dot_prod, +1.0);
      dot_prod = std::max(dot_prod, -1.0);
      const double angle_error = std::acos(dot_prod) / M_PI * 180.0;
      angle_errors.push_back(angle_error);
    }
  }

  std::sort(angle_errors.begin(), angle_errors.end());
  std::vector<double> histogram_bins = {
    15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0};
  const std::string angle_error_msg =
    PrintMeanMedianHistogram(angle_errors, histogram_bins);
  LOG(INFO) << "Position ray angle errors: \n" << angle_error_msg;

  return true;
}

void RemoveBBoxesWithNoCameraModelview(
    const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
    std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes) {
  CHECK_NOTNULL(object_bboxes);

  uint32_t new_bbox_id = 0;

  for (auto object_it = object_bboxes->begin();
       object_it != object_bboxes->end(); ) {
    auto& object = *object_it;

    for (auto bbox_it = object.second.begin();
         bbox_it != object.second.end(); ) {
      const auto& bbox = *bbox_it;
      const std::string basename = stlplus::basename_part(bbox->view_name_);
      if (!ContainsKey(modelviews, basename)) {
        LOG(WARNING) << "View " << basename << " does not exist.";
        bbox_it = object.second.erase(bbox_it);
      } else {
        ++bbox_it;
      }
    }

    if (object.second.empty()) {
      object_it = object_bboxes->erase(object_it);
    } else {
      // Reassign bounding box IDs.
      for (auto& bbox : object.second) {
        bbox->bbox_id_ = new_bbox_id;
        ++new_bbox_id;
      }
      ++object_it;
    }
  }
}

void RemoveObjectsWithNoBBoxes(
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes,
  std::unordered_map<ObjectId, Eigen::Matrix3d>* object_orientations = nullptr,
  std::unordered_map<ObjectId, Eigen::Vector3d>* object_positions = nullptr) {
  CHECK_NOTNULL(object_bboxes);

  uint32_t new_bbox_id = 0;

  for (auto object_it = object_bboxes->begin();
       object_it != object_bboxes->end(); ) {
    auto& object = *object_it;
    const theia::ObjectId object_id = object.first;

    // NOTE:
    // Remove the object if no bounding remains.
    if (object.second.empty()) {
      LOG(INFO) << "Remove the object since no bounding box remains: ("
                << "object ID = " << object_id << ").";
      object_it = object_bboxes->erase(object_it);
      continue;
    }

    // Reassign bounding box IDs.
    for (auto& bbox : object.second) {
      bbox->bbox_id_ = new_bbox_id;
      ++new_bbox_id;
    }

    ++object_it;
  }

  if (object_orientations != nullptr) {
    std::unordered_map<ObjectId, Eigen::Matrix3d> temp_object_orientations;
    for (auto& object : *object_orientations) {
      if (ContainsKey(*object_bboxes, object.first)) {
        temp_object_orientations.emplace(object);
      }
    }
    object_orientations->swap(temp_object_orientations);
    temp_object_orientations.clear();
  }

  if (object_positions != nullptr) {
    std::unordered_map<ObjectId, Eigen::Vector3d> temp_object_positions;
    for (const auto& object : *object_positions) {
      if (ContainsKey(*object_bboxes, object.first)) {
        temp_object_positions.emplace(object);
      }
    }
    object_positions->swap(temp_object_positions);
    temp_object_positions.clear();
  }
}

void RemoveModelviewNotInReconstruction(
    const theia::Reconstruction& reconstruction,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews) {
  CHECK_NOTNULL(modelviews);

  std::unordered_map<std::string, Eigen::Affine3d> subset_modelviews;

  for (const auto& view_id : reconstruction.ViewIds()) {
    const View* view = reconstruction.View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }
    const std::string basename = stlplus::basename_part(view->Name());
    const auto* modelview = FindOrNull(*modelviews, basename);
    if (modelview != nullptr) {
      subset_modelviews.emplace(basename, *modelview);
    }
  }

  modelviews->swap(subset_modelviews);
}

void SaveTopAccuracyBBoxes(
    const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes,
    const std::vector<uint32_t>& sorted_rotation_bbox_ids,
    const std::vector<double>& sorted_rotation_errors,
    const std::vector<uint32_t>& sorted_position_bbox_ids,
    const std::vector<double>& sorted_position_errors) {
  CHECK_GT(FLAGS_top_accuracy_proportion, 0.0);
  CHECK_LE(FLAGS_top_accuracy_proportion, 1.0);

  const std::unordered_map<uint32_t, double> rotation_errors =
      CreateUnorderedMap(sorted_rotation_bbox_ids, sorted_rotation_errors);
  const std::unordered_map<uint32_t, double> position_errors =
      CreateUnorderedMap(sorted_position_bbox_ids, sorted_position_errors);
  const double sum_rotation_error = std::accumulate(
      sorted_rotation_errors.begin(), sorted_rotation_errors.end(), 0.0);
  const double sum_position_error = std::accumulate(
      sorted_position_errors.begin(), sorted_position_errors.end(), 0.0);
  CHECK_GT(sum_rotation_error, 0.0);
  CHECK_GT(sum_position_error, 0.0);

  // Sum two errors.
  std::vector<double> all_mixed_errors;
  std::vector<uint32_t> all_bbox_ids;
  all_mixed_errors.reserve(rotation_errors.size());
  all_bbox_ids.reserve(rotation_errors.size());

  for (const auto& bbox : rotation_errors) {
    const uint32_t bbox_id = bbox.first;
    const double rotation_error = bbox.second;
    const double* position_error = FindOrNull(position_errors, bbox_id);
    if (position_error != nullptr) {
      const double mixed_error =
          rotation_error / sum_rotation_error +
              (*position_error) / sum_position_error;
      all_mixed_errors.push_back(mixed_error);
      all_bbox_ids.push_back(bbox_id);
    }
  }

  // Sort by multiplies errors.
  std::vector<uint32_t> sorted_bbox_ids;
  std::vector<double> sorted_multiplied_errors;
  std::vector<size_t> sorted_indices;
  ArgSort(all_mixed_errors, &sorted_indices, &sorted_multiplied_errors);
  Arrange(all_bbox_ids, sorted_indices, &sorted_bbox_ids);

  // Select top accuracy bounding boxes.
  const int num_subset_bboxes = static_cast<int>(
      FLAGS_top_accuracy_proportion * sorted_bbox_ids.size());
  std::set<uint32_t> selected_bbox_ids;
  for (int i = 0; i < num_subset_bboxes; i++) {
    selected_bbox_ids.insert(sorted_bbox_ids[i]);
  }

  // Copy selected bounding boxes.
  std::unordered_map<ObjectId, DetectedBBoxPtrList> subset_object_bboxes;
  for (const auto& object : object_bboxes) {
    subset_object_bboxes.emplace(object.first, DetectedBBoxPtrList());
    for (const auto& bbox : object.second) {
      if (ContainsKey(selected_bbox_ids, bbox->bbox_id_)) {
        subset_object_bboxes[object.first].emplace_back(
            new DetectedBBox(*bbox));
      }
    }
  }

  RemoveObjectsWithNoBBoxes(&subset_object_bboxes);

  CHECK(WriteNeuralNetBBoxes(
      FLAGS_out_top_accuracy_bounding_boxes_filepath, subset_object_bboxes));

  CHECK(WriteNeuralNetOrientations(
      FLAGS_out_top_accuracy_orientations_filepath, subset_object_bboxes));
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Read calibration file.
  std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
      camera_intrinsics_priors;
  CHECK(theia::ReadCalibration(
      FLAGS_calibration_file, &camera_intrinsics_priors));

  // Read ground truth modelview matrices.
  std::unordered_map<std::string, Eigen::Affine3d> modelviews;
  CHECK(ReadModelviews(FLAGS_ground_truth_data_type,
                       FLAGS_ground_truth_filepath, &modelviews));


  // Remove modelview matrices if the view is not included in the given
  // reconstruction.
  if (FLAGS_reconstruction_for_image_list != "") {
    std::unique_ptr<theia::Reconstruction> reconstruction(
        new theia::Reconstruction());
    CHECK(ReadReconstruction(FLAGS_reconstruction_for_image_list,
                             reconstruction.get()))
    << "Could not read reconstruction file.";
    RemoveModelviewNotInReconstruction(*reconstruction, &modelviews);
  }


  // Read bounding box information.
  std::unordered_map<ObjectId, DetectedBBoxPtrList> object_bboxes;
  ReadNeuralNetBBoxesAndOrientations(FLAGS_bounding_boxes_filepath,
                                     FLAGS_orientations_filepath,
                                     &object_bboxes);

  // Remove bounding boxes if the ground truth modelview of the view does not
  // exist.
  RemoveBBoxesWithNoCameraModelview(modelviews, &object_bboxes);


  // Statistics.
  const int num_views = modelviews.size();
  const int num_objects = object_bboxes.size();
  CHECK_GT(num_objects, 0);

  std::set<std::string> view_names_with_bboxes;
  int num_bboxes = 0;
  for (const auto& object : object_bboxes) {
    num_bboxes += object.second.size();
    for (const auto& bbox : object.second) {
      view_names_with_bboxes.insert(bbox->view_name_);
    }
  }
  const int num_views_with_bboxes = view_names_with_bboxes.size();
  const double mean_num_bboxes_per_object =
      static_cast<double>(num_bboxes) / num_objects;
  const double view_proportion_with_bboxes =
      static_cast<double>(num_views_with_bboxes) / num_views;

  // Save json statistics file.
  JsonFile out_file;
  if (FLAGS_out_json_file != "") {
    CHECK(out_file.Open(FLAGS_out_json_file))
    << "Can't open file '" + FLAGS_out_json_file + "'.";
    out_file.WriteElement("num_views", num_views);
    out_file.WriteElement("num_objects", num_objects);
    out_file.WriteElement("num_bboxes", num_bboxes);
    out_file.WriteElement(
        "mean_num_bboxes_per_object", mean_num_bboxes_per_object);
    out_file.WriteElement(
        "view_proportion_with_bboxes", view_proportion_with_bboxes);
  }


  // Copy object bounding boxes.
  std::unordered_map<ObjectId, DetectedBBoxPtrList> fitted_object_bboxes;
  for (const auto& object : object_bboxes) {
    fitted_object_bboxes.emplace(object.first, DetectedBBoxPtrList());
    for (const auto& bbox : object.second) {
      fitted_object_bboxes[object.first].emplace_back(new DetectedBBox(*bbox));
    }
  }


  LOG(INFO) << "== Evaluate Rotations ==";
  std::vector<double> sorted_rotation_errors;
  std::vector<uint32_t> sorted_rotation_bbox_ids;
  std::unordered_map<ObjectId, Eigen::Matrix3d> object_orientations;
  EvaluateRotations(modelviews, &fitted_object_bboxes,
                    &sorted_rotation_errors, &sorted_rotation_bbox_ids,
                    &object_orientations, &out_file);

  LOG(INFO) << "== Evaluate Positions ==";
  std::vector<double> sorted_position_errors;
  std::vector<uint32_t> sorted_position_bbox_ids;
  std::unordered_map<ObjectId, Eigen::Vector3d> object_positions;
  std::unordered_map<uint32_t, double> camera_to_object_distances;
  EvaluatePositions(camera_intrinsics_priors, modelviews, &fitted_object_bboxes,
                    &sorted_position_errors, &sorted_position_bbox_ids,
                    &object_positions, &camera_to_object_distances, &out_file);

  out_file.Close();

  if (FLAGS_out_top_accuracy_bounding_boxes_filepath != "" &&
      FLAGS_out_top_accuracy_orientations_filepath != "" &&
      FLAGS_top_accuracy_proportion > 0.0) {
    SaveTopAccuracyBBoxes(object_bboxes,
                          sorted_rotation_bbox_ids, sorted_rotation_errors,
                          sorted_position_bbox_ids, sorted_position_errors);
  }

  // NOTE:
  // Object are removed in position evaluation if they are located behind the
  // cameras. Bounding box IDs are re-assigned.
  // RemoveObjectsWithNoBBoxes(
  //     &object_bboxes, &object_orientations, &object_positions);


  if (FLAGS_test_rotation_optimization) {
    LOG(INFO) << "== Test Rotation Optimization ==";
    CHECK(TestRotationOptimization(
        modelviews, object_orientations, fitted_object_bboxes));
  }

  if (FLAGS_test_position_optimization) {
    LOG(INFO) << "== Test Position Optimization ==";
    CHECK(TestPositionOptimization(
        camera_intrinsics_priors, modelviews, object_positions,
        fitted_object_bboxes));
  }


  if (FLAGS_out_fitted_bounding_boxes_filepath != "") {
    CHECK(WriteNeuralNetBBoxes(
        FLAGS_out_fitted_bounding_boxes_filepath, fitted_object_bboxes));
  }

  if (FLAGS_out_fitted_orientations_filepath != "") {
    CHECK(WriteNeuralNetOrientations(
        FLAGS_out_fitted_orientations_filepath, fitted_object_bboxes));
  }

  return 0;
}
