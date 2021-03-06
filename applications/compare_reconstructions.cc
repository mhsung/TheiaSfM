// Copyright (C) 2015 The Regents of the University of California (Regents).
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
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

// @mhsung
#include <stlplus3/file_system.hpp>
#include "exp_json_utils.h"


DEFINE_string(reference_reconstruction, "",
              "Filepath to the reconstruction that is considered the reference "
              "or 'ground truth' reconstruction.");
DEFINE_string(reconstruction_to_align, "",
              "Filepath to the reconstruction that will be aligned to the "
              "ground truth reconstruction. The reported errors/distance are "
              "the distances from this to the reference reconstruction after a "
              "robust alignment has been performed.");
DEFINE_double(robust_alignment_threshold, 0.0,
              "If greater than 0.0, this threshold sets determines inliers for "
              "RANSAC alignment of reconstructions. The inliers are then used "
              "for a least squares alignment.");
DEFINE_bool(normalize_with_common_views, true, "");

// @mhsung
DEFINE_string(out_json_file, "", "");
DEFINE_int32(start_frame, -1, "");
DEFINE_int32(end_frame, -1, "");
DEFINE_string(out_reference_csv, "", "");
DEFINE_string(out_to_align_csv, "", "");

using theia::Reconstruction;
using theia::TrackId;
using theia::ViewId;


// @mhsung
void ExtractFramesFromTwoImageFiles(
    const std::string& image1_file, const std::string& image2_file,
    int* image1_frame, int* image2_frame) {
  CHECK_NOTNULL(image1_frame);
  CHECK_NOTNULL(image2_frame);

  const std::string basename1 = stlplus::basename_part(image1_file);
  const std::string basename2 = stlplus::basename_part(image2_file);

  const std::string common_prefix(basename1.begin(), std::mismatch(
      basename1.begin(), basename1.end(), basename2.begin()).first);
  const int len_common_prefix = common_prefix.size();

  // NOTE:
  // Assume that the part after the common prefix is the frame number.
  const int len_image1_frame = basename1.size() - len_common_prefix;
  (*image1_frame) = std::stoi(
      basename1.substr(len_common_prefix, len_image1_frame));

  const int len_image2_frame = basename2.size() - len_common_prefix;
  (*image2_frame) = std::stoi(
      basename2.substr(len_common_prefix, len_image2_frame));
}

// @mhsung
void ExtractFrameIndicesFromImages(
    const std::vector<std::string>& image_files,
    std::unordered_map<int, std::string>* frame_indices) {
  const int num_images = image_files.size();
  CHECK_GE(num_images, 2);
  CHECK_NOTNULL(frame_indices);
  frame_indices->clear();

  const std::string& image1_file = image_files[0];

  for (int count = 1; count < num_images; ++count) {
    const std::string& image2_file = image_files[count];

    int image1_frame_index, image2_frame_index;
    ExtractFramesFromTwoImageFiles(
        image1_file, image2_file, &image1_frame_index, &image2_frame_index);

    if (count == 1) frame_indices->emplace(image1_frame_index, image1_file);
    frame_indices->emplace(image2_frame_index, image2_file);
  }
}

// @mhsung
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
      "Mean = %lf\nMedian = %lf\nHistogram:\n%s",
      mean,
      sorted_errors[sorted_errors.size() / 2],
      histogram.PrintString().c_str());
  return error_msg;
}

double AngularDifference(const Eigen::Vector3d& rotation1,
                         const Eigen::Vector3d& rotation2) {
  Eigen::Matrix3d rotation1_mat(
      Eigen::AngleAxisd(rotation1.norm(), rotation1.normalized()));
  Eigen::Matrix3d rotation2_mat(
      Eigen::AngleAxisd(rotation2.norm(), rotation2.normalized()));
  Eigen::Matrix3d rotation_loop = rotation1_mat.transpose() * rotation2_mat;
  const double angle_rad = Eigen::AngleAxisd(rotation_loop).angle();
  return (angle_rad / M_PI * 180.0);
}

// Aligns the orientations of the models (ignoring the positions) and reports
// the difference in orientations after alignment.
void EvaluateRotations(const Reconstruction& reference_reconstruction,
                       const Reconstruction& reconstruction_to_align,
                       const std::vector<std::string>& common_view_names,
                       // @mhsung
                       JsonFile* out_file) {
  CHECK_NOTNULL(out_file);

  // Gather all the rotations in common with both views.
  std::vector<Eigen::Vector3d> rotations1, rotations2;
  rotations1.reserve(common_view_names.size());
  rotations2.reserve(common_view_names.size());
  for (const std::string& view_name : common_view_names) {
    const ViewId view_id1 = reference_reconstruction.ViewIdFromName(view_name);
    const ViewId view_id2 = reconstruction_to_align.ViewIdFromName(view_name);
    rotations1.push_back(reference_reconstruction.View(view_id1)
                             ->Camera()
                             .GetOrientationAsAngleAxis());
    rotations2.push_back(reconstruction_to_align.View(view_id2)
                             ->Camera()
                             .GetOrientationAsAngleAxis());
  }

  // Align the rotation estimations.
  theia::AlignRotations(rotations1, &rotations2);

  // Measure the difference in rotations.
  std::vector<double> rotation_error_degrees(rotations1.size());
  for (int i = 0; i < rotations1.size(); i++) {
    rotation_error_degrees[i] = AngularDifference(rotations1[i], rotations2[i]);
  }
  std::sort(rotation_error_degrees.begin(), rotation_error_degrees.end());

  std::vector<double> histogram_bins = {1, 2, 5, 10, 15, 20, 45};
  const std::string rotation_error_msg =
      PrintMeanMedianHistogram(rotation_error_degrees, histogram_bins);
  LOG(INFO) << "Rotation difference when aligning orientations:\n"
            << rotation_error_msg;

  // @mhsung
  if (out_file->IsOpen()) {
    double mean_rotation_error = 0.0, median_rotation_error = 0.0;
    ComputeMeanMedian(rotation_error_degrees,
        &mean_rotation_error, &median_rotation_error);
    out_file->WriteElement("mean_rotation_error", mean_rotation_error);
    out_file->WriteElement("median_rotation_error", median_rotation_error);
  }
}

// Align the reconstructions then evaluate the pose errors.
void EvaluateAlignedPoseError(const std::vector<std::string>& common_view_names,
                              const Reconstruction& reference_reconstruction,
                              Reconstruction* reconstruction_to_align,
                              // @mhsung
                              JsonFile* out_file) {
  CHECK_NOTNULL(out_file);

  if (FLAGS_robust_alignment_threshold > 0.0) {
    AlignReconstructionsRobust(FLAGS_robust_alignment_threshold,
                               reference_reconstruction,
                               reconstruction_to_align);
  } else {
    AlignReconstructions(reference_reconstruction, reconstruction_to_align);
  }

  std::vector<double> rotation_bins = {1, 2, 5, 10, 15, 20, 45};
  std::vector<double> position_bins = {1, 5, 10, 50, 100, 1000 };
  theia::PoseError pose_error(rotation_bins, position_bins);
  std::vector<double> focal_length_errors(common_view_names.size());
  for (int i = 0; i < common_view_names.size(); i++) {
    const ViewId view_id1 =
        reference_reconstruction.ViewIdFromName(common_view_names[i]);
    const ViewId view_id2 =
        reconstruction_to_align->ViewIdFromName(common_view_names[i]);
    const theia::Camera& camera1 =
        reference_reconstruction.View(view_id1)->Camera();
    const theia::Camera& camera2 =
        reconstruction_to_align->View(view_id2)->Camera();

    // Rotation error.
    const double rotation_error =
        AngularDifference(camera1.GetOrientationAsAngleAxis(),
                          camera2.GetOrientationAsAngleAxis());

    // Position error.
    const double position_error =
        (camera1.GetPosition() - camera2.GetPosition()).norm();
    pose_error.AddError(rotation_error, position_error);

    // Focal length error.
    focal_length_errors[i] =
        std::abs(camera1.FocalLength() - camera2.FocalLength()) /
        camera1.FocalLength();
  }
  LOG(INFO) << "Pose error:\n" << pose_error.PrintMeanMedianHistogram();

  std::sort(focal_length_errors.begin(), focal_length_errors.end());
  std::vector<double> histogram_bins = {0.01, 0.05, 0.2, 0.5, 1, 10, 100};
  const std::string focal_length_error_msg =
      PrintMeanMedianHistogram(focal_length_errors, histogram_bins);
  LOG(INFO) << "Focal length errors: \n" << focal_length_error_msg;

  // @mhsung
  if (out_file->IsOpen()) {
    double mean_rotation_error = 0.0, median_rotation_error = 0.0;
    double mean_position_error = 0.0, median_position_error = 0.0;
    pose_error.ComputeMeanMedian(
        &mean_rotation_error, &median_rotation_error,
        &mean_position_error, &median_position_error);
    out_file->WriteElement("mean_aligned_rotation_error", mean_rotation_error);
    out_file->WriteElement("median_aligned_rotation_error",
                           median_rotation_error);
    out_file->WriteElement("mean_aligned_position_error", mean_position_error);
    out_file->WriteElement("median_aligned_position_error",
                           median_position_error);
  }
}

void ComputeTrackLengthHistogram(const Reconstruction& reconstruction) {
  std::vector<int> histogram_bins = {2, 3,  4,  5,  6,  7, 8,
                                     9, 10, 15, 20, 25, 50};
  theia::Histogram<int> histogram(histogram_bins);
  for (const TrackId track_id : reconstruction.TrackIds()) {
    const theia::Track* track = reconstruction.Track(track_id);
    histogram.Add(track->NumViews());
  }
  const std::string hist_msg = histogram.PrintString();
  LOG(INFO) << "Track lengths = \n" << hist_msg;
}

// @mhsung
bool SaveCameraPositionCSVFile(const std::vector<std::string>& view_names,
                               const Reconstruction& reconstruction,
                               const std::string& filename) {
  std::ofstream file(filename);
  if (!file.good()) return false;

  const Eigen::IOFormat csv_format(
      Eigen::FullPrecision, Eigen::DontAlignCols, ",");

  for (const auto& view_name : view_names) {
    const theia::View* view = reconstruction.View(
        reconstruction.ViewIdFromName(view_name));
    const Eigen::Vector3d camera_position = view->Camera().GetPosition();
    file << camera_position.transpose().format(csv_format) << std::endl;
  }

  file.close();
  return true;
}

// @mhsung
Eigen::Vector3d FarthestPoint(const Eigen::Vector3d& query,
                              const std::vector<Eigen::Vector3d>& points) {
  CHECK(!points.empty());
  double max_dist = 0;
  Eigen::Vector3d farthest_point = points[0];

  for (const auto& point : points) {
    const double dist = (query - point).norm();
    if (dist > max_dist) {
      max_dist = dist;
      farthest_point = point;
    }
  }

  return farthest_point;
}

// @mhsung
void NormalizeCameraPositions(
    const std::vector<std::string>& view_names,
    Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);

  std::vector<Eigen::Vector3d> points;
  points.reserve(view_names.size());
  for (const auto& view_name : view_names) {
    const theia::ViewId view_id =
        reconstruction->ViewIdFromName(view_name);
    CHECK_NE(view_id, theia::kInvalidViewId);
    const theia::View* view = CHECK_NOTNULL(reconstruction->View(view_id));
    points.push_back(view->Camera().GetPosition());
  }
  const Eigen::Vector3d farthest_1 = FarthestPoint(points[0], points);
  const Eigen::Vector3d farthest_2 = FarthestPoint(farthest_1, points);
  const Eigen::Vector3d center = 0.5 * (farthest_1 + farthest_2);
  const double size = (farthest_2 - farthest_1).norm();
  CHECK_GT(size, 1.0E-8);

  theia::TransformReconstruction(
      Eigen::Matrix3d::Identity(), -center, 1.0, reconstruction);
  theia::TransformReconstruction(
      Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), 1.0 / size,
      reconstruction);
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  std::unique_ptr<Reconstruction> reference_reconstruction(
      new Reconstruction());
  CHECK(theia::ReadReconstruction(FLAGS_reference_reconstruction,
                                  reference_reconstruction.get()))
      << "Could not read ground truth reconstruction file:"
      << FLAGS_reference_reconstruction;

  std::unique_ptr<Reconstruction> reconstruction_to_align(new Reconstruction());
  CHECK(theia::ReadReconstruction(FLAGS_reconstruction_to_align,
                                  reconstruction_to_align.get()))
      << "Could not read reconstruction file:" << FLAGS_reconstruction_to_align;

  // @mhsung
  // Use a subset of cameras if the frame range is given.
  if (reference_reconstruction &&
      FLAGS_start_frame >= 0 && FLAGS_end_frame >= 0) {
    CHECK_LT(FLAGS_start_frame, FLAGS_end_frame - 1);

    // Get view names.
    std::vector<std::string> view_names;
    view_names.reserve(reconstruction_to_align->ViewIds().size());
    for (const auto& view_id : reconstruction_to_align->ViewIds()) {
      view_names.push_back(reconstruction_to_align->View(view_id)->Name());
    }

    // Extract frame indices.
    std::unordered_map<int, std::string> frame_indices;
    ExtractFrameIndicesFromImages(view_names, &frame_indices);

    // Remove views out of range.
    for (const auto& frame_index : frame_indices) {
      if (frame_index.first < FLAGS_start_frame ||
          frame_index.first > FLAGS_end_frame) {
        const std::string& view_name = frame_index.second;
        reconstruction_to_align->RemoveView(
            reconstruction_to_align->ViewIdFromName(view_name));
      }
    }
  }

  const std::vector<std::string> common_view_names =
      theia::FindCommonViewsByName(*reference_reconstruction,
                                   *reconstruction_to_align);

  // Compare number of cameras.
  LOG(INFO) << "Number of cameras:\n"
            << "\tReconstruction 1: " << reference_reconstruction->NumViews()
            << "\n\tReconstruction 2: " << reconstruction_to_align->NumViews()
            << "\n\tNumber of Common cameras: "
            << common_view_names.size();

  if (common_view_names.size() == 0) {
    LOG(INFO) << "Could not compare reconstructions because they do not have "
                 "any common view names.";
    return 0;
  }

  // Compare number of 3d points.
  LOG(INFO) << "Number of 3d points:\n"
            << "\tReconstruction 1: " << reference_reconstruction->NumTracks()
            << "\n\tReconstruction 2: " << reconstruction_to_align->NumTracks();

  // @mhsung
  if (FLAGS_out_reference_csv != "" && FLAGS_out_to_align_csv != "") {
    CHECK(SaveCameraPositionCSVFile(
        common_view_names, *reference_reconstruction, FLAGS_out_reference_csv));
    CHECK(SaveCameraPositionCSVFile(
        common_view_names, *reconstruction_to_align, FLAGS_out_to_align_csv));
  }

  // @mhsung
  JsonFile out_file;
  if (FLAGS_out_json_file != "") {
    CHECK(out_file.Open(FLAGS_out_json_file))
    << "Can't open file '" + FLAGS_out_json_file + "'.";
    out_file.WriteElement("num_views", reference_reconstruction->NumViews());
    out_file.WriteElement(
        "num_estimated_views", reconstruction_to_align->NumViews());
  }

  if (FLAGS_normalize_with_common_views) {
    NormalizeCameraPositions(common_view_names, reference_reconstruction.get());
    NormalizeCameraPositions(common_view_names, reconstruction_to_align.get());
  }

  // Evaluate rotation independent of positions.
  EvaluateRotations(*reference_reconstruction,
                    *reconstruction_to_align,
                    common_view_names,
                    &out_file);

  // Align models and evaluate position and rotation errors.
  EvaluateAlignedPoseError(common_view_names,
                           *reference_reconstruction,
                           reconstruction_to_align.get(),
                           &out_file);

  // @mhsung
  out_file.Close();

  return 0;
}
