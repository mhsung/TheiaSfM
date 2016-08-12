// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

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
#include "theia/sfm/estimators/estimate_constrained_relative_pose.h"


// Input/output files.
DEFINE_string(reference_data_type, "", "");
DEFINE_string(reference_filepath, "", "");
DEFINE_string(estimate_data_type, "", "");
DEFINE_string(estimate_filepath, "", "");
DEFINE_string(estimate_sequence_index_filepath, "", "");
DEFINE_string(output_filepath, "", "");


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  // Load estimated data.
  std::unordered_map<std::string, Eigen::Matrix3d> reference_orientations;
  CHECK(ReadOrientations(FLAGS_reference_data_type, FLAGS_reference_filepath,
                         &reference_orientations));

  // Load estimated data.
  std::unordered_map<std::string, Eigen::Matrix3d> estimated_orientations;
  CHECK(ReadOrientations(FLAGS_estimate_data_type, FLAGS_estimate_filepath,
                         &estimated_orientations));

  std::unordered_map<std::string, int> seq_indices;
  CHECK(ReadSequenceIndices(
      FLAGS_estimate_sequence_index_filepath, &seq_indices));

  std::ofstream file(FLAGS_output_filepath);
  CHECK(file.good());
  double max_relative_rotation_angle_error = 0.0;

  for (const auto& estimate1 : estimated_orientations) {
    const Eigen::Matrix3d* reference_rotation1 =
        theia::FindOrNull(reference_orientations, estimate1.first);
    if (reference_rotation1 == nullptr) continue;

    const int* seq_index1 = theia::FindOrNull(seq_indices, estimate1.first);
    // Ignore images in no sequence.
    if (seq_index1 != nullptr && (*seq_index1) < 0) continue;

    for (const auto& estimate2 : estimated_orientations) {
      if (estimate1.first >= estimate2.first) continue;

      const Eigen::Matrix3d* reference_rotation2 =
          theia::FindOrNull(reference_orientations, estimate2.first);
      if (reference_rotation2 == nullptr) continue;
      const int* seq_index2 = theia::FindOrNull(seq_indices, estimate2.first);
      // Ignore images in no sequence.
      if (seq_index2 != nullptr && (*seq_index2) < 0) continue;

      // Ignore image pairs in different sequences.
      if (seq_index1 != nullptr && seq_index2 != nullptr
          && (*seq_index1) != (*seq_index2)) {
        continue;
      }

      // Compute relative rotation error.
      const Eigen::Matrix3d estimated_relative_rotation =
          estimate2.second * estimate1.second.transpose();
      const double relative_rotation_angle_error =
          RelativeOrientationAbsAngleError(
              *reference_rotation1, *reference_rotation2,
              estimated_relative_rotation);
      max_relative_rotation_angle_error = std::max(
          relative_rotation_angle_error, max_relative_rotation_angle_error);

      file << estimate1.first << "," << estimate2.first << ","
           << relative_rotation_angle_error << std::endl;
    }
  }

  file.close();
  LOG(INFO) << "Saved '" << FLAGS_output_filepath << "'.";
  VLOG(1) << "Max relative rotation angle error: "
          << max_relative_rotation_angle_error;
}
