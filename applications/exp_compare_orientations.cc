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


// Input/output files.
DEFINE_string(reference_data_type, "", "");
DEFINE_string(reference_filepath, "", "");
DEFINE_string(estimate_data_type, "", "");
DEFINE_string(estimate_filepath, "", "");
DEFINE_string(pivot_image_name, "", "");
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

  // Sync estimated orientations with reference.
  std::unordered_map<std::string, Eigen::Matrix3d>
      relative_estimated_orientations;
  if (FLAGS_pivot_image_name != "") {
    LOG(INFO) << "Pivot view: " << FLAGS_pivot_image_name;
    SyncOrientationSequencesWithPivot(
        FLAGS_pivot_image_name, reference_orientations, estimated_orientations,
        &relative_estimated_orientations);
  } else {
    SyncOrientationSequences(
        reference_orientations, estimated_orientations,
        &relative_estimated_orientations);
  }

  WriteOrientationsAsCameraParams(
      FLAGS_output_filepath, relative_estimated_orientations);
}
