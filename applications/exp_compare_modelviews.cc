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
DEFINE_string(output_filepath, "", "");


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  // Load estimated data.
  std::unordered_map<std::string, Eigen::Affine3d> reference_modelviews;
  CHECK(ReadModelviews(FLAGS_reference_data_type, FLAGS_reference_filepath,
                         &reference_modelviews));

  // Load estimated data.
  std::unordered_map<std::string, Eigen::Affine3d> estimated_modelviews;
  CHECK(ReadModelviews(FLAGS_estimate_data_type, FLAGS_estimate_filepath,
                         &estimated_modelviews));

  // Sync estimated modelviews with reference.
  std::unordered_map<std::string, Eigen::Affine3d>
      relative_estimated_modelviews;
  SyncModelviewSequences(reference_modelviews, estimated_modelviews,
                           &relative_estimated_modelviews);

  WriteModelviews(
      FLAGS_output_filepath, relative_estimated_modelviews);
}
