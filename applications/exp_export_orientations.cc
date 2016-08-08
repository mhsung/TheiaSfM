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
DEFINE_string(input_data_type, "",
              "'param', 'pose', 'modelview', or " "'reconstruction'");
DEFINE_string(input_filepath, "", "");
DEFINE_string(output_filepath, "", "");


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  std::unordered_map<std::string, Eigen::Matrix3d> orientations;
  CHECK(ReadOrientations(
      FLAGS_input_data_type, FLAGS_input_filepath, &orientations));

  WriteOrientationsAsCameraParams(FLAGS_output_filepath, orientations);
}
