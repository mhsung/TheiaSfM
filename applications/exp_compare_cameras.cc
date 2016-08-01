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
DEFINE_string(reference_reconstruction_file, "",
              "Reference reconstruction file in binary format.");
DEFINE_string(target_data_type, "", "");
DEFINE_string(target_filepath, "", "");
DEFINE_string(output_relative_target_camera_param_dir, "", "");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  // Load the reference reconstruction.
  theia::Reconstruction ref_reconstruction;
  CHECK(theia::ReadReconstruction(
      FLAGS_reference_reconstruction_file, &ref_reconstruction))
      << "Could not read reconstruction file: '"
      << FLAGS_reference_reconstruction_file << "'.";

  std::unordered_map<theia::ViewId, Eigen::Matrix3d> ref_orientations;
  GetOrientationsFromReconstruction(ref_reconstruction, &ref_orientations);


  // Load target data.
  std::unordered_map<std::string, Eigen::Matrix3d>
      est_orientations_with_names;
  CHECK(ReadOrientations(FLAGS_target_data_type, FLAGS_target_filepath,
                         &est_orientations_with_names));

  std::unordered_map<theia::ViewId, Eigen::Matrix3d> est_orientations;
  MapOrientationsToViewIds(
      ref_reconstruction, est_orientations_with_names, &est_orientations);


  // Sync target (estimated) modelviews with reference.
  std::unordered_map<theia::ViewId, Eigen::Matrix3d> rel_est_orientations;
  SyncOrientationSequences(
      ref_orientations, est_orientations, &rel_est_orientations);

//  std::unordered_map<theia::ViewId, Eigen::Matrix3d> rel_est_orientations;
//  ComputeRelativeOrientationsFromFirstFrame(
//      est_orientations, &rel_est_orientations);

  std::unordered_map<std::string, Eigen::Matrix3d>
      rel_est_modelviews_with_names;
  MapOrientationsToViewNames(
      ref_reconstruction, rel_est_orientations, &rel_est_modelviews_with_names);
  WriteOrientationsAsCameraParams(
      FLAGS_output_relative_target_camera_param_dir,
      rel_est_modelviews_with_names);
}
