// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stlplus3/file_system.hpp>
#include <theia/theia.h>
#include <unordered_map>
#include <vector>

#include "exp_camera_param_utils.h"
#include "exp_camera_param_io.h"

DEFINE_string(images, "", "Wildcard of images to reconstruct.");
DEFINE_string(data_type, "reconstruction", "");
DEFINE_string(filepath, "", "");
DEFINE_string(calibration_file, "",
              "Calibration file containing image calibration data.");
DEFINE_string(output_reconstruction, "", "");


int main(int argc, char* argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  std::vector<std::string> image_files;
  CHECK(theia::GetFilepathsFromWildcard(FLAGS_images, &image_files))
  << "Could not find images that matched the filepath: " << FLAGS_images
  << ". NOTE that the ~ filepath is not supported.";

  // Load modelview matrices.
  std::unordered_map<std::string, Eigen::Affine3d> modelviews_without_ext;
  CHECK(ReadModelviews(FLAGS_data_type, FLAGS_filepath,
                       &modelviews_without_ext));

  // Add extension to modelview image file names.
  std::unordered_map<std::string, Eigen::Affine3d> modelviews;
  modelviews.reserve(modelviews_without_ext.size());
  for (const auto& image_file : image_files) {
    const std::string filename = stlplus::filename_part(image_file);
    const std::string basename = stlplus::basename_part(filename);
    const Eigen::Affine3d modelview =
      theia::FindOrDie(modelviews_without_ext, basename);
    modelviews.emplace(filename, modelview);
  }
  CHECK_EQ(modelviews_without_ext.size(), modelviews.size())
  << "Images are missing.";


  // Load calibration file if it is provided.
  std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
    camera_intrinsics_priors;
  if (FLAGS_calibration_file.size() != 0) {
    CHECK(theia::ReadCalibration(FLAGS_calibration_file,
                                 &camera_intrinsics_priors))
    << "Could not read calibration file.";
  }

  std::unique_ptr<theia::Reconstruction> reconstruction =
    CreateTheiaReconstructionFromModelviews(
      modelviews, &camera_intrinsics_priors);

  LOG(INFO) << "Writing reconstruction to " << FLAGS_output_reconstruction;
  CHECK(theia::WriteReconstruction(*reconstruction,
                                   FLAGS_output_reconstruction))
  << "Could not write reconstruction to file.";

  reconstruction.release();
  return 0;
}
