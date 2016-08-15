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
#include "applications/exp_feature_track_io.h"
#include "theia/matching/feature_matcher_options.h"

// Input/output files.
DEFINE_string(images, "", "Wildcard of images to reconstruct.");
DEFINE_string(feature_tracks_file, "", "Filename of the feature track file.");
DEFINE_string(output_dir, "", "Output directory.");


void DrawFeatures(
    const std::string& image_filepath,
    const std::list<theia::Feature>& features) {
  CHECK(theia::FileExists(image_filepath))
  << "Image file does not exist: '" << image_filepath << "'";
  const theia::FloatImage image(image_filepath);

  theia::ImageCanvas canvas;
  canvas.AddImage(image);

  for (const auto& feature : features) {
    canvas.DrawFeature(feature, theia::RGBPixel(0.0f, 1.0f, 0.0f), 1.0);
  }

  std::string image_name;
  CHECK(theia::GetFilenameFromFilepath(image_filepath, false, &image_name));

  // Create output directory.
  if (!theia::DirectoryExists(FLAGS_output_dir)) {
    CHECK(theia::CreateNewDirectory(FLAGS_output_dir));
  }

  const std::string output_filepath =
      FLAGS_output_dir + "/" + image_name + ".png";
  canvas.Write(output_filepath);
  LOG(INFO) << "Saved '" << output_filepath << "'.";
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Get the filepaths of the image files.
  std::vector<std::string> image_files;
  CHECK(theia::GetFilepathsFromWildcard(FLAGS_images, &image_files))
  << "Could not find images that matched the filepath: " << FLAGS_images
  << ". NOTE that the ~ filepath is not supported.";
  CHECK_GT(image_files.size(), 0) << "No images found in: " << FLAGS_images;

  // Read the feature tracks.
  std::list<theia::FeatureTrackPtr> feature_tracks;
  CHECK(ReadFeatureTracks(FLAGS_feature_tracks_file, &feature_tracks));

  std::unordered_map<int, std::list<theia::Feature> > image_features;
  GetImageFeaturesFromFeatureTracks(feature_tracks, &image_features);

  const int num_images = image_files.size();
  for (int i = 0; i < num_images; ++i) {
    const std::string image_file = image_files[i];
    // FIXME:
    // Do not use image index.
    const std::list<theia::Feature>* features =
        theia::FindOrNull(image_features, i);
    if (features != nullptr) {
      DrawFeatures(image_file, *features);
    }
  }
}
