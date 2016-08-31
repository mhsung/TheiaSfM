// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <ceres/rotation.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <chrono>  // NOLINT
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "applications/command_line_helpers.h"
#include "applications/exp_feature_track_io.h"
#include "theia/matching/feature_matcher_options.h"


// Input/output files.
DEFINE_string(images_dir, "", "");
DEFINE_string(image_filenames_file, "", "");
DEFINE_string(feature_tracks_file, "", "");
DEFINE_string(output_dir, "", "");


// Get the image filenames.
// The vector index is equal to view ID.
void ReadImageFilenames(std::vector<std::string>* image_filenames) {
  CHECK_NOTNULL(image_filenames)->clear();

  std::ifstream file(FLAGS_image_filenames_file);
  CHECK(file.good()) << "Can't read file: '" << FLAGS_image_filenames_file;

  std::string line;
  while(std::getline(file, line)) {
    if (line == "") continue;
    image_filenames->push_back(line);
  }
}

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
  VLOG(3) << "Saved '" << output_filepath << "'.";
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Get the filenames of the image files.
  // The vector index is equal to view ID.
  std::vector<std::string> image_filenames;
  ReadImageFilenames(&image_filenames);

  // Read the feature tracks.
  std::list<theia::FeatureTrackPtr> feature_tracks;
  CHECK(ReadFeatureTracks(FLAGS_feature_tracks_file, &feature_tracks));

  std::unordered_map<theia::ViewId, std::list<theia::Feature> > image_features;
  GetImageFeaturesFromFeatureTracks(feature_tracks, &image_features);

  for (theia::ViewId view_id = 0; view_id < image_filenames.size(); ++view_id) {
    const std::string image_file =
        FLAGS_images_dir + "/" + image_filenames[view_id];
    const std::list<theia::Feature>* features =
        theia::FindOrNull(image_features, view_id);
    if (features != nullptr) {
      DrawFeatures(image_file, *features);
    }
  }
}
