// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <stlplus3/file_system.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "applications/command_line_helpers.h"
#include "theia/image/image.h"
#include "theia/image/image_canvas.h"
#include "exp_camera_param_utils.h"
#include "exp_camera_param_io.h"

DEFINE_string(images_dir, "", "Directory including image files.");
DEFINE_string(matches_file, "", "Filename of the matches file.");
DEFINE_string(match_pairs_file, "", "Filename of match pair list. Each line "
    "has 'name1,name2' format.");
DEFINE_string(output_dir, "", "Output directory to store feature match "
    "drawing images.");


bool ReadMatchPairs(
    const std::string& filename,
    std::vector<std::pair<std::string, std::string> >* pairs_to_match) {
  CHECK_NOTNULL(pairs_to_match);

  std::ifstream file(filename);
  if (!file.good()) {
    LOG(WARNING) << "Can't read file: '" << filename << "'.";
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {

    // FIXME:
    // Remove '\r' in 'std::getline' function.
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    std::stringstream sstr(line);
    std::string name1, name2;
    if (!std::getline(sstr, name1, ',') || !std::getline(sstr, name2, ',')) {
      LOG(WARNING) << "Wrong format: '" << line << "'.";
      return false;
    }
    pairs_to_match->emplace_back(name1, name2);
  }

  CHECK(!pairs_to_match->empty()) << "No image pair to match.";
  LOG(INFO) << "# of pairs: " << pairs_to_match->size();
  return true;
}

void DrawMatchedFeatures(const theia::ImagePairMatch& match) {
  const std::string image_filepath1 =
      FLAGS_images_dir + "/" + match.image1;
  const std::string image_filepath2 =
      FLAGS_images_dir + "/" + match.image2;

  CHECK(stlplus::file_exists(image_filepath1))
  << "Image file does not exist: '" << image_filepath1 << "'";
  CHECK(stlplus::file_exists(image_filepath2))
  << "Image file does not exist: '" << image_filepath2 << "'";

  const theia::FloatImage image1(image_filepath1);
  const theia::FloatImage image2(image_filepath2);

  theia::ImageCanvas canvas;
  const int canvas_index1 = canvas.AddImage(image1);
  const int canvas_index2 = canvas.AddImage(image2);
  canvas.DrawMatchedFeatures(
      canvas_index1, canvas_index2, match.correspondences);

  const std::string basename_1 = stlplus::basename_part(match.image1);
  const std::string basename_2 = stlplus::basename_part(match.image2);
  const std::string output_filepath =
      FLAGS_output_dir + "/" + basename_1 + "_" + basename_2 + ".png";
  canvas.Write(output_filepath);
  LOG(INFO) << "Saved '" << output_filepath << "'.";
}

void DrawAllMatchedFeatures(
    const std::vector<theia::ImagePairMatch>& image_matches,
    const std::vector<std::pair<std::string, std::string>>& pairs_to_draw) {
  // FIXME:
  // Change the double loop to a more efficient way.
  for (const auto& pair : pairs_to_draw) {
    for (const auto& match : image_matches) {
      const std::string basename_1 = stlplus::basename_part(match.image1);
      const std::string basename_2 = stlplus::basename_part(match.image2);

      if ((pair.first == basename_1 && pair.second == basename_2) ||
          (pair.first == basename_2 && pair.second == basename_1)) {
        DrawMatchedFeatures(match);
        break;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Load matches from file.
  std::vector<std::string> image_files;
  std::vector<theia::CameraIntrinsicsPrior> camera_intrinsics_prior;
  std::vector<theia::ImagePairMatch> image_matches;

  // Read in match file.
  CHECK(theia::ReadMatchesAndGeometry(
      FLAGS_matches_file, &image_files, &camera_intrinsics_prior,
      &image_matches));

  std::vector<std::pair<std::string, std::string> > pairs_to_draw;
  CHECK(ReadMatchPairs(FLAGS_match_pairs_file, &pairs_to_draw));

  // Empty directory.
  if (stlplus::folder_exists(FLAGS_output_dir)) {
    CHECK(stlplus::folder_delete(FLAGS_output_dir, true));
  }
  CHECK(stlplus::folder_create(FLAGS_output_dir));
  CHECK(stlplus::folder_writable(FLAGS_output_dir));

  DrawAllMatchedFeatures(image_matches, pairs_to_draw);
}
