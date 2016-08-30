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
#include "applications/exp_camera_param_io.h"
#include "applications/exp_camera_param_utils.h"
#include "applications/exp_feature_match_utils.h"
#include "applications/exp_feature_track_io.h"
#include "theia/matching/feature_matcher_options.h"


// Input/output files.
DEFINE_string(image_filenames_file, "", "");
DEFINE_string(calibration_file, "",
              "Calibration file containing image calibration data.");
DEFINE_string(feature_tracks_file, "", "Filename of the feature track file.");
DEFINE_string(output_matches_file, "", "Filename of the matches file.");

// Options.
DEFINE_bool(geometrically_verify_matches, true,
            "Set to true to perform geometric verification on the matches.");
DEFINE_double(max_sampson_error_for_verified_match, 4.0,
              "Maximum sampson error for a match to be considered geometrically"
                  "valid.");
DEFINE_int32(min_num_inliers_for_valid_match, 30,
             "Minimum number of geometrically verified inliers that a pair on "
                 "images must have in order to be considered a valid two-view "
                 "match.");
DEFINE_bool(bundle_adjust_two_view_geometry, true,
            "Set to false to turn off 2-view BA.");

// @mhsung
DEFINE_string(initial_orientations_data_type, "", "");
DEFINE_string(initial_orientations_filepath, "", "");


// @mhsung
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

// @mhsung
void ReadInitialOrientations(
    const std::vector<std::string>& image_filenames,
    std::vector<Eigen::Matrix3d>* initial_orientations) {
  CHECK_NOTNULL(initial_orientations)->clear();

  std::unordered_map<std::string, Eigen::Matrix3d>
      initial_orientations_with_names;
  CHECK(ReadOrientations(FLAGS_initial_orientations_data_type,
                         FLAGS_initial_orientations_filepath,
                         &initial_orientations_with_names));

  initial_orientations->reserve(initial_orientations_with_names.size());
  for (const auto& image_name : image_filenames) {
    std::string basename;
    CHECK(theia::GetFilenameFromFilepath(image_name, false, &basename));
    const Eigen::Matrix3d& orientation =
        FindOrDie(initial_orientations_with_names, basename);
    initial_orientations->push_back(orientation);
  }
}

void SetMatchingOptions(theia::FeatureMatcherOptions* matching_options) {
  matching_options->perform_geometric_verification =
      FLAGS_geometrically_verify_matches;
  matching_options->geometric_verification_options.estimate_twoview_info_options
      .max_sampson_error_pixels = FLAGS_max_sampson_error_for_verified_match;
  matching_options->geometric_verification_options.bundle_adjustment =
      FLAGS_bundle_adjust_two_view_geometry;

  // @mhsung
  matching_options->geometric_verification_options.min_num_inlier_matches =
      FLAGS_min_num_inliers_for_valid_match;
}

// Read the camera intrinsics from a file.
void ReadIntrinsicsFromCalibrationFile(
    const std::vector<std::string>& image_filenames,
    std::vector<theia::CameraIntrinsicsPrior>* intrinsics) {
  intrinsics->resize(image_filenames.size());

  std::unordered_map<std::string, theia::CameraIntrinsicsPrior> intrinsics_map;
  if (!theia::ReadCalibration(FLAGS_calibration_file, &intrinsics_map)) {
    return;
  }

  for (int i = 0; i < image_filenames.size(); i++) {
    if (theia::ContainsKey(intrinsics_map, image_filenames[i])) {
      intrinsics->at(i) = theia::FindOrDie(intrinsics_map, image_filenames[i]);
    }
  }
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Get the filenames of the image files.
  // The vector index is equal to view ID.
  std::vector<std::string> image_filenames;
  ReadImageFilenames(&image_filenames);

  // Read the intrinsics from a calibration file.
  std::vector<theia::CameraIntrinsicsPrior> intrinsics;
  ReadIntrinsicsFromCalibrationFile(image_filenames, &intrinsics);

  theia::FeatureMatcherOptions matching_options;
  SetMatchingOptions(&matching_options);

  // @mhsung
  std::unordered_map<std::string, Eigen::Matrix3d> initial_orientations;
  if (FLAGS_initial_orientations_filepath != "") {
    std::unordered_map<std::string, Eigen::Matrix3d>
        initial_orientations_with_basenames;
    CHECK(ReadOrientations(FLAGS_initial_orientations_data_type,
                           FLAGS_initial_orientations_filepath,
                           &initial_orientations_with_basenames));
    CHECK(CheckViewNamesValid(initial_orientations_with_basenames,
                              image_filenames, &initial_orientations));
  }

  // Read the feature tracks.
  std::list<theia::FeatureTrackPtr> feature_tracks;
  CHECK(ReadFeatureTracks(FLAGS_feature_tracks_file, &feature_tracks));

  // Extract matches from the feature tracks.
  std::unordered_map<theia::ViewIdPair, std::list<FeatureCorrespondence> >
      image_pair_correspondences;
  GetCorrespodnencesFromFeatureTracks(
      feature_tracks, &image_pair_correspondences);
  LOG(INFO) << "Parsing feature tracks completed.";

  std::vector<theia::ImagePairMatch> matches;
  CreateMatchesFromCorrespondences(
      image_pair_correspondences, image_filenames, intrinsics,
      initial_orientations, matching_options, &matches);
  LOG(INFO) << "Extracting matches from feature tracks completed.";

  // Write the matches out.
  LOG(INFO) << "Writing matches to file: " << FLAGS_output_matches_file;
  CHECK(theia::WriteMatchesAndGeometry(FLAGS_output_matches_file,
                                       image_filenames, intrinsics, matches))
  << "Could not write the matches to " << FLAGS_output_matches_file;
}
