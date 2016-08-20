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
DEFINE_string(images, "", "Wildcard of images to reconstruct.");
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

// Gets the image filenames and filepaths from the features filepaths.
void GetImageFilesAndFilenames(
    const std::vector<std::string>& features_filepaths,
    std::vector<std::string>* image_filenames) {
  image_filenames->resize(features_filepaths.size());
  for (int i = 0; i < features_filepaths.size(); i++) {
    const std::size_t features_pos = features_filepaths[i].find(".features");
    const std::string image_filepath =
        features_filepaths[i].substr(0, features_pos);
    CHECK(theia::GetFilenameFromFilepath(image_filepath,
                                         true,
                                         &image_filenames->at(i)));
  }
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

  // Get the filepaths of the image files.
  std::vector<std::string> image_files;
  CHECK(theia::GetFilepathsFromWildcard(FLAGS_images, &image_files))
  << "Could not find images that matched the filepath: " << FLAGS_images
  << ". NOTE that the ~ filepath is not supported.";
  CHECK_GT(image_files.size(), 0) << "No images found in: " << FLAGS_images;

  // Get the image files and image filenames.
  std::vector<std::string> image_filenames;
  GetImageFilesAndFilenames(image_files, &image_filenames);

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
    CHECK(CheckOrientationNamesValid(initial_orientations_with_basenames,
                                     image_filenames, &initial_orientations));
  }

  // Read the feature tracks.
  std::list<FeatureTrackPtr> feature_tracks;
  CHECK(ReadFeatureTracks(FLAGS_feature_tracks_file, &feature_tracks));

  // Extract matches from the feature tracks.
  std::unordered_map< std::pair<int, int>, std::list<FeatureCorrespondence> >
      image_pair_correspondences;
  GetCorrespodnencesFromFeatureTracks(
      feature_tracks, &image_pair_correspondences);
  LOG(INFO) << "Parsing feature tracks completed.";

  std::vector<theia::ImagePairMatch> matches;
  //CreateMatchesFromCorrespondences(
  CreateMatchesFromCorrespondencesTest(
      image_pair_correspondences, image_filenames, intrinsics,
      initial_orientations, matching_options, &matches);
  LOG(INFO) << "Extracting matches from feature tracks completed.";

  // Write the matches out.
  LOG(INFO) << "Writing matches to file: " << FLAGS_output_matches_file;
  CHECK(theia::WriteMatchesAndGeometry(FLAGS_output_matches_file,
                                       image_filenames, intrinsics, matches))
  << "Could not write the matches to " << FLAGS_output_matches_file;
}
