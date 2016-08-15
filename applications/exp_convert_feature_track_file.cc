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


void GetPutativeMatches(
    const std::string& image1_name, const std::string& image2_name,
    const std::list<FeatureCorrespondence>& correspondences,
    KeypointsAndDescriptors* features1, KeypointsAndDescriptors* features2,
    std::vector<IndexedFeatureMatch>* putative_matches) {
  CHECK_NOTNULL(features1)->keypoints.clear();
  CHECK_NOTNULL(features2)->keypoints.clear();
  CHECK_NOTNULL(putative_matches)->clear();

  const int num_correspondences = correspondences.size();

  features1->image_name = image1_name;
  features2->image_name = image2_name;
  features1->keypoints.reserve(num_correspondences);
  features2->keypoints.reserve(num_correspondences);
  putative_matches->reserve(num_correspondences);

  int count_corrs = 0;
  for (const auto& correspondence : correspondences) {
    const Feature& keypoint1 = correspondence.feature1;
    const Feature& keypoint2 = correspondence.feature2;
    features1->keypoints.push_back(
        Keypoint(keypoint1[0], keypoint1[1], Keypoint::OTHER));
    features2->keypoints.push_back(
        Keypoint(keypoint2[0], keypoint2[1], Keypoint::OTHER));
    putative_matches->push_back(
        IndexedFeatureMatch(count_corrs, count_corrs, 0.0f));
    ++count_corrs;
  }
}

void CreateMatchesFromCorrespondences(
    std::unordered_map< std::pair<int, int>, std::list<FeatureCorrespondence> >&
    image_pair_correspondences,
    const std::vector<std::string>& image_filenames,
    const std::vector<CameraIntrinsicsPrior>& intrinsics,
    const std::vector<Eigen::Matrix3d>& initial_orientations,
    const FeatureMatcherOptions& options,
    std::vector<theia::ImagePairMatch>* matches) {
  CHECK_NOTNULL(matches)->clear();

  const int num_images = image_filenames.size();
  CHECK_EQ(intrinsics.size(), num_images);
  if (options.use_initial_orientation_constraints) {
    CHECK_EQ(initial_orientations.size(), num_images);
  }

  for (const auto& image_pair : image_pair_correspondences) {
    const int image1_idx = image_pair.first.first;
    const int image2_idx = image_pair.first.second;
    CHECK_LT(image1_idx, num_images);
    CHECK_LT(image2_idx, num_images);
    const std::string& image1_name = image_filenames[image1_idx];
    const std::string& image2_name = image_filenames[image2_idx];

    ImagePairMatch image_pair_match;
    image_pair_match.image1 = image1_name;
    image_pair_match.image2 = image2_name;

    KeypointsAndDescriptors features1, features2;
    std::vector<IndexedFeatureMatch> putative_matches;
    GetPutativeMatches(image1_name, image2_name, image_pair.second,
                       &features1, &features2, &putative_matches);


    // Perform geometric verification if applicable.
    if (options.perform_geometric_verification) {
      const CameraIntrinsicsPrior& intrinsics1 = intrinsics[image1_idx];
      const CameraIntrinsicsPrior& intrinsics2 = intrinsics[image2_idx];

      // @mhsung
      std::unique_ptr<TwoViewMatchGeometricVerification> geometric_verification;
      if (options.use_initial_orientation_constraints) {
        const Eigen::Matrix3d& initial_orientation1 =
            initial_orientations[image1_idx];
        const Eigen::Matrix3d& initial_orientation2 =
            initial_orientations[image2_idx];
        geometric_verification.reset(new TwoViewMatchGeometricVerification(
            options.geometric_verification_options, intrinsics1, intrinsics2,
            features1, features2, initial_orientation1, initial_orientation2,
            putative_matches));
      } else {
        geometric_verification.reset(new TwoViewMatchGeometricVerification(
            options.geometric_verification_options, intrinsics1, intrinsics2,
            features1, features2, putative_matches));
      }

      // If geometric verification fails, do not add the match to the output.
      if (!geometric_verification->VerifyMatches(
          &image_pair_match.correspondences,
          &image_pair_match.twoview_info)) {
        VLOG(2) << "Geometric verification between images " << image1_name
                << " and " << image2_name << " failed.";
        continue;
      }
    } else {
      // If no geometric verification is performed then the putative matches are
      // output.
      image_pair_match.correspondences.reserve(putative_matches.size());
      for (int i = 0; i < putative_matches.size(); i++) {
        const Keypoint& keypoint1 =
            features1.keypoints[putative_matches[i].feature1_ind];
        const Keypoint& keypoint2 =
            features2.keypoints[putative_matches[i].feature2_ind];
        image_pair_match.correspondences.emplace_back(
            Feature(keypoint1.x(), keypoint1.y()),
            Feature(keypoint2.x(), keypoint2.y()));
      }
    }

    // Log information about the matching results.
    VLOG(1) << "Images " << image1_name << " and " << image2_name
            << " were matched with " << image_pair_match.correspondences.size()
            << " verified matches and "
            << image_pair_match.twoview_info.num_homography_inliers
            << " homography matches out of " << putative_matches.size()
            << " putative matches.";
    matches->push_back(image_pair_match);
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
  std::vector<Eigen::Matrix3d> initial_orientations;
  if (FLAGS_initial_orientations_filepath != "") {
    ReadInitialOrientations(image_filenames, &initial_orientations);
    CHECK(initial_orientations.size() == image_filenames.size());
    matching_options.use_initial_orientation_constraints = true;
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
