// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_feature_match_utils.h"

#include <glog/logging.h>


namespace theia {

void GetPutativeMatches(
    const std::string& image1_name, const std::string& image2_name,
    const std::list<theia::FeatureCorrespondence>& correspondences,
    theia::KeypointsAndDescriptors* features1,
    theia::KeypointsAndDescriptors* features2,
    std::vector<theia::IndexedFeatureMatch>* putative_matches) {
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
    const theia::Feature& keypoint1 = correspondence.feature1;
    const theia::Feature& keypoint2 = correspondence.feature2;
    features1->keypoints.push_back(
        theia::Keypoint(keypoint1[0], keypoint1[1], theia::Keypoint::OTHER));
    features2->keypoints.push_back(
        theia::Keypoint(keypoint2[0], keypoint2[1], theia::Keypoint::OTHER));
    putative_matches->push_back(
        theia::IndexedFeatureMatch(count_corrs, count_corrs, 0.0f));
    ++count_corrs;
  }
}

void CreateMatchesFromCorrespondences(
    std::unordered_map<ViewIdPair, std::list<theia::FeatureCorrespondence> >&
    image_pair_correspondences,
    const std::vector<std::string>& image_filenames,
    const std::vector<theia::CameraIntrinsicsPrior>& intrinsics,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    initial_orientations,
    const theia::FeatureMatcherOptions& options,
    std::vector<theia::ImagePairMatch>* matches) {
  CHECK_NOTNULL(matches)->clear();

  const int num_images = image_filenames.size();
  CHECK_EQ(intrinsics.size(), num_images);

  // @mhsung
  int num_tested_pairs = 0;
  int num_pairs_with_inits = 0;
  int num_passed_with_passed_without_inits = 0;
  int num_failed_with_passed_without_inits = 0;
  int num_passed_with_failed_without_inits = 0;
  int num_failed_with_failed_without_inits = 0;

  for (const auto& image_pair : image_pair_correspondences) {
    const ViewId view1_idx = image_pair.first.first;
    const ViewId view2_idx = image_pair.first.second;
    CHECK_LT(view1_idx, num_images);
    CHECK_LT(view2_idx, num_images);
    const std::string& image1_name = image_filenames[view1_idx];
    const std::string& image2_name = image_filenames[view2_idx];

    theia::ImagePairMatch image_pair_match;
    image_pair_match.image1 = image1_name;
    image_pair_match.image2 = image2_name;

    theia::KeypointsAndDescriptors features1, features2;
    std::vector<theia::IndexedFeatureMatch> putative_matches;
    GetPutativeMatches(image1_name, image2_name, image_pair.second,
                       &features1, &features2, &putative_matches);

    // Perform geometric verification if applicable.
    if (options.perform_geometric_verification) {
      // @mhsung
      ++num_tested_pairs;

      const theia::CameraIntrinsicsPrior& intrinsics1 = intrinsics[view1_idx];
      const theia::CameraIntrinsicsPrior& intrinsics2 = intrinsics[view2_idx];

      theia::TwoViewMatchGeometricVerification geometric_verification(
          options.geometric_verification_options, intrinsics1, intrinsics2,
          features1, features2, putative_matches);

      const Eigen::Matrix3d* initial_orientation1 =
          theia::FindOrNull(initial_orientations, image1_name);
      const Eigen::Matrix3d* initial_orientation2 =
          theia::FindOrNull(initial_orientations, image2_name);

      if (initial_orientation1 != nullptr && initial_orientation2 != nullptr) {
        // @mhsung
        // Test without initial orientations.
        const bool ret_without_inits = geometric_verification.VerifyMatches(
            &image_pair_match.correspondences,
            &image_pair_match.twoview_info);

        geometric_verification.SetInitialOrientations(
            initial_orientation1, initial_orientation2);

        const bool ret_with_inits = geometric_verification.VerifyMatches(
            &image_pair_match.correspondences,
            &image_pair_match.twoview_info);

        // @mhsung
        ++num_pairs_with_inits;
        if (ret_with_inits && ret_without_inits) {
          ++num_passed_with_passed_without_inits;
        } else if (!ret_with_inits && ret_without_inits) {
          ++num_failed_with_passed_without_inits;
        } else if (ret_with_inits && !ret_without_inits) {
          ++num_passed_with_failed_without_inits;
        } else if (!ret_with_inits && !ret_without_inits) {
          ++num_failed_with_failed_without_inits;
        }

        if (!ret_with_inits) {
          continue;
        }
      } else {
        // If geometric verification fails, do not add the match to the output.
        if (!geometric_verification.VerifyMatches(
            &image_pair_match.correspondences,
            &image_pair_match.twoview_info)) {
          continue;
        }
      }
    } else {
      // If no geometric verification is performed then the putative matches are
      // output.
      image_pair_match.correspondences.reserve(putative_matches.size());
      for (int i = 0; i < putative_matches.size(); i++) {
        const theia::Keypoint& keypoint1 =
            features1.keypoints[putative_matches[i].feature1_ind];
        const theia::Keypoint& keypoint2 =
            features2.keypoints[putative_matches[i].feature2_ind];
        image_pair_match.correspondences.emplace_back(
            theia::Feature(keypoint1.x(), keypoint1.y()),
            theia::Feature(keypoint2.x(), keypoint2.y()));
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

  VLOG(2) << "# tested pairs: " << num_tested_pairs;
  VLOG(2) << "# pairs w/ inits: " << num_pairs_with_inits;
  VLOG(2) << "# passed w/ & passed w/o inits: "
          << num_passed_with_passed_without_inits;
  VLOG(2) << "# failed w/ & passed w/o inits: "
          << num_failed_with_passed_without_inits;
  VLOG(2) << "# passed w/ & failed w/o inits: "
          << num_passed_with_failed_without_inits;
  VLOG(2) << "# failed w/ & failed w/o inits: "
          << num_failed_with_failed_without_inits;
}
}
