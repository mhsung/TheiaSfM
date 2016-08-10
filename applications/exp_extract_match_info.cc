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
#include "theia/sfm/estimators/estimate_constrained_relative_pose.h"
#include "exp_camera_param_utils.h"
#include "exp_camera_param_io.h"

DEFINE_string(images_dir, "", "Directory including image files.");
DEFINE_string(matches_file, "", "Filename of the matches file.");
DEFINE_string(ground_truth_data_type, "", "");
DEFINE_string(ground_truth_filepath, "", "");
DEFINE_string(output_dir, "", "Output directory to store match information.");


void GetImageNames(
    const std::vector<std::string>& image_files,
    std::vector<std::string>* image_names) {
  CHECK_NOTNULL(image_names);
  image_names->clear();
  image_names->reserve(image_files.size());

  for (const auto& image_file : image_files) {
    std::string basename;
    // Get basename without extension.
    theia::GetFilenameFromFilepath(image_file, false, &basename);
    image_names->push_back(basename);
  }
}

void GetMatchedImageNamePairs(
    const std::vector<theia::ImagePairMatch>& image_matches,
    std::vector<std::pair<std::string, std::string> >* matched_image_pairs) {
  CHECK_NOTNULL(matched_image_pairs);
  matched_image_pairs->clear();
  matched_image_pairs->reserve(image_matches.size());

  for (const auto& match : image_matches) {
    const std::string basename_1 = stlplus::basename_part(match.image1);
    const std::string basename_2 = stlplus::basename_part(match.image2);
    matched_image_pairs->emplace_back(basename_1, basename_2);
  }
}

void GetMatchedImageNamesForAll(
    std::vector<std::string>& image_names,
    const std::vector<std::pair<std::string, std::string> >&
    matched_image_pairs,
    std::unordered_map< std::string, std::list<std::string> >*
    all_matched_images) {
  CHECK_NOTNULL(all_matched_images);
  all_matched_images->clear();
  all_matched_images->reserve(matched_image_pairs.size());

  // Initialize.
  for (const auto& image_name : image_names) {
    (*all_matched_images)[image_name] = std::list<std::string>();
  }

  // Add matched images.
  for (const auto& pair_name : matched_image_pairs) {
    theia::FindOrDie(*all_matched_images, pair_name.first).
        push_back(pair_name.second);
    theia::FindOrDie(*all_matched_images, pair_name.second).
        push_back(pair_name.first);
  }
}

void WriteAllImageNames(
    const std::vector<std::string>& image_names) {
  const std::string output_filepath = FLAGS_output_dir + "/" + "images.txt";
  std::ofstream file(output_filepath);
  CHECK(file.good());
  for (const auto& image_name : image_names) {
    file << image_name << std::endl;
  }
  file.close();

  LOG(INFO) << "Saved '" << output_filepath << "'.";
  VLOG(1) << "Total " << image_names.size() << " images.";
  VLOG(1) << "Total " << image_names.size() * (image_names.size() - 1) / 2
          << " putative pairs.";
}

void WriteAllMatchedPairImageNames(
    const std::vector< std::pair<std::string, std::string> >&
    matched_image_pairs) {
  const std::string output_filepath =
      FLAGS_output_dir + "/" + "matches_pairs.txt";
  std::ofstream file(output_filepath);
  CHECK(file.good());
  for (const auto& pair_name : matched_image_pairs) {
    file << pair_name.first << "," << pair_name.second << std::endl;
  }
  file.close();

  LOG(INFO) << "Saved '" << output_filepath << "'.";
  VLOG(1) << "Total " << matched_image_pairs.size() << " pairs.";
}

void WriteNumPairsForImages(
    const std::unordered_map< std::string, std::list<std::string> >&
    all_matched_images) {

  // Sort by numbers of matched images.
  std::vector< std::pair<std::string, int> > num_matched_images;
  num_matched_images.reserve(all_matched_images.size());
  for (const auto& matched_per_image : all_matched_images) {
    num_matched_images.emplace_back(
        matched_per_image.first, matched_per_image.second.size());
  }

  std::sort(num_matched_images.begin(), num_matched_images.end(),
            [](const std::pair<std::string, int> &left,
               const std::pair<std::string, int> &right) {
      return left.second < right.second;
  });

  // Write number of matched images.
  const std::string output_filepath =
      FLAGS_output_dir + "/" + "num_matched_images.txt";
  std::ofstream file(output_filepath);
  CHECK(file.good());

  for (const auto& count_image_pair : num_matched_images) {
    file << count_image_pair.first << ","
         << count_image_pair.second << std::endl;
  }

  file.close();
  LOG(INFO) << "Saved '" << output_filepath << "'.";
}

void ComputeConnectedComponents(
    const std::vector<std::string>& image_names,
    const std::unordered_map<std::string, std::list<std::string> >&
    all_matched_images) {
  CHECK(!image_names.empty());

  std::unordered_map<std::string, int> image_component_indices;

  // Initialize.
  // '-1' component index indicates unassigned.
  for (const auto& image_name : image_names) {
    image_component_indices[image_name] = -1;
  }

  int component_index = 0;
  for (; true; ++component_index) {
    std::queue<std::string> image_name_queue;

    // Find an image not assigned to previous components.
    for (const auto& image_component_index : image_component_indices) {
      if (image_component_index.second < 0) {
        // Assign image to the current component.
        const std::string& image_name = image_component_index.first;
        image_component_indices[image_name] = component_index;
        image_name_queue.push(image_name);
        break;
      }
    }

    if (image_name_queue.empty()) break;

    while (!image_name_queue.empty()) {
      const std::string image_name = image_name_queue.front();
      image_name_queue.pop();
      CHECK_EQ(image_component_indices[image_name], component_index);

      // Traverse all connected (matched) images.
      const auto& other_image_names =
          theia::FindOrDie(all_matched_images, image_name);
      for (const auto& other_image_name : other_image_names) {
        if (image_component_indices[other_image_name] < 0) {
          image_component_indices[other_image_name] = component_index;
          image_name_queue.push(other_image_name);
        } else {
          CHECK_EQ(image_component_indices[other_image_name], component_index);
        }
      }
    }
  }

  // Next component index is the number of existing components.
  const int num_components = component_index;

  // Collect images in components.
  std::vector< std::list<std::string> > image_components(num_components);
  for (const auto& image_component_index : image_component_indices) {
    CHECK_LE(image_component_index.second, num_components);
    image_components[image_component_index.second].push_back(
        image_component_index.first);
  }

  // Count as disconnected if an image has no pair.
  std::list<std::string> disconnected_image_names;
  int num_disconnected_images = 0;

  int count_components = 0;
  for (int i = 0; i < num_components; i++) {
    const int num_component_images = image_components[i].size();
    if (num_component_images == 1) {
      disconnected_image_names.insert(
          disconnected_image_names.end(),
          image_components[i].begin(), image_components[i].end());
    } else {
      VLOG(1) << " - [" << i << "] # images: " << num_component_images;
      ++count_components;
    }
  }
  VLOG(1) << "# connected components: " << count_components;

  if (!disconnected_image_names.empty()) {
    VLOG(1) << "# disconnected images: " << disconnected_image_names.size();

    // Sort disconnected images by names.
    disconnected_image_names.sort();

    const std::string output_filepath =
        FLAGS_output_dir + "/" + "disconnected_images.txt";
    std::ofstream file(output_filepath);
    CHECK(file.good());
    for (const auto& image_name : disconnected_image_names) {
      file << image_name << std::endl;
    }
    file.close();
    LOG(INFO) << "Saved '" << output_filepath << "'.";
  }
}

void CompareRelativePosesWithGroundTruth(
    const std::vector<theia::ImagePairMatch>& image_matches,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    ground_truth_orientations) {
  const std::string output_filepath =
      FLAGS_output_dir + "/" + "camera_pose_errors.csv";
  std::ofstream file(output_filepath);
  CHECK(file.good());

  double max_relative_rotation_angle_error = 0.0;

  // TEST.
  const int kErrorTol = 30;
  std::unordered_map<std::string, bool> is_image_visited;
  is_image_visited.reserve(ground_truth_orientations.size());
  for (const auto& image : ground_truth_orientations) {
    is_image_visited[image.first] = false;
  }
  //

  for (const auto& match : image_matches) {
    const std::string basename_1 = stlplus::basename_part(match.image1);
    const std::string basename_2 = stlplus::basename_part(match.image2);

    const Eigen::Matrix3d gt_rotation1 =
        theia::FindOrDie(ground_truth_orientations, basename_1);
    const Eigen::Matrix3d gt_rotation2 =
        theia::FindOrDie(ground_truth_orientations, basename_2);

    Eigen::Matrix3d est_relative_rotation;
    ceres::AngleAxisToRotationMatrix(
        match.twoview_info.rotation_2.data(),
        ceres::ColumnMajorAdapter3x3(est_relative_rotation.data()));

    // Compute relative rotation error.
    const double relative_rotation_angle_error =
        RelativeOrientationAbsAngleError(
            gt_rotation1, gt_rotation2, est_relative_rotation);
    max_relative_rotation_angle_error = std::max(
        relative_rotation_angle_error, max_relative_rotation_angle_error);

    file << basename_1 << "," << basename_2 << ","
         << relative_rotation_angle_error << std::endl;

    // TEST.
    if (relative_rotation_angle_error <
        static_cast<double>(kErrorTol)) {
      is_image_visited[basename_1] = true;
      is_image_visited[basename_2] = true;
    }
    //
  }

  file.close();
  LOG(INFO) << "Saved '" << output_filepath << "'.";
  VLOG(1) << "Max relative rotation angle error: "
          << max_relative_rotation_angle_error;

  // TEST.
  const std::string test_output_filepath =
      FLAGS_output_dir + "/" + "disconnected_images_tol_"
      + std::to_string(kErrorTol) + ".txt";
  std::ofstream test_file(test_output_filepath);
  CHECK(test_file.good());
  for (const auto& image : is_image_visited) {
    if (!image.second) {
      test_file << image.first << std::endl;
    }
  }
  test_file.close();
  LOG(INFO) << "Saved '" << test_output_filepath << "'.";
  //
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

  // Extract information.
  std::vector<std::string> image_names;
  GetImageNames(image_files, &image_names);

  std::vector< std::pair<std::string, std::string> > matched_image_pairs;
  GetMatchedImageNamePairs(image_matches, &matched_image_pairs);

  std::unordered_map< std::string, std::list<std::string> >
      all_matched_images;
  GetMatchedImageNamesForAll(
      image_names, matched_image_pairs, &all_matched_images);

  // Empty directory.
  if (!stlplus::folder_exists(FLAGS_output_dir)) {
    CHECK(stlplus::folder_create(FLAGS_output_dir));
  }

  WriteAllImageNames(image_names);
  WriteAllMatchedPairImageNames(matched_image_pairs);
  WriteNumPairsForImages(all_matched_images);
  ComputeConnectedComponents(image_names, all_matched_images);

  if (FLAGS_ground_truth_filepath != "") {
    std::unordered_map<std::string, Eigen::Matrix3d> ground_truth_orientations;
    CHECK(ReadOrientations(
        FLAGS_ground_truth_data_type, FLAGS_ground_truth_filepath,
        &ground_truth_orientations));

    CompareRelativePosesWithGroundTruth(
        image_matches, ground_truth_orientations);
  }
}
