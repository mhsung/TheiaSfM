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
#include <memory>
#include <stlplus3/file_system.hpp>
#include <string>
#include <sstream>
#include <vector>

#include "applications/exp_bounding_box_utils.h"
#include "applications/exp_feature_match_utils.h"
#include "applications/exp_feature_track_io.h"


// Input/output files.
DEFINE_string(image_filenames_file,
              "/Users/msung/Developer/data/7-scenes/office/seq-01/feature_tracks"
                  "/image_filename.txt", "");
DEFINE_string(feature_tracks_file,
              "/Users/msung/Developer/data/7-scenes/office/seq-01/feature_tracks"
                  "/feature_tracks.txt", "");
DEFINE_string(bounding_boxes_dir, "/Users/msung/Developer/data/7-scenes/office/seq-01/convnet/cropped", "");
DEFINE_string(output_dir, "/Users/msung/Developer/data/7-scenes/office/seq-01/convnet"
    "/objects", "");
DEFINE_double(min_iou_score_for_overlap, 0.5, "");


namespace {
    struct ObjectSequence {
        std::list<std::pair<theia::ViewId, std::string> > bounding_boxes;
        std::set<theia::TrackId> feature_track_indices_to_match;
    };
    typedef std::unique_ptr<ObjectSequence> ObjectSequencePtr;
}

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

bool GetStartAndEndIndices(
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    theia::ViewId* start_index, theia::ViewId* end_index) {
  CHECK_NOTNULL(start_index);
  CHECK_NOTNULL(end_index);

  (*start_index) = std::numeric_limits<theia::ViewId>::max();
  (*end_index) = 0;

  for (const auto& feature_track : feature_tracks) {
    const theia::ViewId track_start_index = feature_track->start_index_;
    const theia::ViewId track_end_index =
        feature_track->start_index_ + feature_track->length_;
    *start_index = std::min(*start_index, track_start_index);
    *end_index = std::max(*end_index, track_end_index);
  }

  VLOG(2) << "Start View ID: " << *start_index;
  VLOG(2) << "End View ID: " << *end_index;
  return (*end_index > *start_index);
}

void GetIncludedFeatureTrackIndices(
    const theia::ViewId& view_id,
    const Eigen::Vector4d& bounding_box,
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    std::set<theia::TrackId>* feature_tracks_indices) {
  CHECK_NOTNULL(feature_tracks_indices)->clear();

  theia::TrackId feature_track_id = 0;

  for (const auto& feature_track : feature_tracks) {
    // Skip if feature track view range does not include the view.
    if (view_id < feature_track->start_index_ ||
        view_id >= feature_track->start_index_ + feature_track->length_) {
      continue;
    }

    CHECK(feature_track->points_.size() == feature_track->length_);
    const Eigen::Vector2d image_point =
        feature_track->points_[view_id - feature_track->start_index_];

    if (theia::IsPointIncludedInBoundingBox(image_point, bounding_box)) {
      feature_tracks_indices->insert(feature_track_id);
    }

    ++feature_track_id;
  }
}

template <typename T>
double ComputeSetIOU(const std::set<T>& set_1, const std::set<T>& set_2) {
  int num_intersection = 0;
  for (const T& element_1 : set_1) {
    if (theia::ContainsKey(set_2, element_1)) {
      ++num_intersection;
    }
  }

  const int num_union =
      static_cast<int>(set_1.size() + set_2.size()) - num_intersection;
  return static_cast<double>(num_intersection) / num_union;
}

void ExtractObjectSequences(
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    const std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>&
    view_bounding_boxes,
    std::list<ObjectSequencePtr>* object_sequences) {
  CHECK_NOTNULL(object_sequences)->clear();
  CHECK(!feature_tracks.empty());
  CHECK(!view_bounding_boxes.empty());

  // Get start and end indices of the entire view sequence.
  theia::ViewId start_view_id, end_view_id;
  CHECK(GetStartAndEndIndices(feature_tracks, &start_view_id, &end_view_id));

  // FIXME:
  // Manage active sets of feature tracks and object sequences according to
  // the current view ID.

  for (theia::ViewId view_id = start_view_id; view_id < end_view_id;
       ++view_id) {
    const theia::ViewBoundingBoxes* bounding_boxes =
        theia::FindOrNull(view_bounding_boxes, view_id);
    if (bounding_boxes == nullptr) continue;

    for (const auto& bounding_box : *bounding_boxes) {
      const std::string& bounding_box_name = bounding_box.first;

      // Find feature tracks included in the bounding box.
      std::set<theia::TrackId> feature_track_indices;
      GetIncludedFeatureTrackIndices(
          view_id, bounding_box.second, feature_tracks, &feature_track_indices);

      // Find best matched object sequence.
      ObjectSequence* best_matched_object_sequence = nullptr;
      double max_iou_score = FLAGS_min_iou_score_for_overlap;

      for (auto& object_sequence : *object_sequences) {
        const double iou_score = ComputeSetIOU(
            feature_track_indices,
            object_sequence->feature_track_indices_to_match);

        VLOG(3) << "(" << object_sequence->bounding_boxes.back().first << " - "
                << object_sequence->bounding_boxes.back().second << ", "
                << view_id << " - " << bounding_box_name << ") IOU Score: "
                << iou_score;

        if (iou_score > max_iou_score) {
          max_iou_score = iou_score;
          best_matched_object_sequence = object_sequence.get();
        }
      }

      // Create a new one if nothing is matched
      if (best_matched_object_sequence == nullptr) {
        object_sequences->emplace_back(new ObjectSequence);
        best_matched_object_sequence = object_sequences->back().get();
      }

      best_matched_object_sequence->bounding_boxes.emplace_back(
          view_id, bounding_box_name);
      // NOTE:
      // Feature track indices update strategy: Replace all with new ones.
      best_matched_object_sequence->feature_track_indices_to_match =
          feature_track_indices;
    }

    VLOG(3) << "View (" << view_id << "): "
            << object_sequences->size() << " object sequences.";
  }
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
  LOG(INFO) << feature_tracks.size() << " feature track(s) are loaded.";

  // Read bounding boxes.
  std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>
      view_bounding_boxes;
  theia::ReadMultipleBoundingBoxes(
      FLAGS_bounding_boxes_dir, image_filenames, &view_bounding_boxes);

  // Extract object sequences.
  std::list<ObjectSequencePtr> object_sequences;
  ExtractObjectSequences(feature_tracks, view_bounding_boxes,
                         &object_sequences);
  LOG(INFO) << object_sequences.size() << " object sequence(s) are extracted.";


  // Create directory.
  if (!stlplus::folder_exists(FLAGS_output_dir)) {
    CHECK(stlplus::folder_create(FLAGS_output_dir));
  }
  CHECK(stlplus::folder_writable(FLAGS_output_dir));

  // Write object sequences.
  const int num_digits = std::to_string(object_sequences.size()).length();
  int object_id = 0;
  for (const auto& object_sequence : object_sequences) {
    std::stringstream output_dir_sstr;
    output_dir_sstr << FLAGS_output_dir << "/"
                    << std::setfill('0') << std::setw(num_digits) << object_id;

    // Create directory.
    if (!stlplus::folder_exists(output_dir_sstr.str())) {
      CHECK(stlplus::folder_create(output_dir_sstr.str()));
    }
    CHECK(stlplus::folder_writable(output_dir_sstr.str()));

    for (const auto& object_bounding_box : object_sequence->bounding_boxes) {
      const theia::ViewId view_id = object_bounding_box.first;
      const std::string& name = object_bounding_box.second;
      const Eigen::Vector4d& bounding_box = theia::FindOrDie(
          theia::FindOrDie(view_bounding_boxes, view_id), name);

      // Set output filepath to be the same with image filename.
      const std::string image_basename =
          stlplus::basename_part(image_filenames[view_id]);
      const std::string filepath =
          output_dir_sstr.str() + "/" + image_basename + ".txt";
      CHECK(theia::WriteBoundingBox(filepath, bounding_box));
    }

    ++object_id;
  }
}
