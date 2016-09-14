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
#include <list>
#include <memory>
#include <stlplus3/file_system.hpp>
#include <string>
#include <sstream>
#include <vector>

#include "applications/exp_bounding_box_utils.h"
#include "applications/exp_drawing_utils.h"
#include "applications/exp_feature_match_utils.h"
#include "applications/exp_feature_track_io.h"


// Input/output files.
DEFINE_string(images_dir, "", "");
DEFINE_string(image_filenames_file, "", "");
DEFINE_string(feature_tracks_file, "", "");
DEFINE_string(input_bounding_boxes_dir, "", "");
DEFINE_string(output_bounding_boxes_dir, "", "");
DEFINE_string(output_bounding_box_images_dir, "", "");
DEFINE_int32(min_num_intersecions_for_overlap, 5, "");
DEFINE_int32(min_num_views_for_sequence, 50, "");
DEFINE_bool(merge_all_matched_objects, false, "");


namespace {
    typedef uint32_t ObjectId;
    struct ObjectSequence {
        std::map<theia::ViewId, std::string> bounding_boxes_;
        std::set<theia::TrackId> feature_track_indices_to_match_;
        std::set<ObjectId> conflicted_object_indices_;
    };
    typedef std::unique_ptr<ObjectSequence> ObjectSequencePtr;
    typedef std::unordered_map<ObjectId, int> NumObjectIntersections;

    // Static variable.
    static ObjectId new_object_id = 0;
}

template <typename T>
int ComputeNumIntersections(
    const std::set<T>& set_1, const std::set<T>&set_2) {
  int num_intersection = 0;
  for (const T& element_1 : set_1) {
    if (theia::ContainsKey(set_2, element_1)) {
      ++num_intersection;
    }
  }
  return num_intersection;
}

template <typename T>
double ComputeIntersectionOverUnion(
    const std::set<T>& set_1, const std::set<T>& set_2) {
  const int num_intersection = ComputeNumIntersections(set_1, set_2);
  const int num_union =
      static_cast<int>(set_1.size() + set_2.size()) - num_intersection;
  return static_cast<double>(num_intersection) / num_union;
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

void GetIncludedFeatureTrackIndices(
    const theia::ViewId& view_id,
    const Eigen::Vector4d& bounding_box,
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    std::set<theia::TrackId>* feature_tracks_indices) {
  CHECK_NOTNULL(feature_tracks_indices)->clear();

  theia::TrackId feature_track_id = 0;

  for (const auto& feature_track : feature_tracks) {
    // Skip if feature track view range does not include the view.
    if (view_id >= feature_track->start_index_ &&
        view_id < feature_track->start_index_ + feature_track->length_) {
      CHECK(feature_track->points_.size() == feature_track->length_);
      const Eigen::Vector2d image_point =
          feature_track->points_[view_id - feature_track->start_index_];

      if (theia::IsPointIncludedInBoundingBox(image_point, bounding_box)) {
        feature_tracks_indices->insert(feature_track_id);
      }
    }
    ++feature_track_id;
  }
}

void FindMaximallyIntersectedBoundingBoxes(
    const std::unordered_map<std::string, std::set<theia::TrackId> >&
    bbox_feature_track_indices,
    const std::unordered_map<ObjectId, ObjectSequencePtr>& objects,
    std::unordered_map<std::string, NumObjectIntersections>*
    bbox_intersected_objects) {
  CHECK_NOTNULL(bbox_intersected_objects)->clear();

  // Make empty list.
  for (const auto& bbox : bbox_feature_track_indices) {
    (*bbox_intersected_objects)[bbox.first] = NumObjectIntersections();
  }

  for (const auto& object : objects) {
    CHECK(object.second != nullptr);
    int max_num_intersections = 0;
    std::string max_bbox_name("");

    for (const auto& bbox : bbox_feature_track_indices) {
      CHECK_NE(bbox.first, "");
      const int num_intersections = ComputeNumIntersections(
          bbox.second, object.second->feature_track_indices_to_match_);

      if (num_intersections >= FLAGS_min_num_intersecions_for_overlap &&
          num_intersections > max_num_intersections) {
        max_num_intersections = num_intersections;
        max_bbox_name = bbox.first;
      }
    }

    if (max_num_intersections > 0) {
      (*bbox_intersected_objects)[max_bbox_name].emplace(
          object.first, max_num_intersections);
    }
  }
}

void RemovedConflictedObjectSequences(
    const std::unordered_map<ObjectId, ObjectSequencePtr>& objects,
    const NumObjectIntersections& intersected_objects,
    std::list<ObjectId>* filtered_intersected_object_indices) {
  CHECK_NOTNULL(filtered_intersected_object_indices)->clear();

  NumObjectIntersections copied_intersected_objects = intersected_objects;
  while (!copied_intersected_objects.empty()) {

    // Find maximally intersected object.
    int max_num_intersections = 0;
    ObjectId max_object_id = 0;
    for (const auto& object : copied_intersected_objects) {
      if (object.second > max_num_intersections) {
        max_num_intersections = object.second;
        max_object_id = object.first;
      }
    }
    if (max_num_intersections == 0) break;
    filtered_intersected_object_indices->push_back(max_object_id);
    const ObjectSequence* max_object =
        theia::FindOrDie(objects, max_object_id).get();

    // Delete the maximally intersected object and its conflicted objects.
    auto it = copied_intersected_objects.begin();
    while (it != copied_intersected_objects.end()) {
      const ObjectId other_object_id = it->first;
      if (other_object_id == max_object_id ||
          theia::ContainsKey(max_object->conflicted_object_indices_,
                             other_object_id)) {
        it = copied_intersected_objects.erase(it);
      } else {
        ++it;
      }
    }
  }
}

ObjectId MergeIntersectedObjectSequence(
    std::list<ObjectId>& intersected_object_indices,
    std::unordered_map<ObjectId, ObjectSequencePtr>* objects) {
  CHECK_NOTNULL(objects);

  // Add new object sequence.
  const ObjectId output_object_id = new_object_id++;
  objects->emplace(output_object_id, ObjectSequencePtr(new ObjectSequence));
  ObjectSequence* output_object =
      theia::FindOrDie(*objects, output_object_id).get();

  for (auto& object_id : intersected_object_indices) {
    const ObjectSequence* object = theia::FindOrDie(*objects, object_id).get();
    // Views in @other must not exist before adding them.
    for (const auto& object_bboxes : object->bounding_boxes_) {
      CHECK(!theia::ContainsKey(output_object->bounding_boxes_,
                                object_bboxes.first));
    }

    // Merge information.
    output_object->bounding_boxes_.insert(
        object->bounding_boxes_.begin(),
        object->bounding_boxes_.end());
    output_object->feature_track_indices_to_match_.insert(
        object->feature_track_indices_to_match_.begin(),
        object->feature_track_indices_to_match_.end());
    output_object->conflicted_object_indices_.insert(
        object->conflicted_object_indices_.begin(),
        object->conflicted_object_indices_.end());

    // Set conflicted object.
    for (const auto& conflicted_object_id :
        object->conflicted_object_indices_) {
      ObjectSequence* conflicted_object =
          theia::FindOrDie(*objects, conflicted_object_id).get();
      CHECK(theia::ContainsKey(
          conflicted_object->conflicted_object_indices_, object_id));
      conflicted_object->conflicted_object_indices_.erase(object_id);
      conflicted_object->conflicted_object_indices_.insert(output_object_id);
    }

    // Delete merged object.
    for(auto it = objects->begin(); it != objects->end(); ++it) {
      if (it->first == object_id) {
        it = objects->erase(it);
        break;
      }
    }
  }

  return output_object_id;
}

void ComputeObjectSequencesInView(
    const theia::ViewId view_id,
    const std::unordered_map<std::string, std::set<theia::TrackId> >&
    bbox_feature_track_indices,
    std::unordered_map<ObjectId, ObjectSequencePtr>* objects) {
  CHECK_NOTNULL(objects);

  std::unordered_map<std::string, NumObjectIntersections>
      bbox_intersected_objects;
  FindMaximallyIntersectedBoundingBoxes(
      bbox_feature_track_indices, *objects, &bbox_intersected_objects);

  std::list<ObjectId> view_object_indices;
  for (auto& bbox : bbox_intersected_objects) {
    auto& intersected_objects = bbox.second;
    std::list<ObjectId> filtered_intersected_object_indices;
    RemovedConflictedObjectSequences(*objects, intersected_objects,
                                     &filtered_intersected_object_indices);

    ObjectId output_object_id;
    if (filtered_intersected_object_indices.size() == 1) {
      output_object_id = filtered_intersected_object_indices.front();
    } else {
      if (FLAGS_merge_all_matched_objects) {
        if (!filtered_intersected_object_indices.empty()) {
          VLOG(3) << "Merge objects:";
          for (auto& object_id : filtered_intersected_object_indices) {
            const ObjectSequence* object =
                theia::FindOrDie(*objects, object_id).get();
            const auto& start_bbox = object->bounding_boxes_.begin();
            VLOG(3) << " - Object: " << object_id << ","
                    << " View: " << start_bbox->first << ","
                    << " BBox: " << start_bbox->second;
          }
        }

        output_object_id = MergeIntersectedObjectSequence
            (filtered_intersected_object_indices, objects);
        VLOG(3) << "Created a new object ("
                << output_object_id << ").";
        VLOG(3) << " - View: " << view_id << ", BBox: " << bbox.first << "";
      } else {
        if (!filtered_intersected_object_indices.empty()) {
          // Choose the maximally intersected one.
          output_object_id = filtered_intersected_object_indices.front();
        } else {
          output_object_id = MergeIntersectedObjectSequence
              (filtered_intersected_object_indices, objects);
          VLOG(3) << "Created a new object ("
                  << output_object_id << ").";
          VLOG(3) << " - View: " << view_id << ", BBox: " << bbox.first << "";
        }
      }
    }

    // Add new bounding box and feature tracks.
    ObjectSequence* output_object =
        theia::FindOrDie(*objects, output_object_id).get();
    output_object->bounding_boxes_.emplace(view_id, bbox.first);
    const auto& feature_track_indices =
        theia::FindOrDie(bbox_feature_track_indices, bbox.first);
    output_object->feature_track_indices_to_match_.insert(
        feature_track_indices.begin(), feature_track_indices.end());

    view_object_indices.push_back(output_object_id);
  }

  // Set conflicted each other for new object sequences.
  for (const auto& object1_id : view_object_indices) {
    ObjectSequence* object1 = theia::FindOrDie(*objects, object1_id).get();
    for (const auto& object2_id : view_object_indices) {
      if (object1_id != object2_id) {
        object1->conflicted_object_indices_.insert(object2_id);
      }
    }
  }
}

void ExtractObjectSequences(
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    const std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>&
    view_bboxes,
    std::unordered_map<ObjectId, ObjectSequencePtr>* objects) {
  CHECK_NOTNULL(objects)->clear();
  CHECK(!feature_tracks.empty());
  CHECK(!view_bboxes.empty());

  // Get start and end indices of the entire view sequence.
  theia::ViewId start_view_id, end_view_id;
  CHECK(theia::GetStartAndEndIndices(
      feature_tracks, &start_view_id, &end_view_id));

  // Iterate from start to end view.
  for (theia::ViewId view_id = start_view_id; view_id < end_view_id;
       ++view_id) {
    const theia::ViewBoundingBoxes* bboxes =
        theia::FindOrNull(view_bboxes, view_id);
    if (bboxes == nullptr || bboxes->empty()) continue;

    // Find feature tracks included in the bounding box.
    std::unordered_map<std::string, std::set<theia::TrackId> >
        bbox_feature_track_indices;

    for (const auto& bbox : *bboxes) {
      const auto& bbox_name = bbox.first;
      std::set<theia::TrackId> feature_track_indices;
      GetIncludedFeatureTrackIndices(
          view_id, bbox.second, feature_tracks, &feature_track_indices);
      bbox_feature_track_indices.emplace(bbox_name, feature_track_indices);
    }

    ComputeObjectSequencesInView(
        view_id, bbox_feature_track_indices, objects);

    VLOG(3) << "View (" << view_id << "): "
            << objects->size() << " object sequences.";
  }
}

void FilterShortObjectSequences(
    std::unordered_map<ObjectId, ObjectSequencePtr>* objects) {
  CHECK_NOTNULL(objects);
  for (auto it = objects->begin(); it != objects->end(); ) {
    const int num_bboxes = it->second->bounding_boxes_.size();
    if (num_bboxes < FLAGS_min_num_views_for_sequence) {
      VLOG(3) << "Object (" << it->first << ") has "
              << num_bboxes << " bounding boxes - Deleted.";
      it = objects->erase(it);
    } else {
      VLOG(3) << "Object (" << it->first << ") has "
              << num_bboxes << " bounding boxes.";
      ++it;
    }
  }
}

void WriteObjectSequences(
    const std::vector<std::string>& image_filenames,
    const std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>&
    view_bboxes,
    const std::unordered_map<ObjectId, ObjectSequencePtr>& objects) {
  // Create directory.
  if (!stlplus::folder_exists(FLAGS_output_bounding_boxes_dir)) {
    CHECK(stlplus::folder_create(FLAGS_output_bounding_boxes_dir));
  }
  CHECK(stlplus::folder_writable(FLAGS_output_bounding_boxes_dir));

  // Write object sequences.
  // const int num_digits = static_cast<int>(
  //    std::to_string(objects.size()).length());

  for (const auto& object : objects) {
    std::stringstream output_dir_sstr;
    output_dir_sstr << FLAGS_output_bounding_boxes_dir << "/"
                    //<< std::setfill('0') << std::setw(num_digits)
                    << object.first;

    // Create directory.
    if (!stlplus::folder_exists(output_dir_sstr.str())) {
      CHECK(stlplus::folder_create(output_dir_sstr.str()));
    }
    CHECK(stlplus::folder_writable(output_dir_sstr.str()));

    for (const auto& object_bboxes : object.second->bounding_boxes_) {
      const theia::ViewId view_id = object_bboxes.first;
      const std::string& name = object_bboxes.second;
      const Eigen::Vector4d& bounding_box = theia::FindOrDie(
          theia::FindOrDie(view_bboxes, view_id), name);

      // Set output filepath to be the same with image filename.
      const std::string image_basename =
          stlplus::basename_part(image_filenames[view_id]);
      const std::string filepath =
          output_dir_sstr.str() + "/" + image_basename + ".txt";
      CHECK(theia::WriteBoundingBox(filepath, bounding_box));
    }
  }
}

void GetViewBoundingBoxesWithObjectIndices(
    const std::list<ObjectSequencePtr>& objects,
    const std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>&
    view_bounding_boxes,
    std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>*
    view_bounding_boxes_with_object_indices) {
  CHECK_NOTNULL(view_bounding_boxes_with_object_indices)->clear();

  int object_id = 0;
  for (const auto& object : objects) {
    const std::string object_id_str = std::to_string(object_id);

    for (const auto& object_bounding_box : object->bounding_boxes_) {
      const theia::ViewId view_id = object_bounding_box.first;
      const std::string& name = object_bounding_box.second;
      const Eigen::Vector4d& bounding_box = theia::FindOrDie(
          theia::FindOrDie(view_bounding_boxes, view_id), name);

      // Use object index as bounding box name.
      (*view_bounding_boxes_with_object_indices)[view_id][object_id_str] =
          bounding_box;
    }

    ++object_id;
  }
}

void DrawObjectSequences(
    const std::vector<std::string>& image_filenames,
    const std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>&
    view_bounding_boxes,
    const std::list<ObjectSequencePtr>& object_sequences) {
  std::unordered_map<theia::ViewId, theia::ViewBoundingBoxes>
      view_bounding_boxes_with_object_indices;
  GetViewBoundingBoxesWithObjectIndices(
      object_sequences, view_bounding_boxes,
      &view_bounding_boxes_with_object_indices);

  // Create directory.
  if (!stlplus::folder_exists(FLAGS_output_bounding_box_images_dir)) {
    CHECK(stlplus::folder_create(FLAGS_output_bounding_box_images_dir));
  }
  CHECK(stlplus::folder_writable(FLAGS_output_bounding_box_images_dir));

  for (const auto& bounding_boxes : view_bounding_boxes_with_object_indices) {
    const theia::ViewId view_id = bounding_boxes.first;
    const std::string image_filename = image_filenames[view_id];
    const std::string input_image_filepath =
        FLAGS_images_dir + "/" + image_filename;
    CHECK(theia::FileExists(input_image_filepath))
    << "Image file does not exist: '" << input_image_filepath << "'";
    const theia::FloatImage image(input_image_filepath);

    theia::ImageCanvas canvas;
    canvas.AddImage(image);

    for (const auto& bounding_box : bounding_boxes.second) {
      // Assume that bounding box name is object ID.
      const int object_id = std::stoi(bounding_box.first);

      // Set bounding box color based on object index.
      const theia::RGBPixel color =
          theia::LabelColor(static_cast<uint32_t>(object_id));
      theia::DrawBox(bounding_box.second, color, &canvas);
    }

    const std::string output_image_filepath =
        FLAGS_output_bounding_box_images_dir + "/" + image_filename;
    canvas.Write(output_image_filepath);
    VLOG(3) << "Saved '" << output_image_filepath << "'.";
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
      FLAGS_input_bounding_boxes_dir, image_filenames, &view_bounding_boxes);

  // Extract object sequences.
  std::unordered_map<ObjectId, ObjectSequencePtr> objects;
  ExtractObjectSequences(feature_tracks, view_bounding_boxes, &objects);
  LOG(INFO) << objects.size() << " object sequence(s) are extracted.";

  // Filter too short object sequences.
  FilterShortObjectSequences(&objects);
  LOG(INFO) << objects.size() << " object sequence(s) remain after "
      "filtering.";

  // Write object sequences.
  WriteObjectSequences(image_filenames, view_bounding_boxes, objects);

  // Draw object bounding boxes in images.
  // DrawObjectSequences(image_filenames, view_bounding_boxes,
  // object_sequences);
}
