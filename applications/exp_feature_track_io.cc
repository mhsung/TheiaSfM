// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_feature_track_io.h"

#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <sstream>
#include <stlplus3/file_system.hpp>


namespace theia {

FeatureTrack* FeatureTrack::Parse(const std::string& str) {
  FeatureTrack* feature_tracks = new FeatureTrack;
  std::stringstream sstr(str);
  sstr >> feature_tracks->start_index_;
  sstr >> feature_tracks->length_;
  CHECK_GE(feature_tracks->start_index_, 0);
  CHECK_GT(feature_tracks->length_, 1);
  feature_tracks->points_.reserve(feature_tracks->length_);
  for (int i = 0; i < feature_tracks->length_; ++i) {
    double x, y;
    sstr >> x;
    sstr >> y;
    CHECK(!sstr.eof());
    feature_tracks->points_.push_back(Eigen::Vector2d(x, y));
  }
  return feature_tracks;
}

bool ReadFeatureTracks(const std::string& feature_tracks_file,
                       std::list<theia::FeatureTrackPtr>* feature_tracks) {
  CHECK_NOTNULL(feature_tracks)->clear();

  std::ifstream tracks_reader(feature_tracks_file);
  if (!tracks_reader.is_open()) {
    LOG(ERROR) << "Could not open the feature tracks file: "
               << feature_tracks_file << " for reading.";
    return false;
  }

  std::string line;
  while (std::getline(tracks_reader, line)) {
    feature_tracks->emplace_back(FeatureTrack::Parse(line));
  }

  return true;
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

void GetImageFeaturesFromFeatureTracks(
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    std::unordered_map<ViewId, std::list<Feature> >* image_features) {
  CHECK_NOTNULL(image_features)->clear();

  for (const FeatureTrackPtr& feature_track : feature_tracks) {
    for (ViewId i = 0; i < feature_track->length_; ++i) {
      const Eigen::Vector2d feature = feature_track->points_[i];
      const ViewId view_id = feature_track->start_index_ + i;
      (*image_features)[view_id].push_back(feature);
    }
  }
}

void GetCorrespodnencesFromFeatureTracks(
    const std::list<FeatureTrackPtr>& feature_tracks,
    std::unordered_map<theia::ViewIdPair,
        std::list<theia::FeatureCorrespondence> >* image_pair_correspondences) {
  CHECK_NOTNULL(image_pair_correspondences)->clear();

  for (const FeatureTrackPtr& feature_track : feature_tracks) {
    // For all pairs in frames in the track.
    for (ViewId i1 = 0; i1 < feature_track->length_ - 1; ++i1) {
      const Eigen::Vector2d feature1 = feature_track->points_[i1];
      const ViewId view1_id = feature_track->start_index_ + i1;

      for (ViewId i2 = i1 + 1; i2 < feature_track->length_; ++i2) {
        const Eigen::Vector2d feature2 = feature_track->points_[i2];
        const ViewId view2_id = feature_track->start_index_ + i2;

        (*image_pair_correspondences)[std::make_pair(view1_id, view2_id)]
            .push_back(FeatureCorrespondence(feature1, feature2));
      }
    }
  }
}

}