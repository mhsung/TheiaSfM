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