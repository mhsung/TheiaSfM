// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <theia/theia.h>
#include <Eigen/Core>

#include <list>
#include <memory>
#include <vector>
#include <unordered_map>

#include "theia/matching/feature_correspondence.h"


namespace theia {

struct FeatureTrack {
    ViewId start_index_;
    ViewId length_;
    std::vector<Eigen::Vector2d> points_;

    static FeatureTrack* Parse(const std::string& str);
};
typedef std::unique_ptr<FeatureTrack> FeatureTrackPtr;

bool ReadFeatureTracks(const std::string& feature_tracks_file,
                       std::list<FeatureTrackPtr>* feature_tracks);

void GetImageFeaturesFromFeatureTracks(
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    std::unordered_map<ViewId, std::list<Feature> >* image_features);

void GetCorrespodnencesFromFeatureTracks(
    const std::list<theia::FeatureTrackPtr>& feature_tracks,
    std::unordered_map<ViewIdPair, std::list<FeatureCorrespondence> >*
    image_pair_correspondences);

}