// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <theia/theia.h>
#include <Eigen/Geometry>
#include <vector>


namespace theia {

void GetPutativeMatches(
    const std::string& image1_name, const std::string& image2_name,
    const std::list<theia::FeatureCorrespondence>& correspondences,
    theia::KeypointsAndDescriptors* features1,
    theia::KeypointsAndDescriptors* features2,
    std::vector<theia::IndexedFeatureMatch>* putative_matches);

void CreateMatchesFromCorrespondences(
    std::unordered_map<ViewIdPair, std::list<theia::FeatureCorrespondence> >&
    image_pair_correspondences,
    const std::vector<std::string>& image_filenames,
    const std::vector<theia::CameraIntrinsicsPrior>& intrinsics,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    initial_orientations,
    const theia::FeatureMatcherOptions& options,
    std::vector<theia::ImagePairMatch>* matches);
}