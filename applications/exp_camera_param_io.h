// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <theia/theia.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace theia;


// Read Cameras.
// The output 'orientations' are Theia camera rotations.
// Data types: 'param', 'modelview', or 'reconstruction'.
bool ReadOrientations(
    const std::string& data_type, const std::string& filepath,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void ReadOrientationsFromCameraParams(
    const std::string& camera_param_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void ReadOrientationsFromModelviews(
    const std::string& modelview_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void ReadOrientationsFromReconstruction(
    const std::string& reconstruction_filepath,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);


// Write Cameras.
void WriteOrientationsAsCameraParams(
    const std::string& camera_param_dir,
    const std::unordered_map<std::string, Eigen::Matrix3d>& orientations);

void WriteOrientationsAsCameraParams(
    const std::string& camera_param_dir, const Reconstruction& reconstruction);

void WriteModelviews(
    const std::string& modelview_dir, const Reconstruction& reconstruction);

void PrintFovy(const int image_height, const Reconstruction& reconstruction);


void GetOrientationsFromReconstruction(
    const Reconstruction& reconstruction,
    std::unordered_map<ViewId, Eigen::Matrix3d>* orientations);

void MapOrientationsToViewIds(
    const Reconstruction& reconstruction,
    const std::unordered_map<std::string, Eigen::Matrix3d>& name_orientations,
    std::unordered_map<ViewId, Eigen::Matrix3d>* id_orientations);

void MapOrientationsToViewNames(
    const Reconstruction& reconstruction,
    const std::unordered_map<ViewId, Eigen::Matrix3d>& id_orientations,
    std::unordered_map<std::string, Eigen::Matrix3d>* name_orientations);

void ComputeRelativeOrientationsFromFirstFrame(
    const std::unordered_map<ViewId, Eigen::Matrix3d>& orientations,
    std::unordered_map<ViewId, Eigen::Matrix3d>* relative_orientations);

void SyncOrientationSequences(
    const std::unordered_map<ViewId, Eigen::Matrix3d>&
    reference_orientations,
    const std::unordered_map<ViewId, Eigen::Matrix3d>&
    estimated_orientations,
    std::unordered_map<ViewId, Eigen::Matrix3d>*
    synced_estimated_orientations);
