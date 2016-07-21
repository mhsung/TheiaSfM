#include <Eigen/Core>
#include <theia/theia.h>
#include <vector>


// Read Cameras.
void ReadRotationsFromCameraParams(
    const std::string& camera_param_dir,
    const std::vector<std::string>& image_names,
    const theia::Reconstruction& ref_reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations);

void ReadRotationsFromModelviews(
    const std::string& modelview_dir,
    const std::vector<std::string>& image_names,
    const theia::Reconstruction& ref_reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations);

void ReadRotationsFromReconstruction(
    const std::string& target_reconstruction_filepath,
    const std::vector<std::string>& image_names,
    const theia::Reconstruction& ref_reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations);

void GetRotationsFromReconstruction(
    const theia::Reconstruction& reconstruction,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* modelview_rotations);

void ComputeRelativeRotationFromFirstFrame(
    const std::unordered_map<theia::ViewId, Eigen::Matrix3d>& ref_rotations,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* relative_rotations);

void SyncRotationLists(
    const std::unordered_map<theia::ViewId, Eigen::Matrix3d>& ref_rotations,
    const std::unordered_map<theia::ViewId, Eigen::Matrix3d>& est_rotations,
    std::unordered_map<theia::ViewId, Eigen::Matrix3d>* synced_est_rotations);

// Write camera information.
void WriteFovy(const theia::Reconstruction& reconstruction);

void WriteModelviews(
    const theia::Reconstruction& reconstruction,
    const std::string& output_dir);