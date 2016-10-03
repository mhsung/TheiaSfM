// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <theia/theia.h>
#include <Eigen/Core>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <stlplus3/file_system.hpp>
#include <unordered_map>
#include <vector>

using namespace theia;

// 'CameraParams': Azimuth, elevation, and theta.
//    Used in RenderForCNN (ICCV 2015')
// 'Modelview': OpenGL modelview (world-to-camera) matrix.
// 'CameraMatrix': Camera-to-world matrix used in Microsoft 7-scenes.


// -- I/O functions -- //

// Read OpenGL modelviews.
// Data types: 'pose', 'modelview', 'reconstruction'.
bool ReadModelviews(
    const std::string& data_type, const std::string& filepath,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews);

void ReadModelviewsFromCameraMatrices(
    const std::string& camera_matrix_dir,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews);

void ReadModelviewsFromModelviews(
    const std::string& modelview_dir,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews);

void ReadModelviewsFromReconstruction(
    const std::string& reconstruction_filepath,
    std::unordered_map<std::string, Eigen::Affine3d>* modelviews);


// Read camera orientations.
// The output 'orientations' are Theia camera rotations.
// Data types: 'param', 'pose', 'modelview', 'reconstruction'.
bool ReadOrientations(
    const std::string& data_type, const std::string& filepath,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void ReadOrientationsFromCameraParams(
    const std::string& camera_param_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void ReadOrientationsFromCameraMatrices(
    const std::string& camera_matrix_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void ReadOrientationsFromModelviews(
    const std::string& modelview_dir,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void ReadOrientationsFromReconstruction(
    const std::string& reconstruction_filepath,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);


// Write cameras.
void WriteOrientationsAsCameraParams(
    const std::string& camera_param_dir,
    const std::unordered_map<std::string, Eigen::Matrix3d>& orientations);

void WriteOrientationsAsCameraParams(
    const std::string& camera_param_dir, const Reconstruction& reconstruction);

void WriteModelviews(
    const std::string& modelview_dir,
    const std::unordered_map<std::string, Eigen::Affine3d>& modelviews);

void WriteModelviews(
    const std::string& modelview_dir, const Reconstruction& reconstruction);

void PrintFovy(const int image_height, const Reconstruction& reconstruction);


// -- Utility functions -- //

template<typename T>
void MapViewNamesToIds(
    const Reconstruction& reconstruction,
    const std::unordered_map<std::string, T>& values_with_names,
    std::unordered_map<ViewId, T>* values_with_ids);

template<typename T>
void MapViewIdsToNames(
    const Reconstruction& reconstruction,
    const std::unordered_map<ViewId, T>& values_with_ids,
    std::unordered_map<std::string, T>* values_with_names);

template<typename T>
bool CheckViewNamesValid(
    const std::unordered_map<std::string, T>& values_with_view_names,
    const std::vector<std::string>& image_filenames,
    std::unordered_map<std::string, T>* values_with_image_filenames =
    nullptr);

void GetOrientationsFromReconstruction(
    const Reconstruction& reconstruction,
    std::unordered_map<ViewId, Eigen::Matrix3d>* orientations);

void GetOrientationsFromReconstruction(
    const Reconstruction& reconstruction,
    std::unordered_map<std::string, Eigen::Matrix3d>* orientations);

void SyncOrientationSequences(
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    reference_orientations,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    estimated_orientations,
    std::unordered_map<std::string, Eigen::Matrix3d>*
    synced_estimated_orientations);

void SyncOrientationSequencesWithPivot(
    const std::string& pivot_view_name,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    reference_orientations,
    const std::unordered_map<std::string, Eigen::Matrix3d>&
    estimated_orientations,
    std::unordered_map<std::string, Eigen::Matrix3d>*
    synced_estimated_orientations);

void SyncModelviewSequences(
    const std::unordered_map<std::string, Eigen::Affine3d>&
    reference_modelviews,
    const std::unordered_map<std::string, Eigen::Affine3d>&
    estimated_modelviews,
    std::unordered_map<std::string, Eigen::Affine3d>*
    synced_estimated_modelviews);

bool ReadSequenceIndices(
    const std::string filepath,
    std::unordered_map<std::string, int>* _seq_indices);

void SetCameraIntrinsics(
  const theia::CameraIntrinsicsPrior& camera_intrinsic_prior,
  theia::Camera* camera);

std::unique_ptr<Reconstruction> CreateTheiaReconstructionFromModelviews(
  const std::unordered_map<std::string, Eigen::Affine3d>& modelviews,
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>*
  camera_intrinsics_priors = nullptr);


// -- Template function implementation -- //

template<typename T>
void MapViewNamesToIds(
    const Reconstruction& reconstruction,
    const std::unordered_map<std::string, T>& values_with_names,
    std::unordered_map<ViewId, T>* values_with_ids) {
  CHECK_NOTNULL(values_with_ids);
  values_with_ids->clear();
  values_with_ids->reserve(values_with_names.size());

  for(const auto& value : values_with_names) {
    // FIXME:
    // Now we simply assume that all view name has '.png' extension.
    const auto& view_name = value.first + ".png";
    const ViewId view_id = reconstruction.ViewIdFromName(view_name);
    if (view_id == kInvalidViewId) {
      LOG(WARNING) << "Invalid view name: '" << view_name << "'";
    } else {
      const auto& orientation = value.second;
      (*values_with_ids)[view_id] = orientation;
    }
  }
}

template<typename T>
void MapViewIdsToNames(
    const Reconstruction& reconstruction,
    const std::unordered_map<ViewId, T>& values_with_ids,
    std::unordered_map<std::string, T>* values_with_names) {
  CHECK_NOTNULL(values_with_names);
  values_with_names->clear();
  values_with_names->reserve(values_with_ids.size());

  for(const auto& value : values_with_ids) {
    const ViewId view_id = value.first;
    const View* view = reconstruction.View(view_id);
    if (view == nullptr) {
      LOG(WARNING) << "Invalid view ID: '" << view_id << "'";
    } else {
      const std::string basename = stlplus::basename_part(view->Name());
      const auto& orientation = value.second;
      (*values_with_names)[basename] = orientation;
    }
  }
}

template<typename T>
bool CheckViewNamesValid(
    const std::unordered_map<std::string, T>& values_with_view_names,
    const std::vector<std::string>& image_filenames,
    std::unordered_map<std::string, T>*
    values_with_image_filenames) {
  if (values_with_image_filenames) {
    values_with_image_filenames->reserve(values_with_view_names.size());
  }

  for (const auto& value : values_with_view_names) {
    const std::string& view_name = value.first;

    bool image_exists = false;
    for (const auto& image_filename : image_filenames) {
      std::string image_basename;
      CHECK(theia::GetFilenameFromFilepath(
          image_filename, false, &image_basename));
      if (image_basename == view_name) {

        if (values_with_image_filenames) {
          values_with_image_filenames->emplace(
              image_filename, value.second);
        }

        image_exists = true;
        break;
      }
    }

    if (!image_exists) {
      LOG(WARNING) << "Image '" << view_name << "' does not exist.";
      return false;
    }
  }

  return true;
}
