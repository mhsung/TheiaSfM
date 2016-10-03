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


// FIXME:
// Move these functions to the other place.
// -- Eigen I/O functions -- //

template<typename Scalar, int Row, int Column>
bool ReadEigenMatrixFromCSV(
    const std::string& _filepath,
    Eigen::Matrix<Scalar, Row, Column>* _matrix,
    const char _delimiter = ',');

template<typename Scalar, int Row, int Column>
bool WriteEigenMatrixToCSV(
    const std::string& _filepath,
    const Eigen::Matrix<Scalar, Row, Column>& _matrix);

template<typename Scalar, int Row, int Column>
bool ReadEigenMatrixFromBinary(
    const std::string& _filepath,
    Eigen::Matrix<Scalar, Row, Column>* _matrix);

template<typename Scalar, int Row, int Column>
bool WriteEigenMatrixToBinary(
    const std::string& _filepath,
    const Eigen::Matrix<Scalar, Row, Column>& _matrix);


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

template<typename Scalar>
Scalar string_to_number(const std::string& _str)
{
  if (_str.size() == 0) return 0;
  std::istringstream sstr(_str);
  Scalar value = 0;
  if (!(sstr >> std::dec >> value)) throw std::invalid_argument(_str);
  return value;
}

template<typename Scalar, int Row, int Column>
bool ReadEigenMatrixFromCSV(
    const std::string& _filepath,
    Eigen::Matrix<Scalar, Row, Column>* _matrix,
    const char _delimiter /*= ','*/)
{
  CHECK(_matrix != nullptr);

  std::ifstream file(_filepath);
  if (!file.good()) {
    LOG(WARNING) << "Can't open the file: '" << _filepath << "'";
    return false;
  }

  typedef std::vector<Scalar> StdVector;
  typedef std::unique_ptr<StdVector> StdVectorPtr;
  typedef std::vector<StdVectorPtr> StdMatrix;
  StdMatrix std_matrix;

  std::string line("");
  int num_rows = 0, num_cols = -1;

  for (; std::getline(file, line); ++num_rows) {
    // Stop reading when the line is blank.
    if (line == "") break;
    std::stringstream sstr(line);
    StdVectorPtr vec(new StdVector);

    std::string token("");
    while (std::getline(sstr, token, _delimiter)) {
      // Stop reading when the token is blank.
      if (token == "") break;
      try {
        const Scalar value = string_to_number<Scalar>(token);
        vec->push_back(value);
      }
      catch (std::exception& e) {
        LOG(WARNING) << "'" << _filepath << "': " << e.what();
        return false;
      }
    }

    if (num_cols >= 0 && num_cols != vec->size()) {
      LOG(WARNING) << "'" << _filepath << "': "
                   << "The number of cols does not match ("
                   << num_cols << " != " << vec->size() << ")";
      return false;
    }

    num_cols = static_cast<int>(vec->size());
    std_matrix.push_back(std::move(vec));
  }

  (*_matrix) = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(
      num_rows, num_cols);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      (*_matrix)(i, j) = (*std_matrix[i])[j];
    }
  }

  file.close();
  return true;
}

template<typename Scalar, int Row, int Column>
bool WriteEigenMatrixToCSV(
    const std::string& _filepath,
    const Eigen::Matrix<Scalar, Row, Column>& _matrix)
{
  std::ofstream file(_filepath);
  if (!file.good()) {
    LOG(WARNING) << "Can't write the file: '" << _filepath << "'";
    return false;
  }

  const Eigen::IOFormat csv_format(
      Eigen::FullPrecision, Eigen::DontAlignCols, ",");
  file << _matrix.format(csv_format);
  file.close();
  return true;
}

template<typename Scalar, int Row, int Column>
bool read_eigen_matrix_from_binary(
    const std::string& _filepath,
    const Eigen::Matrix<Scalar, Row, Column>* _matrix) {
  CHECK(_matrix != nullptr);

  std::ifstream file(_filepath, std::ios::in | std::ios::binary);
  if (!file.good()) {
    LOG(WARNING) << "Can't open the file: '" << _filepath << "'";
    return false;
  }

  int32_t rows = 0, cols = 0;
  file.read((char*)(&rows), sizeof(int32_t));
  file.read((char*)(&cols), sizeof(int32_t));
  _matrix->resize(rows, cols);
  file.read((char *)_matrix->data(), rows*cols*sizeof(Scalar));
  file.close();
  return true;
}

template<typename Scalar, int Row, int Column>
bool WriteEigenMatrixToBinary(
    const std::string& _filepath,
    const Eigen::Matrix<Scalar, Row, Column>& _matrix) {
  std::ofstream file(_filepath,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.good()) {
    LOG(WARNING) << "Can't write the file: '" << _filepath << "'";
    return false;
  }

  int32_t rows = static_cast<int32_t>(_matrix.rows());
  int32_t cols = static_cast<int32_t>(_matrix.cols());
  file.write((char*) (&rows), sizeof(int32_t));
  file.write((char*) (&cols), sizeof(int32_t));
  file.write((char*) _matrix.data(), rows * cols * sizeof(Scalar));
  file.close();
  return true;
}

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
