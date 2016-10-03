// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


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
bool ReadEigenMatrixFromBinary(
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
