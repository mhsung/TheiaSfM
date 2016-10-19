// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_neural_net_output_reader.h"

#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <sstream>
#include <stlplus3/file_system.hpp>

#include "exp_matrix_utils.h"


namespace theia {

bool ReadNeuralNetBBoxesAndOrientations(
  const std::string& bbox_info_filepath,
  const std::string& orientation_filepath,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes) {
  std::list<DetectedBBoxPtr> bboxes;

  // Read bounding box information.
  std::ifstream file(bbox_info_filepath);
  if (!file.good()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream sstr(line);
    std::string token;

    DetectedBBoxPtr bbox(new DetectedBBox);

    CHECK(std::getline(sstr, token, ','));
    bbox->bbox_id_ = std::stoi(token);

    CHECK(std::getline(sstr, token, ','));
    bbox->view_name_ = token;

    CHECK(std::getline(sstr, token, ','));
    bbox->category_id_ = std::stoi(token);

    for (int i = 0; i < 4; i++) {
      CHECK(std::getline(sstr, token, ','));
      bbox->bbox_[i] = std::stod(token);
    }

    CHECK(std::getline(sstr, token, ','));
    bbox->bbox_score_ = std::stod(token);

    CHECK(std::getline(sstr, token, ','));
    bbox->object_id_ = std::stoi(token);

    // Add bbox.
    bboxes.push_back(std::move(bbox));
  }

  file.close();

  const int num_bboxes = bboxes.size();
  LOG(INFO) << "Loaded " << num_bboxes << " bounding box information.";

  // Read orientations.
  Eigen::MatrixXd orientation_matrix(num_bboxes, 4);
  CHECK(ReadEigenMatrixFromCSV(orientation_filepath, &orientation_matrix));

  int bbox_id = 0;
  for (auto it = bboxes.begin(); it != bboxes.end(); ++it, ++bbox_id) {
    (*it)->camera_param_ =
        orientation_matrix.block<1, 3>(bbox_id, 0).transpose();
    (*it)->orientation_score_ = orientation_matrix(bbox_id, 3);
  }
  LOG(INFO) << "Loaded " << num_bboxes << " bounding box orientation.";

  // Collect object bounding boxes.
  for (auto& bbox : bboxes) {
    (*object_bboxes)[bbox->object_id_].push_back(std::move(bbox));
  }
  const int num_object = object_bboxes->size();
  LOG(INFO) << "Loaded " << num_object << " objects.";

  return true;
}

void SortByBBoxIds(
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes,
  std::vector<const DetectedBBox*>* all_bboxes) {
  CHECK_NOTNULL(all_bboxes)->clear();

  int num_all_bboxes = 0;
  for (const auto& object : object_bboxes) {
    num_all_bboxes += object.second.size();
  }
  all_bboxes->resize(num_all_bboxes);

  int count_bboxes = 0;
  for (const auto& object : object_bboxes) {
    for (const auto& bbox : object.second) {
      CHECK_LT(bbox->bbox_id_, num_all_bboxes);
      (*all_bboxes)[bbox->bbox_id_] = bbox.get();
      ++count_bboxes;
    }
  }
  CHECK_EQ(count_bboxes, num_all_bboxes);
}

bool WriteNeuralNetBBoxes(
  const std::string& filepath,
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes) {
  std::vector<const DetectedBBox*> all_bboxes;
  SortByBBoxIds(object_bboxes, &all_bboxes);
  const int num_all_bboxes = all_bboxes.size();

  // Read bounding box information.
  std::ofstream file(filepath);
  if (!file.good()) {
    return false;
  }

  for (const auto& bbox : all_bboxes) {
    CHECK_LT(bbox->bbox_id_, num_all_bboxes);
    file << bbox->bbox_id_ << ","
         << bbox->view_name_ << ","
         << bbox->category_id_ << ",";
    for (int i = 0; i < 4; i++) {
      file << bbox->bbox_[i] << ",";
    }
    file << bbox->bbox_score_ << ","
         << bbox->object_id_ << std::endl;
  }

  file.close();

  LOG(INFO) << "Saved " << num_all_bboxes << " bounding boxes.";

  return true;
}

bool WriteNeuralNetOrientations(
  const std::string& filepath,
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes) {
  std::vector<const DetectedBBox*> all_bboxes;
  SortByBBoxIds(object_bboxes, &all_bboxes);
  const int num_all_bboxes = all_bboxes.size();

  Eigen::MatrixXd orientation_matrix(num_all_bboxes, 4);

  for (const auto& bbox : all_bboxes) {
    CHECK_LT(bbox->bbox_id_, num_all_bboxes);
    orientation_matrix.block<1, 3>(bbox->bbox_id_, 0) =
        bbox->camera_param_.transpose();
    orientation_matrix(bbox->bbox_id_, 3) =
        bbox->orientation_score_;
  }

  if (!WriteEigenMatrixToCSV(filepath, orientation_matrix)) {
    return false;
  }

  LOG(INFO) << "Saved " << num_all_bboxes << " orientations.";

  return true;
}

}