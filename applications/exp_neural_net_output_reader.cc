// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "exp_neural_net_output_reader.h"

#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <sstream>
#include <stlplus3/file_system.hpp>

#include "exp_matrix_utils.h"


namespace theia {

bool ReadDetectedBBoxes(
  const std::string& bbox_info_file, const std::string& orientation_file,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes) {
  std::list<DetectedBBoxPtr> bboxes;

  // Read bounding box information.
  std::ifstream file(bbox_info_file);
  if (!file.good()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream sstr(line);
    std::string token;

    DetectedBBoxPtr bbox(new DetectedBBox);

    CHECK(std::getline(sstr, token, ','));
    // const int bbox_id = std::stoi(token);

    CHECK(std::getline(sstr, token, ','));
    bbox->view_name_ = token;

    CHECK(std::getline(sstr, token, ','));
    bbox->category_id_ = std::stoi(token);

    for (int i = 0; i < 4; i++) {
      CHECK(std::getline(sstr, token, ','));
      bbox->bbox_[i] = std::stod(token);
    }

    CHECK(std::getline(sstr, token, ','));
    bbox->score_ = std::stod(token);

    CHECK(std::getline(sstr, token, ','));
    bbox->object_id_ = std::stoi(token);

    // Add bbox.
    bboxes.push_back(std::move(bbox));
  }

  const int num_bboxs = bboxes.size();
  LOG(INFO) << "Loaded " << num_bboxs << " bounding box information.";

  // Read orientations.
  Eigen::MatrixXi orientation_matrix(num_bboxs, 3);
  CHECK(ReadEigenMatrixFromCSV(orientation_file, &orientation_matrix));

  int bbox_id = 0;
  for (auto it = bboxes.begin(); it != bboxes.end(); ++it, ++bbox_id) {
    (*it)->camera_param_ = orientation_matrix.row(bbox_id).cast<double>();
  }
  LOG(INFO) << "Loaded " << num_bboxs << " bounding box orientation.";

  // Collect object bounding boxes.
  for (auto& bbox : bboxes) {
    (*object_bboxes)[bbox->object_id_].push_back(std::move(bbox));
  }
  const int num_object = object_bboxes->size();
  LOG(INFO) << "Loaded " << num_object << " objects.";
}

}
