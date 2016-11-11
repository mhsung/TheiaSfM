// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <theia/theia.h>
#include <Eigen/Core>

#include <list>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>


namespace theia {

struct DetectedBBox {
  uint32_t category_id_;
  ObjectId object_id_;
  uint32_t bbox_id_;
  std::string view_name_;
  Eigen::Vector4d bbox_;    // [x1, y1, x2, y2]
  double bbox_score_;
  Eigen::Vector3d camera_param_;
  double orientation_score_;

  DetectedBBox()
      : category_id_(0),
        object_id_(0),
        bbox_id_(0),
        view_name_(""),
        bbox_(Eigen::Vector4d::Zero()),
        bbox_score_(0.0),
        camera_param_(Eigen::Vector3d::Zero()),
        orientation_score_(0.0) {
  }

  DetectedBBox(const DetectedBBox& other)
      : category_id_(other.category_id_),
        object_id_(other.object_id_),
        bbox_id_(other.bbox_id_),
        view_name_(other.view_name_),
        bbox_(other.bbox_),
        bbox_score_(other.bbox_score_),
        camera_param_(other.camera_param_),
        orientation_score_(other.orientation_score_) {
  }

  Eigen::Vector2d bbox_center() const {
    return Eigen::Vector2d (
      0.5 * (bbox_[0] + bbox_[2]), 0.5 * (bbox_[1] + bbox_[3]));
  }
};

typedef std::unique_ptr<DetectedBBox> DetectedBBoxPtr;
typedef std::list<DetectedBBoxPtr> DetectedBBoxPtrList;

// NOTE:
// Do not read orientations if orientation_filepath == "".
bool ReadNeuralNetBBoxesAndOrientations(
  const std::string& bbox_info_filepath,
  const std::string& orientation_filepath,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes);

bool GetObjectRelatedViewNames(
    const std::string& bbox_info_filepath,
    std::set<std::string>* view_names);

void SortByBBoxIds(
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes,
  std::vector<const DetectedBBox*>* all_bboxes);

bool WriteNeuralNetBBoxes(
  const std::string& filepath,
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes);

bool WriteNeuralNetOrientations(
  const std::string& filepath,
  const std::unordered_map<ObjectId, DetectedBBoxPtrList>& object_bboxes);

}