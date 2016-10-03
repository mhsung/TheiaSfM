// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <theia/theia.h>
#include <Eigen/Core>

#include <list>
#include <memory>
#include <vector>
#include <unordered_map>


namespace theia {

struct DetectedBBox {
  uint32_t category_id_;
  ObjectId object_id_;
  std::string view_name_;
  Eigen::Vector4d bbox_;    // [x1, y1, x2, y2]
  double score_;
  Eigen::Vector3d camera_param_;
};
typedef std::unique_ptr<DetectedBBox> DetectedBBoxPtr;
typedef std::list<DetectedBBoxPtr> DetectedBBoxPtrList;

bool ReadDetectedBBoxes(
  const std::string& bbox_info_file, const std::string& orientation_file,
  std::unordered_map<ObjectId, DetectedBBoxPtrList>* object_bboxes);

}