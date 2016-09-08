// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_EXP_GLOBAL_RECONSTRUCTION_ESTIMATOR_H_
#define THEIA_SFM_EXP_GLOBAL_RECONSTRUCTION_ESTIMATOR_H_

#include "theia/sfm/bundle_adjustment/bundle_adjustment.h"
#include "theia/sfm/filter_view_pairs_from_relative_translation.h"
#include "theia/sfm/global_reconstruction_estimator.h"
#include "theia/sfm/reconstruction_estimator.h"
#include "theia/sfm/reconstruction_estimator_options.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/types.h"
#include "theia/solvers/sample_consensus_estimator.h"
#include "theia/util/util.h"
// @mhsung
#include "theia/sfm/object_view_constraint.h"

namespace theia {

class Reconstruction;
class ViewGraph;

class ExpGlobalReconstructionEstimator : public GlobalReconstructionEstimator {
 public:
  ExpGlobalReconstructionEstimator(
      const ReconstructionEstimatorOptions& options);

  ReconstructionEstimatorSummary Estimate(ViewGraph* view_graph,
                                          Reconstruction* reconstruction);

private:
  void FilterInitialOrientations();

  virtual bool EstimateGlobalRotations();
  virtual bool EstimatePosition();

  std::unordered_map<ObjectId, ObjectViewOrientations>
      object_view_orientations_;
  std::unordered_map<ObjectId, ObjectViewPositionDirections>
      object_view_position_directions_;

  DISALLOW_COPY_AND_ASSIGN(ExpGlobalReconstructionEstimator);
};

}  // namespace theia

#endif  // THEIA_SFM_EXP_GLOBAL_RECONSTRUCTION_ESTIMATOR_H_
