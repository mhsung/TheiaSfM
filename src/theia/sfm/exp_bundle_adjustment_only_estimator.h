// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'global_reconstruction_estimator.h'

#ifndef THEIA_SFM_EXP_BUNDLE_ADJUSTMENT_ONLY_ESTIMATOR_H_
#define THEIA_SFM_EXP_BUNDLE_ADJUSTMENT_ONLY_ESTIMATOR_H_

#include "theia/sfm/bundle_adjustment/bundle_adjustment.h"
#include "theia/sfm/filter_view_pairs_from_relative_translation.h"
#include "theia/sfm/reconstruction_estimator.h"
#include "theia/sfm/reconstruction_estimator_options.h"
#include "theia/sfm/twoview_info.h"
#include "theia/sfm/types.h"
#include "theia/solvers/sample_consensus_estimator.h"
#include "theia/util/util.h"

namespace theia {

class Reconstruction;
class ViewGraph;

class ExpBundleAdjustmentOnlyEstimator : public ReconstructionEstimator {
 public:
  ExpBundleAdjustmentOnlyEstimator(
      const ReconstructionEstimatorOptions& options);

  ReconstructionEstimatorSummary Estimate(ViewGraph* view_graph,
                                          Reconstruction* reconstruction);

 private:
  bool FilterInitialViewGraph();
  void CalibrateCameras();
  void EstimateStructure();
  bool BundleAdjustment();

  ViewGraph* view_graph_;
  Reconstruction* reconstruction_;

  ReconstructionEstimatorOptions options_;
  BundleAdjustmentOptions bundle_adjustment_options_;

  DISALLOW_COPY_AND_ASSIGN(ExpBundleAdjustmentOnlyEstimator);
};

}  // namespace theia

#endif  // #ifndef THEIA_SFM_EXP_BUNDLE_ADJUSTMENT_ONLY_ESTIMATOR_H_
