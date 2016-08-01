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

  DISALLOW_COPY_AND_ASSIGN(ExpGlobalReconstructionEstimator);
};

}  // namespace theia

#endif  // THEIA_SFM_EXP_GLOBAL_RECONSTRUCTION_ESTIMATOR_H_
