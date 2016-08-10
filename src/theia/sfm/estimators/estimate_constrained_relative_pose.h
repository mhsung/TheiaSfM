// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_ESTIMATORS_ESTIMATE_CONSTRAINED_RELATIVE_POSE_H_
#define THEIA_SFM_ESTIMATORS_ESTIMATE_CONSTRAINED_RELATIVE_POSE_H_

#include <Eigen/Core>
#include <vector>

#include "theia/sfm/create_and_initialize_ransac_variant.h"
#include "theia/sfm/estimators/estimate_relative_pose.h"

namespace theia {

struct FeatureCorrespondence;
struct RansacParameters;
struct RansacSummary;
struct RelativePose;

double RelativeOrientationAbsAngleError(
    const Eigen::Matrix3d& ground_truth_rotation1,
    const Eigen::Matrix3d& ground_truth_rotation2,
    const Eigen::Matrix3d& relative_rotation12);

bool EstimateConstrainedRelativePose(
    const Eigen::Matrix3d& initial_orientation1,
    const Eigen::Matrix3d& initial_orientation2,
    const RansacParameters& ransac_params,
    const RansacType& ransac_type,
    const std::vector<FeatureCorrespondence>& normalized_correspondences,
    RelativePose* relative_pose,
    RansacSummary* ransac_summary);

}  // namespace theia

#endif  // THEIA_SFM_ESTIMATORS_ESTIMATE_CONSTRAINED_RELATIVE_POSE_H_
