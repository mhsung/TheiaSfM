// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include "theia/sfm/estimators/estimate_constrained_relative_pose.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <limits>
#include <memory>
#include <vector>

#include "theia/matching/feature_correspondence.h"
#include "theia/sfm/create_and_initialize_ransac_variant.h"
#include "theia/sfm/estimators/estimate_relative_pose.h"
#include "theia/sfm/pose/essential_matrix_utils.h"
#include "theia/sfm/pose/five_point_relative_pose.h"
#include "theia/sfm/pose/util.h"
#include "theia/sfm/triangulation/triangulation.h"
#include "theia/solvers/estimator.h"
#include "theia/solvers/sample_consensus_estimator.h"
#include "theia/util/util.h"

namespace theia {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;

// An estimator for computing the relative pose from 5 feature
// correspondences. The feature correspondences should be normalized
// by the focal length with the principal point at (0, 0).
class ConstrainedRelativePoseEstimator
    : public Estimator<FeatureCorrespondence, RelativePose> {
 public:
  // @mhsung
  ConstrainedRelativePoseEstimator(
      const Eigen::Matrix3d& initial_orientation1,
      const Eigen::Matrix3d& initial_orientation2)
  : initial_orientation1_(initial_orientation1),
    initial_orientation2_(initial_orientation2) { }

  // 5 correspondences are needed to determine an essential matrix and thus a
  // relative pose.
  double SampleSize() const { return 5; }

  // Estimates candidate relative poses from correspondences.
  bool EstimateModel(const std::vector<FeatureCorrespondence>& correspondences,
                     std::vector<RelativePose>* relative_poses) const {
    std::vector<Eigen::Vector2d> image1_points, image2_points;
    image1_points.reserve(correspondences.size());
    image2_points.reserve(correspondences.size());
    for (int i = 0; i < correspondences.size(); i++) {
      image1_points.emplace_back(correspondences[i].feature1);
      image2_points.emplace_back(correspondences[i].feature2);
    }

    std::vector<Matrix3d> essential_matrices;
    if (!FivePointRelativePose(image1_points,
                               image2_points,
                               &essential_matrices)) {
      return false;
    }

    relative_poses->reserve(essential_matrices.size() * 4);
    for (const Eigen::Matrix3d& essential_matrix : essential_matrices) {
      RelativePose relative_pose;
      relative_pose.essential_matrix = essential_matrix;

      // The best relative pose decomposition should have at least 4
      // triangulated points in front of the camera. This is because one point
      // may be at infinity.
      const int num_points_in_front_of_cameras = GetBestPoseFromEssentialMatrix(
          essential_matrix,
          correspondences,
          &relative_pose.rotation,
          &relative_pose.position);

      // @mhsung
      if (!SatisfiesConstraint(relative_pose)) {
        VLOG(2) << "Failed to satisfy initial orientation constraints.";
        continue;
      }

      if (num_points_in_front_of_cameras >= 4) {
        relative_poses->push_back(relative_pose);
      }
    }
    return relative_poses->size() > 0;
  }

  // The error for a correspondences given a model. This is the squared sampson
  // error.
  double Error(const FeatureCorrespondence& correspondence,
               const RelativePose& relative_pose) const {
    if (IsTriangulatedPointInFrontOfCameras(correspondence,
                                            relative_pose.rotation,
                                            relative_pose.position)) {
      return SquaredSampsonDistance(relative_pose.essential_matrix,
                                    correspondence.feature1,
                                    correspondence.feature2);
    }
    return std::numeric_limits<double>::max();
  }

  // @mhsung
  bool SatisfiesConstraint(const RelativePose& relative_pose) const {
    // FIXME:
    const double kErrorTol = 30.0;
    const double relative_rotation_angle_error =
        RelativeOrientationAbsAngleError(
            initial_orientation1_, initial_orientation2_,
            relative_pose.rotation);
    if (relative_rotation_angle_error > kErrorTol) {
      return false;
    }

    return true;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(ConstrainedRelativePoseEstimator);

  // @mhsung
  const Eigen::Matrix3d initial_orientation1_;
  const Eigen::Matrix3d initial_orientation2_;
};

}  // namespace

double RelativeOrientationAbsAngleError(
    const Eigen::Matrix3d& ground_truth_rotation1,
    const Eigen::Matrix3d& ground_truth_rotation2,
    const Eigen::Matrix3d& relative_rotation12) {
  const Eigen::Matrix3d relative_rotation_error =
      ground_truth_rotation2.transpose() * relative_rotation12 *
      ground_truth_rotation1;

  double relative_rotation_angle_error =
      Eigen::AngleAxisd(relative_rotation_error).angle() / M_PI * 180.0;

  // Make the angle error to be in [-180, +180] range.
  while (relative_rotation_angle_error > +180.0) {
    relative_rotation_angle_error -= 360.0;
  }
  while (relative_rotation_angle_error < -180.0) {
    relative_rotation_angle_error += 360.0;
  }

  // Return the absolute angle error.
  return std::abs(relative_rotation_angle_error);
}

bool EstimateConstrainedRelativePose(
    const Eigen::Matrix3d& initial_orientation1,
    const Eigen::Matrix3d& initial_orientation2,
    const RansacParameters& ransac_params,
    const RansacType& ransac_type,
    const std::vector<FeatureCorrespondence>& normalized_correspondences,
    RelativePose* relative_pose,
    RansacSummary* ransac_summary) {
  // @mhsung
  ConstrainedRelativePoseEstimator relative_pose_estimator(
      initial_orientation1, initial_orientation2);
  std::unique_ptr<SampleConsensusEstimator<
      ConstrainedRelativePoseEstimator> > ransac =
      CreateAndInitializeRansacVariant(ransac_type,
                                       ransac_params,
                                       relative_pose_estimator);
  // Estimate the relative pose.
  return ransac->Estimate(normalized_correspondences,
                          relative_pose,
                          ransac_summary);

  CHECK(relative_pose_estimator.SatisfiesConstraint(*relative_pose));
}

}  // namespace theia
