// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#ifndef THEIA_SFM_GLOBAL_POSE_ESTIMATION_PAIRWISE_POSITION_ERROR_H_
#define THEIA_SFM_GLOBAL_POSE_ESTIMATION_PAIRWISE_POSITION_ERROR_H_

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace theia {

struct PairwisePositionError {
  PairwisePositionError(const double weight);

  // The error is given by the position error described above.
  template <typename T>
  bool operator()(const T* position1, const T* position2, T* residuals) const;

  static ceres::CostFunction* Create(const double weight);

  const double weight_;
};

template <typename T>
bool PairwisePositionError::operator() (const T* position1,
                                        const T* position2,
                                        T* residuals) const {

  residuals[0] = T(weight_) * (position2[0] - position1[0]);
  residuals[1] = T(weight_) * (position2[1] - position1[1]);
  residuals[2] = T(weight_) * (position2[2] - position1[2]);
  return true;
}

}  // namespace theia

#endif  // THEIA_SFM_GLOBAL_POSE_ESTIMATION_PAIRWISE_POSITION_ERROR_H_
