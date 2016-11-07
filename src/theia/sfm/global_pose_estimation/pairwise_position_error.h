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
  bool operator()(const T* prev_position,
                  const T* curr_position,
                  const T* next_position,
                  T* residuals) const;

  static ceres::CostFunction* Create(const double weight);

  const double weight_;
};

template <typename T>
bool PairwisePositionError::operator() (const T* prev_position,
                                        const T* curr_position,
                                        const T* next_position,
                                        T* residuals) const {

  // Compute first derivative error.
  // residuals[0] = T(weight_) * (next_position[0] - curr_position[0]);
  // residuals[1] = T(weight_) * (next_position[1] - curr_position[1]);
  // residuals[2] = T(weight_) * (next_position[2] - curr_position[2]);

  // Compute second derivative error.
  residuals[0] = T(weight_) * ((next_position[0] - curr_position[0]) -
                               (curr_position[0] - prev_position[0]));
  residuals[1] = T(weight_) * ((next_position[1] - curr_position[1]) -
                               (curr_position[1] - prev_position[1]));
  residuals[2] = T(weight_) * ((next_position[2] - curr_position[2]) -
                               (curr_position[2] - prev_position[2]));
  return true;
}

}  // namespace theia

#endif  // THEIA_SFM_GLOBAL_POSE_ESTIMATION_PAIRWISE_POSITION_ERROR_H_
