// Author: Minhyuk Sung (mhsung@cs.stanford.edu)
// Copied from 'pairwise_translation_error.h'

#ifndef THEIA_SFM_GLOBAL_POSE_ESTIMATION_SINGLE_TRANSLATION_ERROR_H_
#define THEIA_SFM_GLOBAL_POSE_ESTIMATION_SINGLE_TRANSLATION_ERROR_H_

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace theia {

// Computes the error between a translation direction and the direction formed
// from translation such that c_i - scalar * t_i is minimized.
struct SingleTranslationError {
  SingleTranslationError(const Eigen::Vector3d& translation_direction,
                         const double weight);

  // The error is given by the translation error described above.
  template <typename T>
  bool operator()(const T* translation, T* residuals) const;

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& translation_direction, const double weight);

  const Eigen::Vector3d translation_direction_;
  const double weight_;
};

template <typename T>
bool SingleTranslationError::operator() (const T* translation,
                                         T* residuals) const {
  const T kNormTolerance = T(1e-12);

  T norm = sqrt(translation[0] * translation[0] +
                    translation[1] * translation[1] +
                    translation[2] * translation[2]);

  // If the norm is very small then the translations are very close together. In
  // this case, avoid dividing by a tiny number which will cause the weight of
  // the residual term to potentially skyrocket.
  if (T(norm) < kNormTolerance) {
    norm = T(1.0);
  }

  residuals[0] =
      T(weight_) * (translation[0] / norm - T(translation_direction_[0]));
  residuals[1] =
      T(weight_) * (translation[1] / norm - T(translation_direction_[1]));
  residuals[2] =
      T(weight_) * (translation[2] / norm - T(translation_direction_[2]));
  return true;
}

}  // namespace theia

#endif  // THEIA_SFM_GLOBAL_POSE_ESTIMATION_SINGLE_TRANSLATION_ERROR_H_
