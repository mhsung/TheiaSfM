// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "theia/sfm/reconstruction_estimator.h"

#include <glog/logging.h>

#include "theia/sfm/incremental_reconstruction_estimator.h"
#include "theia/sfm/global_reconstruction_estimator.h"
#include "theia/sfm/reconstruction_estimator_options.h"
// @mhsung
#include "theia/sfm/exp_global_reconstruction_estimator.h"

namespace theia {

ReconstructionEstimator* ReconstructionEstimator::Create(
    const ReconstructionEstimatorOptions& options) {
  switch (options.reconstruction_estimator_type) {
    case ReconstructionEstimatorType::GLOBAL:
      return new GlobalReconstructionEstimator(options);
      break;
    case ReconstructionEstimatorType::INCREMENTAL:
      return new IncrementalReconstructionEstimator(options);
      break;
    // @mhsung
    case ReconstructionEstimatorType::EXP_GLOBAL:
      return new ExpGlobalReconstructionEstimator(options);
      break;
    default:
      LOG(FATAL) << "Invalid reconstruction estimator specified.";
  }
  return nullptr;
}

}  // namespace theia
