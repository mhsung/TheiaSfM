// Copyright (C) 2015 The Regents of the University of California (Regents).
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

#include "theia/sfm/view.h"

#include <string>
#include <unordered_set>
#include <unordered_map>

#include "theia/util/map_util.h"
#include "theia/sfm/camera/camera.h"
#include "theia/sfm/types.h"
#include "theia/sfm/feature.h"
#include "theia/sfm/camera_intrinsics_prior.h"

namespace theia {

// @mhsung
View::View() : name_(""), is_estimated_(false),
               is_orientation_initialized_(false),
               init_orientation_(Eigen::Vector3d::Zero()),
               is_position_dir_initialized_(false),
               init_position_dir_(Eigen::Vector3d::Zero()) {}

// @mhsung
View::View(const std::string& name)
    : name_(name), is_estimated_(false),
      is_orientation_initialized_(false),
      is_position_dir_initialized_(false),
      init_position_dir_(Eigen::Vector3d::Zero()) {}

const std::string& View::Name() const {
  return name_;
}

void View::SetEstimated(bool is_estimated) {
  is_estimated_ = is_estimated;
}

bool View::IsEstimated() const {
  return is_estimated_;
}

const class Camera& View::Camera() const {
  return camera_;
}

class Camera* View::MutableCamera() {
  return &camera_;
}

const struct CameraIntrinsicsPrior& View::CameraIntrinsicsPrior() const {
  return camera_intrinsics_prior_;
}

struct CameraIntrinsicsPrior* View::MutableCameraIntrinsicsPrior() {
  return &camera_intrinsics_prior_;
}

int View::NumFeatures() const {
  return features_.size();
}

std::vector<TrackId> View::TrackIds() const {
  std::vector<TrackId> track_ids;
  track_ids.reserve(features_.size());
  for (const auto& track : features_) {
    track_ids.emplace_back(track.first);
  }
  return track_ids;
}

const Feature* View::GetFeature(const TrackId track_id) const {
  return FindOrNull(features_, track_id);
}

void View::AddFeature(const TrackId track_id, const Feature& feature) {
  features_[track_id] = feature;
}

bool View::RemoveFeature(const TrackId track_id) {
  return features_.erase(track_id) > 0;
}

// @mhsung
void View::SetInitialOrientation(const Eigen::Vector3d& orientation) {
  init_orientation_ = orientation;
  is_orientation_initialized_ = true;
}

// @mhsung
Eigen::Vector3d View::GetInitialOrientation() const {
  CHECK(is_orientation_initialized_);
  return init_orientation_;
}

// @mhsung
void View::RemoveInitialOrientation() {
  init_orientation_.setZero();
  is_orientation_initialized_ = false;
}

// @mhsung
void View::SetInitialPositionDirection(const Eigen::Vector3d& position_dir) {
  init_position_dir_ = position_dir;
  is_position_dir_initialized_ = true;
}

// @mhsung
Eigen::Vector3d View::GetInitialPositionDirection() const {
  CHECK(is_position_dir_initialized_);
  return init_position_dir_;
}

// @mhsung
void View::RemoveInitialPositionDirection() {
  init_position_dir_.setZero();
  is_position_dir_initialized_ = false;
}

}  // namespace theia
