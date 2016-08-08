// Copyright (C) 2013 The Regents of the University of California (Regents).
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

#include <ceres/rotation.h>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <stlplus3/file_system.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "applications/command_line_helpers.h"
#include "theia/image/image.h"
#include "theia/image/image_canvas.h"
#include "exp_camera_param_utils.h"
#include "exp_camera_param_io.h"

DEFINE_string(images_dir, "", "Directory including image files.");
DEFINE_string(matches_file, "", "Filename of the matches file.");
DEFINE_string(selected_image_name, "frame-000425", "");
DEFINE_string(output_dir, "", "Output directory to store feature match "
    "drawing images.");


void DrawAllMatchedFeatures(
    const std::vector<theia::ImagePairMatch>& image_matches) {
  for (const auto& match : image_matches) {
    const std::string basename_1 = stlplus::basename_part(match.image1);
    const std::string basename_2 = stlplus::basename_part(match.image2);

    if (FLAGS_selected_image_name != "" &&
        FLAGS_selected_image_name != basename_1 &&
        FLAGS_selected_image_name != basename_2) {
      continue;
    }

    const std::string image_filepath1 = FLAGS_images_dir + "/" + match.image1;
    const std::string image_filepath2 = FLAGS_images_dir + "/" + match.image2;

    CHECK(stlplus::file_exists(image_filepath1))
    << "Image file does not exist: '" << image_filepath1 << "'";
    CHECK(stlplus::file_exists(image_filepath2))
    << "Image file does not exist: '" << image_filepath2 << "'";

    const theia::FloatImage image1(image_filepath1);
    const theia::FloatImage image2(image_filepath2);

    theia::ImageCanvas canvas;
    const int canvas_index1 = canvas.AddImage(image1);
    const int canvas_index2 = canvas.AddImage(image2);
    canvas.DrawMatchedFeatures(
        canvas_index1, canvas_index2, match.correspondences);

    const std::string output_filepath =
        FLAGS_output_dir + "/" + basename_1 + "_" + basename_2 + ".png";
    canvas.Write(output_filepath);
    LOG(INFO) << "Saved '" << output_filepath << "'.";
  }

  LOG(INFO) << "Total " << image_matches.size() << " pairs.";
}

int main(int argc, char *argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Load matches from file.
  std::vector<std::string> image_files;
  std::vector<theia::CameraIntrinsicsPrior> camera_intrinsics_prior;
  std::vector<theia::ImagePairMatch> image_matches;

  // Read in match file.
  CHECK(theia::ReadMatchesAndGeometry(
      FLAGS_matches_file, &image_files, &camera_intrinsics_prior,
      &image_matches));

  // Empty directory.
  if (stlplus::folder_exists(FLAGS_output_dir)) {
    CHECK(stlplus::folder_delete(FLAGS_output_dir, true));
  }
  CHECK(stlplus::folder_create(FLAGS_output_dir));
  CHECK(stlplus::folder_writable(FLAGS_output_dir));

  DrawAllMatchedFeatures(image_matches);
}
