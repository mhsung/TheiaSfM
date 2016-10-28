#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>
#include <algorithm>
#include <set>
#include <vector>


DEFINE_string(reconstruction, "", "Reconstruction file to be viewed.");


void GetViewTrackIdSet(const theia::Reconstruction& reconstruction,
                       const theia::View& view,
                       std::set<theia::TrackId>* track_ids) {
  CHECK_NOTNULL(track_ids)->clear();

  for (const theia::TrackId track_id : view.TrackIds()) {
    const auto* track = reconstruction.Track(track_id);
    if (track == nullptr || !track->IsEstimated()) {
      continue;
    }
    track_ids->insert(track_id);
  }
}

int NumSharedTracks(const theia::Reconstruction& reconstruction,
                    const theia::View& view1, const theia::View& view2) {
  std::set<theia::TrackId> view1_track_ids, view2_track_ids;
  GetViewTrackIdSet(reconstruction, view1, &view1_track_ids);
  GetViewTrackIdSet(reconstruction, view2, &view2_track_ids);

  std::vector<theia::TrackId> intersection;
  std::set_intersection(view1_track_ids.begin(), view1_track_ids.end(),
                        view2_track_ids.begin(), view2_track_ids.end(),
                        std::back_inserter(intersection));
  return intersection.size();
}

void ShowConsecutiveViewInfo(const theia::Reconstruction& reconstruction,
                             const std::vector<theia::ViewId>& view_ids) {
  LOG(INFO) << "== Consecutive view pair stats ==";
  LOG(INFO) << "View name, # shared tracks, distance, angles";

  for (const theia::ViewId view_id : view_ids) {
    if (view_id == 0) continue;

    const auto* view = reconstruction.View(view_id);
    CHECK (view != nullptr && view->IsEstimated());

    // Get previous view.
    const theia::ViewId prev_view_id = view_id - 1;
    const auto* prev_view = reconstruction.View(prev_view_id);
    if (prev_view == nullptr || !prev_view->IsEstimated()) {
      continue;
    }

    const int num_shared_trackes =
        NumSharedTracks(reconstruction, *view, *prev_view);

    const double distance = (view->Camera().GetPosition() -
                             prev_view->Camera().GetPosition()).norm();

    const Eigen::AngleAxisd relative_rotation(
        view->Camera().GetOrientationAsRotationMatrix() *
        prev_view->Camera().GetOrientationAsRotationMatrix().transpose());
    const double angle = relative_rotation.angle() / M_PI * 180.0;

    LOG(INFO) << view->Name() << ", " << num_shared_trackes << ", "
              << distance << ", " << angle;
  }
}

int main(int argc, char* argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Output as a binary file.
  std::unique_ptr<theia::Reconstruction> reconstruction(
      new theia::Reconstruction());
  CHECK(ReadReconstruction(FLAGS_reconstruction, reconstruction.get()))
      << "Could not read reconstruction file.";

  // Centers the reconstruction based on the absolute deviation of 3D points.
  reconstruction->Normalize();


  // Get sorted view IDs.
  std::vector<theia::ViewId> view_ids;
  view_ids.reserve(reconstruction->NumViews());
  for (const theia::ViewId view_id : reconstruction->ViewIds()) {
    const auto* view = reconstruction->View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }
    view_ids.push_back(view_id);
  }
  std::sort(view_ids.begin(), view_ids.end());

  ShowConsecutiveViewInfo(*reconstruction, view_ids);

  reconstruction.release();

  return 0;
}
