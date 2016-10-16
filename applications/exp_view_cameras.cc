// Author: Minhyuk Sung (mhsung@cs.stanford.edu)

#include <Eigen/Core>
#include <glog/logging.h>
#include <gflags/gflags.h>
// @mhsung
#include <iomanip>
#include <theia/theia.h>
// @mhsung
#include <theia/image/image.h>
// @mhsung
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#ifdef FREEGLUT
#include <GL/freeglut.h>
#else  // FREEGLUT
#include <GLUT/glut.h>
#endif  // FREEGLUT
#else  // __APPLE__
#ifdef _WIN32
#include <windows.h>
#include <GL/glew.h>
#include <GL/glut.h>
#else  // _WIN32
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif  // _WIN32
#endif  // __APPLE__

#include "exp_camera_param_utils.h"
#include "exp_camera_param_io.h"

DEFINE_string(data_type_list, "", "comma-seperated.");
DEFINE_string(filepath_list, "", "comma-seperated.");
DEFINE_string(calibration_file, "",
              "Calibration file containing image calibration data.");
DEFINE_string(snapshot_file, "", "");
DEFINE_double(robust_alignment_threshold, 0.0,
              "If greater than 0.0, this threshold sets determines inliers for "
                "RANSAC alignment of reconstructions. The inliers are then used "
                "for a least squares alignment.");

// Containers for the data.
std::vector< std::vector<theia::Camera> > cameras_list;
std::vector<Eigen::Vector3d> world_points;
std::vector<Eigen::Vector3f> point_colors;
std::vector<int> num_views_for_track;

// Parameters for OpenGL.
int width = 1200;
int height = 800;

// OpenGL camera parameters.
Eigen::Vector3f viewer_position(0.0, 0.0, 0.0);
float zoom = -200.0;
float delta_zoom = 1.1;

// Rotation values for the navigation
Eigen::Vector2f navigation_rotation(0.0, 0.0);

// Position of the mouse when pressed
int mouse_pressed_x = 0, mouse_pressed_y = 0;
float last_x_offset = 0.0, last_y_offset = 0.0;
// Mouse button states
int left_mouse_button_active = 0, right_mouse_button_active = 0;

// Visualization parameters.
bool draw_axes = false;
float point_size = 1.0;
float normalized_focal_length = 1.0;
int min_num_views_for_track = 3;
double anti_aliasing_blend = 0.01;

void GetPerspectiveParams(double* aspect_ratio, double* fovy) {
  double focal_length = 800.0;
  *aspect_ratio = static_cast<double>(width) / static_cast<double>(height);
  *fovy = 2 * atan(height / (2.0 * focal_length)) * 180.0 / M_PI;
}

void ChangeSize(int w, int h) {
  // Prevent a divide by zero, when window is too short
  // (you cant make a window of zero width).
  if (h == 0) h = 1;

  width = w;
  height = h;

  // Use the Projection Matrix
  glMatrixMode(GL_PROJECTION);

  // Reset Matrix
  glLoadIdentity();

  // Set the viewport to be the entire window
  double aspect_ratio, fovy;
  GetPerspectiveParams(&aspect_ratio, &fovy);
  glViewport(0, 0, w, h);

  // Set the correct perspective.
  gluPerspective(fovy, aspect_ratio, 0.001f, 100000.0f);

  // Get Back to the Reconstructionview
  glMatrixMode(GL_MODELVIEW);
}

void DrawAxes(float length) {
  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glDisable(GL_LIGHTING);
  glLineWidth(5.0);
  glBegin(GL_LINES);
  glColor3f(1.0, 0.0, 0.0);
  glVertex3f(0, 0, 0);
  glVertex3f(length, 0, 0);

  glColor3f(0.0, 1.0, 0.0);
  glVertex3f(0, 0, 0);
  glVertex3f(0, length, 0);

  glColor3f(0.0, 0.0, 1.0);
  glVertex3f(0, 0, 0);
  glVertex3f(0, 0, length);
  glEnd();

  glPopAttrib();
  glLineWidth(1.0);
}

void DrawCamera(const theia::Camera& camera, const Eigen::Vector3f& color) {
  glPushMatrix();
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Zero();
  transformation_matrix.block<3, 3>(0, 0) =
      camera.GetOrientationAsRotationMatrix().transpose();
  transformation_matrix.col(3).head<3>() = camera.GetPosition();
  transformation_matrix(3, 3) = 1.0;

  // Apply world pose transformation.
  glMultMatrixd(reinterpret_cast<GLdouble*>(transformation_matrix.data()));

  // Draw Cameras.
  glColor3f(color[0], color[1], color[2]);

  // Create the camera wireframe. If intrinsic parameters are not set then use
  // the focal length as a guess.
  const float image_width =
      (camera.ImageWidth() == 0) ? camera.FocalLength() : camera.ImageWidth();
  const float image_height =
      (camera.ImageHeight() == 0) ? camera.FocalLength() : camera.ImageHeight();
  const float normalized_width = (image_width / 2.0) / camera.FocalLength();
  const float normalized_height = (image_height / 2.0) / camera.FocalLength();

  const Eigen::Vector3f top_left =
      normalized_focal_length *
      Eigen::Vector3f(-normalized_width, -normalized_height, 1);
  const Eigen::Vector3f top_right =
      normalized_focal_length *
      Eigen::Vector3f(normalized_width, -normalized_height, 1);
  const Eigen::Vector3f bottom_right =
      normalized_focal_length *
      Eigen::Vector3f(normalized_width, normalized_height, 1);
  const Eigen::Vector3f bottom_left =
      normalized_focal_length *
      Eigen::Vector3f(-normalized_width, normalized_height, 1);

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glBegin(GL_TRIANGLE_FAN);
  glVertex3f(0.0, 0.0, 0.0);
  glVertex3f(top_right[0], top_right[1], top_right[2]);
  glVertex3f(top_left[0], top_left[1], top_left[2]);
  glVertex3f(bottom_left[0], bottom_left[1], bottom_left[2]);
  glVertex3f(bottom_right[0], bottom_right[1], bottom_right[2]);
  glVertex3f(top_right[0], top_right[1], top_right[2]);
  glEnd();
  glPopMatrix();
}

void DrawPoints(const float point_scale,
                const float color_scale,
                const float alpha_scale) {
  const float default_point_size = point_size;
  const float default_alpha_scale = anti_aliasing_blend;

  // TODO(cmsweeney): Render points with the actual 3D point color! This would
  // require Theia to save the colors during feature extraction.
  //const Eigen::Vector3f default_color(0.05, 0.05, 0.05);

  // Enable anti-aliasing for round points and alpha blending that helps make
  // points look nicer.
  glDisable(GL_LIGHTING);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_BLEND);
  glEnable(GL_POINT_SMOOTH);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // The coordinates for calculating point attenuation. This allows for points
  // to get smaller as the OpenGL camera moves farther away.
  GLfloat point_size_coords[3];
  point_size_coords[0] = 1.0f;
  point_size_coords[1] = 0.055f;
  point_size_coords[2] = 0.0f;
  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, point_size_coords);


  glPointSize(point_scale * default_point_size);
  glBegin(GL_POINTS);
  for (int i = 0; i < world_points.size(); i++) {
    if (num_views_for_track[i] < min_num_views_for_track) {
      continue;
    }
    const Eigen::Vector3f color = point_colors[i] / 255.0;
    glColor4f(color_scale * color[0],
              color_scale * color[1],
              color_scale * color[2],
              alpha_scale * default_alpha_scale);

    glVertex3d(world_points[i].x(), world_points[i].y(), world_points[i].z());
  }
  glEnd();
}

void RenderScene() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Transformation to the viewer origin.
  glTranslatef(0.0, 0.0, zoom);
  glRotatef(navigation_rotation[0], 1.0f, 0.0f, 0.0f);
  glRotatef(navigation_rotation[1], 0.0f, 1.0f, 0.0f);
  if (draw_axes) {
    DrawAxes(10.0);
  }

  // Transformation from the viewer origin to the reconstruction origin.
  glTranslatef(viewer_position[0], viewer_position[1], viewer_position[2]);

  glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

  // Each 3D point is rendered 3 times with different point sizes, color
  // intensity, and alpha blending. This allows for a more complete texture-like
  // rendering of the 3D points. These values were found to experimentally
  // produce nice visualizations on most scenes.
  const float small_point_scale = 1.0, medium_point_scale = 5.0,
              large_point_scale = 10.0;
  const float small_color_scale = 1.0, medium_color_scale = 1.2,
              large_color_scale = 1.5;
  const float small_alpha_scale = 1.0, medium_alpha_scale = 2.1,
              large_alpha_scale = 3.3;

  DrawPoints(small_point_scale, small_color_scale, small_alpha_scale);
  DrawPoints(medium_point_scale, medium_color_scale, medium_alpha_scale);
  DrawPoints(large_point_scale, large_color_scale, large_alpha_scale);

  // Draw the cameras.
  theia::InitRandomGenerator();

  for (int k = 0; k < cameras_list.size(); k++) {
    // Set random color.
    const Eigen::Vector3f color(
      static_cast<float>(theia::RandDouble(0.0, 1.0)),
      static_cast<float>(theia::RandDouble(0.0, 1.0)),
      static_cast<float>(theia::RandDouble(0.0, 1.0)) );

    for (int i = 0; i < cameras_list[k].size(); i++) {
      DrawCamera(cameras_list[k][i], color);
    }
  }

  glutSwapBuffers();
}

void MouseButton(int button, int state, int x, int y) {
  // get the mouse buttons
  if (button == GLUT_RIGHT_BUTTON) {
    if (state == GLUT_DOWN) {
      right_mouse_button_active += 1;
    } else {
      right_mouse_button_active -= 1;
    }
  } else if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_DOWN) {
      left_mouse_button_active += 1;
      last_x_offset = 0.0;
      last_y_offset = 0.0;
    } else {
      left_mouse_button_active -= 1;
    }
  }

  // scroll event - wheel reports as button 3 (scroll up) and button 4 (scroll
  // down)
  if ((button == 3) || (button == 4)) {
    // Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
    if (state == GLUT_UP) return;  // Disregard redundant GLUT_UP events
    if (button == 3) {
      zoom *= delta_zoom;
    } else {
      zoom /= delta_zoom;
    }
  }

  mouse_pressed_x = x;
  mouse_pressed_y = y;
}

void MouseMove(int x, int y) {
  float x_offset = 0.0, y_offset = 0.0;

  // Rotation controls
  if (right_mouse_button_active) {
    navigation_rotation[0] += ((mouse_pressed_y - y) * 180.0f) / 200.0f;
    navigation_rotation[1] += ((mouse_pressed_x - x) * 180.0f) / 200.0f;

    mouse_pressed_y = y;
    mouse_pressed_x = x;

  } else if (left_mouse_button_active) {
    float delta_x = 0, delta_y = 0;
    const Eigen::AngleAxisf rotation(
        Eigen::AngleAxisf(theia::DegToRad(navigation_rotation[0]),
                          Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(theia::DegToRad(navigation_rotation[1]),
                          Eigen::Vector3f::UnitY()));

    // Panning controls.
    x_offset = (mouse_pressed_x - x);
    if (last_x_offset != 0.0) {
      delta_x = -(x_offset - last_x_offset) / 8.0;
    }
    last_x_offset = x_offset;

    y_offset = (mouse_pressed_y - y);
    if (last_y_offset != 0.0) {
      delta_y = (y_offset - last_y_offset) / 8.0;
    }
    last_y_offset = y_offset;

    // Compute the new viewer origin origin.
    viewer_position +=
        rotation.inverse() * Eigen::Vector3f(delta_x, delta_y, 0);
  }
}

// @mhsung
void Snapshot(const std::string filepath) {
  GLenum buffer(GL_BACK);
  const uint32_t offset = 8;
  uint32_t w = glutGet(GLUT_WINDOW_WIDTH) - offset;
  uint32_t h = glutGet(GLUT_WINDOW_HEIGHT) - offset;

  std::vector<GLubyte> fbuffer(w * h * 3);
  glReadBuffer(buffer);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  RenderScene();
  glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, &fbuffer[0]);

  theia::Image<theia::uchar> image(w, h, 3);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      int fbuffer_offset = 3 * (y * w + x);
      int image_offset = 3 * ((h - y - 1 )* w + x);
      for (uint32_t c = 0; c < 3; ++c)
        // Flip y-axis.
        image(x, h - y - 1, c) = fbuffer[fbuffer_offset + c];
    }
  }
  image.Write(filepath);
}

// @mhsung
void SnapshotRotatingAroundYAxis(const int num_samples) {
  const float angle = 360.0f / num_samples;
  const float original_rotation = navigation_rotation[1];
  for (int i = 0; i < num_samples; ++i) {
    std::stringstream sstr;
    sstr << "snapshot_" << std::setfill('0') << std::setw(2) << i << ".png";
    navigation_rotation[1] += angle;
    Snapshot(sstr.str());
  }
  navigation_rotation[1] = original_rotation;
  RenderScene();
}

// @mhsung
void ResetViewpoint() {
  viewer_position.setZero();
  zoom = -100.0;
  navigation_rotation.setZero();
  mouse_pressed_x = 0;
  mouse_pressed_y = 0;
  last_x_offset = 0.0;
  last_y_offset = 0.0;
  left_mouse_button_active = 0;
  right_mouse_button_active = 0;
  point_size = 1.0;
}

// @mhsung
void Idle() {
  RenderScene();

  // If snapshot file name is given, save a snapshot and close application.
  if (FLAGS_snapshot_file != "") {
    Snapshot(FLAGS_snapshot_file);
    exit(-1);
  }
}

void Keyboard(unsigned char key, int x, int y) {
  switch (key) {
    case 'r':  // reset viewpoint
      ResetViewpoint();
      break;
    case 'z':
      zoom *= delta_zoom;
      break;
    case 'Z':
      zoom /= delta_zoom;
      break;
    case 'p':
      point_size /= 1.2;
      break;
    case 'P':
      point_size *= 1.2;
      break;
    case 'f':
      normalized_focal_length /= 1.2;
      break;
    case 'F':
      normalized_focal_length *= 1.2;
      break;
//    case 'c':
//      draw_cameras = !draw_cameras;
//      break;
    case 'a':
      draw_axes = !draw_axes;
      break;
    case 't':
      ++min_num_views_for_track;
      break;
    case 'T':
      --min_num_views_for_track;
      break;
    case 'b':
      if (anti_aliasing_blend > 0) {
        anti_aliasing_blend -= 0.01;
      }
      break;
    case 'B':
      if (anti_aliasing_blend < 1.0) {
        anti_aliasing_blend += 0.01;
      }
    // @mhsung
    case 'x':
      navigation_rotation[0] = -90.0f;
      break;
    case 'y':
      navigation_rotation[0] = 0.0f;
      break;
    case 's':
      Snapshot("snapshot.png");
      break;
    case 'S':
      SnapshotRotatingAroundYAxis(72);
      break;
  }
}

void ReadCalibrationFiles(
  const std::string& calibration_file,
  std::unordered_map<std::string, theia::CameraIntrinsicsPrior>*
  camera_intrinsics_priors) {
  CHECK_NOTNULL(camera_intrinsics_priors);

  // Load calibration file with filename extensions.
  std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
    camera_intrinsics_priors_with_ext;
  CHECK(theia::ReadCalibration(calibration_file,
                               &camera_intrinsics_priors_with_ext))
  << "Could not read calibration file.";

  // Remove extensions.
  camera_intrinsics_priors->clear();
  camera_intrinsics_priors->reserve(camera_intrinsics_priors_with_ext.size());
  for (const auto& camera_intrinsics_prior :
    camera_intrinsics_priors_with_ext) {
    const std::string basename = stlplus::basename_part
      (camera_intrinsics_prior.first);
    camera_intrinsics_priors->emplace(basename, camera_intrinsics_prior.second);
  }
}

double Median(std::vector<double>* data) {
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n / 2;
  std::nth_element(data->begin(), mid_point, data->end());
  return *mid_point;
}

void NormalizeCameraPoses(
  theia::Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);

  // Compute the marginal median of the 3D points.
  std::vector<std::vector<double> > points(3);
  Eigen::Vector3d median;
  for (const theia::ViewId view_id : reconstruction->ViewIds()) {
    const auto* view = reconstruction->View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }
    const Eigen::Vector3d point = view->Camera().GetPosition();
    points[0].push_back(point[0]);
    points[1].push_back(point[1]);
    points[2].push_back(point[2]);
  }
  median(0) = Median(&points[0]);
  median(1) = Median(&points[1]);
  median(2) = Median(&points[2]);

  // Apply position transformation.
  TransformReconstruction(
    Eigen::Matrix3d::Identity(), -median, 1.0, reconstruction);

  // Find the median absolute deviation of the points from the median.
  std::vector<double> distance_to_median;
  distance_to_median.reserve(reconstruction->NumViews());
  for (const theia::ViewId view_id : reconstruction->ViewIds()) {
    const auto* view = reconstruction->View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }
    const Eigen::Vector3d point = view->Camera().GetPosition();
    distance_to_median.emplace_back((point - median).lpNorm<1>());
  }
  // This will scale the reconstruction so that the median absolute deviation of
  // the points is 100.
  const double scale = 100.0 / Median(&distance_to_median);

  TransformReconstruction(Eigen::Matrix3d::Identity(),
                          Eigen::Vector3d::Zero(),
                          scale,
                          reconstruction);

  // Compute a rotation such that the x-y plane is aligned to the dominating
  // plane of the cameras.
  std::vector<Eigen::Vector3d> cameras;
  const auto& view_ids = reconstruction->ViewIds();
  for (const theia::ViewId view_id : view_ids) {
    const auto* view = reconstruction->View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }
    cameras.emplace_back(view->Camera().GetPosition());
  }

  // Robustly estimate the dominant plane from the cameras. This will correspond
  // to a plan that is parallel to the ground plane for the majority of
  // reconstructions. We start with a small threshold and gradually increase it
  // until at an inlier set of at least 50% is found.
  RansacParameters ransac_params;
  ransac_params.max_iterations = 1000;
  ransac_params.error_thresh = 0.01;
  Plane plane;
  RansacSummary unused_summary;
  Eigen::Matrix3d rotation_for_dominant_plane = Eigen::Matrix3d::Identity();
  if (EstimateDominantPlaneFromPoints(ransac_params,
                                      RansacType::LMED,
                                      cameras,
                                      &plane,
                                      &unused_summary)) {
    // Set the rotation such that the plane normal points in the upward
    // direction. Choose the sign of the normal that will minimize the rotation
    // (this hopes to prevent having a rotation that flips the scene upside
    // down).
    const Eigen::Quaterniond rotation_quat1 =
      Eigen::Quaterniond::FromTwoVectors(plane.unit_normal,
                                         Eigen::Vector3d(0, 0, 1.0));
    const Eigen::Quaterniond rotation_quat2 =
      Eigen::Quaterniond::FromTwoVectors(-plane.unit_normal,
                                         Eigen::Vector3d(0, 0, 1.0));
    const Eigen::AngleAxisd rotation1_aa(rotation_quat1);
    const Eigen::AngleAxisd rotation2_aa(rotation_quat2);

    if (rotation1_aa.angle() < rotation2_aa.angle()) {
      rotation_for_dominant_plane = rotation1_aa.toRotationMatrix();
    } else {
      rotation_for_dominant_plane = rotation2_aa.toRotationMatrix();
    }
  }

  TransformReconstruction(rotation_for_dominant_plane,
                          Eigen::Vector3d::Zero(),
                          1.0,
                          reconstruction);
}

std::unique_ptr<Reconstruction> AddCameraList(
  const std::string& data_type, const std::string filepath,
  const std::unordered_map<std::string, theia::CameraIntrinsicsPrior>*
  camera_intrinsics_priors,
  theia::Reconstruction* reference_reconstruction = nullptr) {

  // Load modelview matrices.
  std::unordered_map<std::string, Eigen::Affine3d> modelviews;
  CHECK(ReadModelviews(data_type, filepath, &modelviews));

  // Create reconstruction.
  std::unique_ptr<theia::Reconstruction> reconstruction =
    CreateTheiaReconstructionFromModelviews(
      modelviews, camera_intrinsics_priors);

  if (reference_reconstruction) {
    if (FLAGS_robust_alignment_threshold > 0.0) {
      // Align the reconstruction to ground truth.
      AlignReconstructionsRobust(FLAGS_robust_alignment_threshold,
                                 *reference_reconstruction,
                                 reconstruction.get());
    } else {
      AlignReconstructions(*reference_reconstruction, reconstruction.get());
    }
  } else {
    // Centers the reconstruction based on the absolute deviation of 3D points.
    //reconstruction->Normalize();
    NormalizeCameraPoses(reconstruction.get());
  }

  // Set up camera drawing.
  cameras_list.emplace_back();
  std::vector<theia::Camera>& cameras = cameras_list.back();

  cameras.reserve(reconstruction->NumViews());
  for (const theia::ViewId view_id : reconstruction->ViewIds()) {
    const auto* view = reconstruction->View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }
    cameras.emplace_back(view->Camera());
  }

  return std::move(reconstruction);
}

void SplitString(const std::string& str, const char delimiter,
                 std::vector<std::string>* tokens) {
  CHECK_NOTNULL(tokens)->clear();

  std::stringstream sstr(str);
  std::string token;
  while (std::getline(sstr, token, delimiter)) {
    if (token.empty() || token == " ") break;
    tokens->push_back(token);
  }
}

int main(int argc, char* argv[]) {
  THEIA_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

//  // Output as a binary file.
//  std::unique_ptr<theia::Reconstruction> reconstruction(
//      new theia::Reconstruction());
//  CHECK(ReadReconstruction(FLAGS_reconstruction, reconstruction.get()))
//      << "Could not read reconstruction file.";

  std::vector<std::string> data_type_list;
  std::vector<std::string> filepath_list;
  SplitString(FLAGS_data_type_list, ',', &data_type_list);
  SplitString(FLAGS_filepath_list, ',', &filepath_list);
  CHECK_EQ(data_type_list.size(), filepath_list.size());
  const int num_reconstructions = filepath_list.size();

  // Read camera intrinsics if provided.
  std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
    camera_intrinsics_priors;
  if (FLAGS_calibration_file.size() != 0) {
    ReadCalibrationFiles(FLAGS_calibration_file, &camera_intrinsics_priors);
  }

  std::unique_ptr<theia::Reconstruction> reference_reconstruction(nullptr);
  for (int i = 0; i < num_reconstructions; i++) {
    LOG(INFO) << "Load '" << filepath_list[i] << "'.";
    std::unique_ptr<theia::Reconstruction> reconstruction =
      AddCameraList(data_type_list[i], filepath_list[i],
                    &camera_intrinsics_priors, reference_reconstruction.get());

    // Consider the first one as reference.
    if (i == 0) {
      reference_reconstruction = std::move(reconstruction);
    } else {
      reconstruction.release();
    }
  }


  // Set up opengl and glut.
  glutInit(&argc, argv);
  glutInitWindowPosition(600, 600);
  glutInitWindowSize(1200, 800);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutCreateWindow("Theia Reconstruction Viewer");

  // Set the camera
  gluLookAt(0.0f, 0.0f, -6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

  // register callbacks
  glutDisplayFunc(RenderScene);
  glutReshapeFunc(ChangeSize);
  glutMouseFunc(MouseButton);
  glutMotionFunc(MouseMove);
  glutKeyboardFunc(Keyboard);
  //glutIdleFunc(RenderScene);
  glutIdleFunc(Idle);

  // enter GLUT event processing loop
  glutMainLoop();

  return 0;
}
