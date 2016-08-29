# Basic programs
add_executable(match_features match_features.cc)
target_link_libraries(match_features theia ${GFLAGS_LIBRARIES})

add_executable(extract_features extract_features.cc)
target_link_libraries(extract_features theia ${GFLAGS_LIBRARIES})

add_executable(compute_two_view_geometry compute_two_view_geometry.cc)
target_link_libraries(compute_two_view_geometry theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

# Tools for building SfM models
add_executable(build_reconstruction build_reconstruction.cc)
target_link_libraries(build_reconstruction theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(build_1dsfm_reconstruction build_1dsfm_reconstruction.cc)
target_link_libraries(build_1dsfm_reconstruction theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

# Reconstruction visualization.
find_package(OpenGL REQUIRED)
if (NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  find_package(GLUT REQUIRED)
  if (GLUT_FOUND)
    message ("-- Found GLUT: ${GLUT_INCLUDE_DIR}")
  else (GLUT FOUND)
    message (FATAL_ERROR "-- Can't find GLUT. Please set GLUT_INCLUDE_DIR & GLUT_LIBRARY")
  endif (GLUT_FOUND)
  include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIR} )
  set(EXTRA_GL_LIBRARIES ${GLUT_LIBRARIES})
else (NOT MSVC)
  find_package(GLEW REQUIRED)
  if (GLEW_FOUND)
    message ("-- Found GLEW: ${GLEW_INCLUDE_DIRS}")
  else (GLEW FOUND)
    message (FATAL_ERROR "-- Can't find GLEW. Please set GLEW_INCLUDE_DIR & GLEW_LIBRARY")
  endif (GLEW_FOUND)
  include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLEW_INCLUDE_DIRS} )
  set(EXTRA_GL_LIBRARIES ${GLEW_LIBRARIES})
endif (NOT MSVC)

add_executable(view_reconstruction view_reconstruction.cc)
target_link_libraries(view_reconstruction theia ${GFLAGS_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${EXTRA_GL_LIBRARIES})

# Useful tools for analyzing reconstructions.
add_executable(compute_reconstruction_statistics compute_reconstruction_statistics.cc)
target_link_libraries(compute_reconstruction_statistics theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(colorize_reconstruction colorize_reconstruction.cc)
target_link_libraries(colorize_reconstruction theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(compare_reconstructions compare_reconstructions.cc)
target_link_libraries(compare_reconstructions theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(compute_matching_relative_pose_errors compute_matching_relative_pose_errors.cc)
target_link_libraries(compute_matching_relative_pose_errors theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(evaluate_relative_translation_optimization evaluate_relative_translation_optimization.cc)
target_link_libraries(evaluate_relative_translation_optimization theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(verify_1dsfm_input verify_1dsfm_input.cc)
target_link_libraries(verify_1dsfm_input theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

# File conversions and exporters.
add_executable(convert_sift_key_file convert_sift_key_file.cc)
target_link_libraries(convert_sift_key_file theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(convert_bundle_file convert_bundle_file.cc)
target_link_libraries(convert_bundle_file theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(export_to_nvm_file export_to_nvm_file.cc)
target_link_libraries(export_to_nvm_file theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(convert_nvm_file convert_nvm_file.cc)
target_link_libraries(convert_nvm_file theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(convert_theia_reconstruction_to_bundler_file convert_theia_reconstruction_to_bundler_file.cc)
target_link_libraries(convert_theia_reconstruction_to_bundler_file theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(write_reconstruction_ply_file write_reconstruction_ply_file.cc)
target_link_libraries(write_reconstruction_ply_file theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(export_reconstruction_to_pmvs export_reconstruction_to_pmvs.cc)
target_link_libraries(export_reconstruction_to_pmvs theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(create_calibration_file_from_exif create_calibration_file_from_exif.cc)
target_link_libraries(create_calibration_file_from_exif theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(create_reconstruction_from_strecha_dataset create_reconstruction_from_strecha_dataset.cc)
target_link_libraries(create_reconstruction_from_strecha_dataset theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

# @mhsung
add_executable(exp_export_modelviews exp_export_modelviews.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_export_modelviews theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_compare_modelviews exp_compare_modelviews.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_compare_modelviews theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_export_orientations exp_export_orientations.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_export_orientations theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_compare_relative_orientations exp_compare_relative_orientations.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_compare_relative_orientations theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_compare_orientations exp_compare_orientations.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_compare_orientations theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_match_features exp_match_features.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_match_features theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_two_view_match_features exp_two_view_match_features.cc)
target_link_libraries(exp_two_view_match_features theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_build_reconstruction exp_build_reconstruction.cc
  exp_bounding_box_utils.h exp_bounding_box_utils.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_build_reconstruction theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_extract_match_info exp_extract_match_info.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_extract_match_info theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_draw_matches exp_draw_matches.cc)
target_link_libraries(exp_draw_matches theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_convert_feature_track_file exp_convert_feature_track_file.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc
  exp_feature_match_utils.h exp_feature_match_utils.cc
  exp_feature_track_io.h exp_feature_track_io.cc)
target_link_libraries(exp_convert_feature_track_file theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_draw_feature_tracks exp_draw_feature_tracks.cc
  exp_feature_track_io.h exp_feature_track_io.cc)
target_link_libraries(exp_draw_feature_tracks theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_export_camera_view_directions exp_export_camera_view_directions.cc
  exp_bounding_box_utils.h exp_bounding_box_utils.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_export_camera_view_directions theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(exp_run_bundle_adjustment exp_run_bundle_adjustment.cc
  exp_bounding_box_utils.h exp_bounding_box_utils.cc
  exp_camera_param_utils.h exp_camera_param_utils.cc exp_camera_param_io.h exp_camera_param_io.cc)
target_link_libraries(exp_run_bundle_adjustment theia ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

