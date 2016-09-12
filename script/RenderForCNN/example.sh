# Detect objects.
../py-faster-rcnn/detect.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
    --target_class=car

# Estimate best orientation.
./estimate.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
    --target_class=car

# Estimate orientation scores.
./estimate_pred.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
    --target_class=car

# Find the largest bounding box.
./find_largest.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
    -target_class=car

# Fit orientation in the time sequence.
../run_plot.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
	--param_data_names=''\
	--param_data_dirs='' \
	--convnet_dir=convnet/largest/preds \
	--output_plot_file=convnet/largest/fitted_views.png \
	--output_convnet_seam_fitting=convnet/largest/fitted_views

# Interpolate orientations.
../run_interpolate_camera_params.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
    --input_param_data_dir=convnet/largest/fitted_views \
    --output_param_data_dir=convnet/largest/interp_fitted_views

# Interpolate bounding boxes.
../run_interpolate_bboxes.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
	--input_bbox_data_dir=convnet/largest/bboxes \
	--output_bbox_data_dir=convnet/largest/interp_bboxes

# Render 3D model.
./render_object.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
    --view_dir=convnet/largest/interp_fitted_views \
    --render_dir=convnet/largest/render_interp_fitted_views \
    --target_class=car

# Composite images.
./composite_object.py \
    --data_dir=/Users/msung/Developer/data/MVI_0219 \
    --render_dir=convnet/largest/render_interp_fitted_views \
    --bbox_dir=convnet/largest/interp_bboxes \
    --composite_dir=convnet/largest/composite_interp_fitted \
    --target_class=car

# Create video.
ffmpeg -i /Users/msung/Developer/data/MVI_0219/convnet/largest/composite_interp_fitted/MVI_0219_%04d.png \
    -pix_fmt yuv420p -r 24 -b:v 8000k -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    /Users/msung/Developer/data/MVI_0219/convnet/largest/composite_interp_fitted.mp4
