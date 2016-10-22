#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import gflags
import os
import sys


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('image_wildcard', '', '')

# Check whether 'Caffe' python path and 'OSMesaViewer' path are set properly.
os.system('echo $PATHONPATH')
os.system('echo $PATH')
BASE_DIR = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    FLAGS(sys.argv)

    # Detect objects.
    python_file = os.path.join(BASE_DIR, 'py-faster-rcnn', 'detect_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Track objects.
    python_file = os.path.join(BASE_DIR, 'SORT', 'sort_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Crop images.
    python_file = os.path.join(BASE_DIR, 'py-faster-rcnn', 'crop_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Estimate best orientation.
    python_file = os.path.join(
        BASE_DIR, 'RenderForCNN', 'estimate_best_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Estimate orientation scores.
    python_file = os.path.join(
        BASE_DIR, 'RenderForCNN', 'estimate_scores_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Do seam fitting in camera parameter space.
    python_file = os.path.join(
        BASE_DIR, 'RenderForCNN', 'seam_fitting_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Render 3D model.
    python_file = os.path.join(BASE_DIR, 'RenderForCNN', 'render_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Composite images.
    python_file = os.path.join(BASE_DIR, 'RenderForCNN', 'composite_multi.py')
    cmd = python_file + ' --data_dir=' + FLAGS.data_dir
    print(cmd)
    os.system(cmd)

    # Create video.
    composite_image_wildcard = os.path.join(
        FLAGS.data_dir, 'convnet', 'object_composite_fitted',
        FLAGS.image_wildcard)

    composite_video_file = os.path.join(
        FLAGS.data_dir, 'convnet', 'composite_fitted.mp4')

    cmd = 'ffmpeg -i ' + composite_image_wildcard + ' '\
          '-pix_fmt yuv420p -r 24 -b:v 8000k ' + \
          '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' + \
          composite_video_file
    print(cmd)
    os.system(cmd)
