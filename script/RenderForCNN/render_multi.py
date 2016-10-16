#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../'))
sys.path.append(os.path.join(BASE_DIR, '3rdparty', 'RenderForCNN'))
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'render_pipeline'))

from joblib import Parallel, delayed
from PIL import Image
import cnn_utils
import gflags
import glob
import multiprocessing


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('class_name_file', os.path.join(
    BASE_DIR, 'script/RenderForCNN/class_names.txt'), '')
gflags.DEFINE_string('bbox_file', 'convnet/object_bboxes.csv', '')
gflags.DEFINE_string('orientation_file',
    'convnet/object_orientations_fitted.csv', '')
gflags.DEFINE_string('out_render_dir', 'convnet/object_render_fitted', '')

gflags.DEFINE_bool('with_object_index', True, '')
gflags.DEFINE_bool('use_opengl', True, '')


def render_model(bbox_idx, row, class_names, preds):
    class_name = class_names[int(row['class_index'])]
    model_obj_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'sample_model',
        class_name + '.obj')

    out_file = os.path.join(FLAGS.data_dir, FLAGS.out_render_dir,
                            str(bbox_idx).zfill(num_digits))

    # io_redirect = ''
    io_redirect = '> /dev/null 2>&1'
    if FLAGS.use_opengl:
        cmd = 'OSMesaViewer '
        cmd += '--mesh_file=' + model_obj_file + ' '
        cmd += '--set_camera_param '
        cmd += '--azimuth_deg=' + str(int(preds[bbox_idx, 0])) + ' '
        cmd += '--elevation_deg=' + str(int(preds[bbox_idx, 1])) + ' '
        cmd += '--theta_deg=' + str(int(preds[bbox_idx, 2])) + ' '
        cmd += '--snapshot_file=' + out_file
    else:
        cmd = 'python %s -m %s -a %s -e %s -t %s -d %s -o %s' % \
              (os.path.join(g_render4cnn_root_folder, 'demo_render',
                            'render_class_view.py'), model_obj_file,
               str(int(preds[bbox_idx, 0])),
               str(int(preds[bbox_idx, 1])),
               str(int(preds[bbox_idx, 2])),
               str(3.0), out_file + '.png')
    print ">> Running rendering command: \n \t %s" % (cmd)
    os.system('%s %s' % (cmd, io_redirect))

    # Remove background margin.
    if os.path.exists(out_file + '.png'):
        out_im = Image.open(out_file + '.png')
        bbox = out_im.getbbox()
        out_im = out_im.crop(bbox)
        out_im.save(out_file + '.png')
    else:
        print('Warning: Rendering failed: ' + out_file)


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read class names to be detected.
    class_names = cnn_utils.read_class_names(FLAGS.class_name_file)

    # Read bounding boxes.
    df, num_digits = cnn_utils.read_bboxes(
        os.path.join(FLAGS.data_dir, FLAGS.bbox_file),
        FLAGS.with_object_index)

    # Read estimated best orientations.
    preds = cnn_utils.read_orientations(
        os.path.join(FLAGS.data_dir, FLAGS.orientation_file))
    assert (len(df.index) == preds.shape[0])

    if not os.path.exists(os.path.join(
            FLAGS.data_dir, FLAGS.out_render_dir)):
        os.makedirs(os.path.join(
            FLAGS.data_dir, FLAGS.out_render_dir))

    # for bbox_idx, row in df.iterrows():
    #     render_model(bbox_idx, row, class_names, preds)

    # Parallel processing.
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(
        render_model)(bbox_idx, row, class_names, preds)
                                         for bbox_idx, row in df.iterrows())
