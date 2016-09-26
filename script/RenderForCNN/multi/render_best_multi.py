#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../../3rdparty/RenderForCNN'))
sys.path.append(BASE_DIR)
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'render_pipeline'))

import cnn_utils
import gflags
import glob
from PIL import Image


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('class_name_file', os.path.join(
    BASE_DIR, '../../script/RenderForCNN/multi/class_names.txt'), '')
gflags.DEFINE_string('bbox_file', 'convnet/bboxes.csv', '')
gflags.DEFINE_string('best_orientation_file', 'convnet/orientations_best.csv', '')
gflags.DEFINE_string('out_render_dir', 'convnet/render_best', '')

gflags.DEFINE_bool('use_opengl', True, '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read class names to be detected.
    class_names = cnn_utils.read_class_names(FLAGS.class_name_file)

    # Read bounding boxes.
    df, num_digits = cnn_utils.read_bboxes(
            os.path.join(FLAGS.data_dir, FLAGS.bbox_file))

    preds = cnn_utils.read_orientations(
            os.path.join(FLAGS.data_dir, FLAGS.best_orientation_file))
    assert (len(df.index) == preds.shape[0])

    # Render 3D model with given orientations.
    for bbox_idx, row in df.iterrows():
        class_name = class_names[int(row['class_index'])]
        model_obj_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../sample_model',
        class_name + '.obj')

        if not os.path.exists(os.path.join(
                FLAGS.data_dir, FLAGS.out_render_dir)):
            os.makedirs(os.path.join(
                FLAGS.data_dir, FLAGS.out_render_dir))

        out_file = os.path.join(FLAGS.data_dir, FLAGS.out_render_dir,
                str(bbox_idx).zfill(num_digits) + '.png')

        # io_redirect = ''
        io_redirect = '> /dev/null 2>&1'
        if FLAGS.use_opengl:
            cmd = 'OSMesaViewer --mesh_file=%s --azimuth_deg=%s --elevation_deg=%s --theta_deg=%s --snapshot_file=%s' % \
                    model_obj_file,
                    str(int(preds[bbox_idx, 0])),
                    str(int(preds[bbox_idx, 1])),
                    str(int(preds[bbox_idx, 2])),
                    out_file
        else:
            cmd = 'python %s -m %s -a %s -e %s -t %s -d %s -o %s' % \
                    (os.path.join(g_render4cnn_root_folder, 'demo_render',
                        'render_class_view.py'), model_obj_file,
                        str(int(preds[bbox_idx, 0])),
                        str(int(preds[bbox_idx, 1])),
                        str(int(preds[bbox_idx, 2])),
                        str(3.0), out_file)
        print ">> Running rendering command: \n \t %s" % (cmd)
        os.system('%s %s' % (cmd, io_redirect))

        # Remove background margin.
        if os.path.exists(out_file):
            im2 = Image.open(out_file)
            bbox = im2.getbbox()
            im2 = im2.crop(bbox)
            im2.save(out_file)
        else:
            print('Warning: Rendering failed: ' + out_file)
        break

