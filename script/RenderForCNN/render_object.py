#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../3rdparty/RenderForCNN'))
sys.path.append(BASE_DIR)
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'render_pipeline'))

import gflags
import glob
from PIL import Image


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('view_dir', 'convnet/largest/views', '')
gflags.DEFINE_string('render_dir', 'convnet/largest/render_views', '')
gflags.DEFINE_string('target_class', 'chair', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    model_obj_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'sample_model',
        FLAGS.target_class + '.obj')

    view_file_list = glob.glob(
        os.path.join(FLAGS.data_dir, FLAGS.view_dir, '*.txt'))
    for view_file in view_file_list:
        basename = os.path.splitext(os.path.basename(view_file))[0]

        out_img_file = os.path.join(
            FLAGS.data_dir, FLAGS.render_dir, basename + '.png')
        if os.path.exists(out_img_file):
            continue
        print(out_img_file)

        if not os.path.exists(os.path.join(
                FLAGS.data_dir, FLAGS.render_dir)):
            os.makedirs(os.path.join(
                FLAGS.data_dir, FLAGS.render_dir))

        estimated_viewpoints = [[float(x) for x in line.rstrip().split(' ')]
                                for line in open(view_file, 'r')]
        v = estimated_viewpoints[0]

        # Change the model.obj file to render images of a different
        # model/category
        # #io_redirect = ''
        io_redirect = '> /dev/null 2>&1'
        python_cmd = 'python %s -m %s -a %s -e %s -t %s -d %s -o %s' % \
                     (os.path.join(
                         g_render4cnn_root_folder, 'demo_render',
                         'render_class_view.py'), model_obj_file,
                      str(v[0]), str(v[1]), str(v[2]), str(3.0),
                      out_img_file)
        print ">> Running rendering command: \n \t %s" % (python_cmd)
        os.system('%s %s' % (python_cmd, io_redirect))

        # Remove background margin.
        if os.path.exists(out_img_file):
            im2 = Image.open(out_img_file)
            bbox = im2.getbbox()
            im2 = im2.crop(bbox)
            im2.save(out_img_file)
        else:
            print('Warning: Rendering failed: ' + out_img_file)
