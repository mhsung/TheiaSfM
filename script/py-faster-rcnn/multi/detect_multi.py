#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../../'))
sys.path.append(os.path.join(BASE_DIR, '3rdparty', 'py-faster-rcnn', 'lib'))
sys.path.append(os.path.join(BASE_DIR, '3rdparty', 'py-faster-rcnn', 'tools'))
sys.path.append(os.path.join(BASE_DIR, 'script', 'RenderForCNN', 'multi'))

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe
import cv2
import cnn_utils
import glob
import gflags
import numpy as np
import pandas as pd


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16', 'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel')}


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('class_name_file', os.path.join(
    BASE_DIR, '../../script/RenderForCNN/multi/class_names.txt'), '')
gflags.DEFINE_string('out_bbox_file', 'convnet/bboxes.csv', '')

gflags.DEFINE_integer('gpu_id', 0, 'GPU device id to use [0]')
gflags.DEFINE_bool('cpu_mode', False, 'Use CPU mode (overrides --gpu)')
gflags.DEFINE_string('demo_net', 'vgg16', 'Network to use [vgg16]')
# gflags.DEFINE_double('conf_thresh', 0.8, '')
gflags.DEFINE_float('conf_thresh', 0.5, '')
gflags.DEFINE_float('nms_thresh', 0.3, '')


def detect_bboxes(net, im_names, subset_classes):
    """Detect object classes in an image using pre-computed object proposals."""
    df = cnn_utils.create_bbox_data_frame()

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)

        # Load the demo image.
        im_file = os.path.join(FLAGS.data_dir, 'images', im_name)
        im = cv2.imread(im_file)

        # Detect all object classes and regress object bounds.
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(
            timer.total_time, boxes.shape[0])

        # Detect for each class
        for subset_cls_ind in range(len(class_names_to_be_detected)):
            cls = class_names_to_be_detected[subset_cls_ind]
            try:
                cls_ind = CLASSES.index(cls)
            except:
                print('error: class does not exist in training data: '
                      '{0}'.format(cls))
                exit(-1)

            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, FLAGS.nms_thresh)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= FLAGS.conf_thresh)[0]
            if len(inds) > 0:
                print ('{} {}(s) are detected.'.format(len(inds), cls))

            for i in inds:
                # Append a row.
                df.loc[len(df)] = [
                    im_name, subset_cls_ind,
                    dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3],
                    dets[i, -1]]

    return df


if __name__ == '__main__':
    FLAGS(sys.argv)

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[FLAGS.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[FLAGS.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if FLAGS.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(FLAGS.gpu_id)
        cfg.GPU_ID = FLAGS.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # Read class names to be detected.
    class_names_to_be_detected = cnn_utils.read_class_names(
        FLAGS.class_name_file)

    # Read image names.
    im_names = [os.path.basename(x) for x in
                glob.glob(os.path.join(FLAGS.data_dir, 'images', '*.png'))]
    im_names.sort()

    # Detect and save bounding boxes.
    df = detect_bboxes(net, im_names, class_names_to_be_detected)
    out_file = os.path.join(FLAGS.data_dir, FLAGS.out_bbox_file)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # Save class indices as integer.
    df['class_index'] = df['class_index'].map(lambda x: '%i' % x)
    df.to_csv(out_file, header=False)
