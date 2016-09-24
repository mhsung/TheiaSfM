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
    os.path.abspath(__file__)), '../../3rdparty/py-faster-rcnn'))
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'tools'))

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from PIL import Image
from utils.timer import Timer
import glob
import gflags
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, cv2
import argparse

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
gflags.DEFINE_string('output_dir', 'convnet/output', '')
gflags.DEFINE_string('target_class', 'chair', '')
gflags.DEFINE_integer('gpu_id', 0, 'GPU device id to use [0]')

gflags.DEFINE_bool('cpu_mode', False, 'Use CPU mode (overrides --gpu)')
gflags.DEFINE_string('demo_net', 'vgg16', 'Network to use [vgg16]')
# gflags.DEFINE_double('conf_thresh', 0.8, '')
gflags.DEFINE_float('conf_thresh', 0.5, '')
gflags.DEFINE_float('nms_thresh', 0.3, '')


'''
def vis_detections(im, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    #if len(inds) == 0:
    #    return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.output_dir)):
        os.makedirs(os.path.join(FLAGS.data_dir, FLAGS.output_dir))

    detection_im_file = os.path.join(FLAGS.data_dir, FLAGS.output_dir, image_name)
    plt.savefig(detection_im_file, bbox_inches='tight')
    plt.close(fig)
'''

def write_cropped_images(imdata, image_name, class_name, dets, thresh=0.5):
    """Write cropped images."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    print ('{} {}(s) are detected.'.format(len(inds), FLAGS.target_class))
    if len(inds) == 0:
        return

    basename = os.path.splitext(image_name)[0]
    # print(os.path.join(FLAGS.data_dir, FLAGS.output_dir, basename))
    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.output_dir, basename)):
        os.makedirs(os.path.join(FLAGS.data_dir, FLAGS.output_dir, basename))

    imdata = imdata[:, :, (2, 1, 0)]

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        im = Image.fromarray(imdata, 'RGB')
        cropped_im = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        cropped_im_file = os.path.join(FLAGS.data_dir, FLAGS.output_dir, basename,
                class_name + '_' + str(i))
        print(cropped_im_file)
        cropped_im.save(cropped_im_file + '.png')
        np.savetxt(cropped_im_file + '_bbox.txt',
                dets[i, :].reshape(1, 5), delimiter=' ', fmt='%f')

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(FLAGS.data_dir, 'images', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    print ('Target class: {}'.format(FLAGS.target_class))

    # Visualize detections for each class
    # FLAGS.conf_thresh = 0.8
    FLAGS.conf_thresh = 0.5
    FLAGS.nms_thresh = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        #
        if cls != FLAGS.target_class:
            continue
        #
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, FLAGS.nms_thresh)
        dets = dets[keep, :]
        write_cropped_images(im, image_name, cls, dets, thresh=FLAGS.conf_thresh)
        #vis_detections(im, image_name, cls, dets, thresh=FLAGS.conf_thresh)

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

    #im_names = ['MVI_0018_001.png']
    im_names = [os.path.basename(x) for x in
            glob.glob(os.path.join(FLAGS.data_dir, 'images', '*.png'))]
    im_names.sort()

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)

    plt.show()
