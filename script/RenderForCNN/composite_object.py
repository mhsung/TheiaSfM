#!/usr/bin/env python

from PIL import Image
import cv2
import gflags
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import sys


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('render_dir', 'convnet/largest/render_views', '')
gflags.DEFINE_string('bbox_dir', 'convnet/largest/bboxes', '')
gflags.DEFINE_string('composite_dir', 'convnet/composite_all', '')

gflags.DEFINE_string('target_class', 'chair', '')
gflags.DEFINE_bool('draw_bboxes', True, '')
gflags.DEFINE_bool('composite_rendered', True, '')
gflags.DEFINE_bool('half_size_output', True, '')


def vis_estimations(image_name):
    """Draw detected bounding boxes."""

    basename = os.path.splitext(os.path.basename(image_name))[0]

    output_im_file = os.path.join(
        FLAGS.data_dir, FLAGS.composite_dir, image_name)
    if os.path.exists(output_im_file):
        return

    # Load the demo image
    im_file = os.path.join(FLAGS.data_dir, 'images', image_name)
    im = cv2.imread(im_file)
    im_width = np.size(im, 1)
    im_height = np.size(im, 0)

    # Add alpha channel
    (B, G, R) = cv2.split(im)
    im = cv2.merge((B, G, R, B))
    im[:,:,3] = 255

    # Load detected object bounding boxes
    bbox_file = os.path.join(
        FLAGS.data_dir, FLAGS.bbox_dir, basename + '.txt')

    rendered_im_file = os.path.join(
        FLAGS.data_dir, FLAGS.render_dir, basename + '.png')

    # Composite rendered 3D model rendered images
    if FLAGS.composite_rendered and \
        os.path.exists(bbox_file) and \
        os.path.exists(rendered_im_file):
        print(rendered_im_file)

        dets = np.genfromtxt(bbox_file, delimiter=' ', dtype=float)
        bbox = dets[:4]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        rendered_im = cv2.imread(rendered_im_file, cv2.IMREAD_UNCHANGED)
        model_im_width = np.size(rendered_im, 1)
        model_im_height = np.size(rendered_im, 0)

        scale_width = bbox_width / model_im_width
        scale_height = bbox_height / model_im_height
        resized_model_im = cv2.resize(
            rendered_im, None, fx=scale_width, fy=scale_height,
            interpolation=cv2.INTER_CUBIC)

        sx = max(int(round(bbox[0])), 0)
        sy = max(int(round(bbox[1])), 0)
        ex = min(sx + np.size(resized_model_im, 1), np.size(im, 1))
        ey = min(sy + np.size(resized_model_im, 0), np.size(im, 0))
        roi = im[sy:ey, sx:ex]

        # Create mask
        (B, G, R, A) = cv2.split(resized_model_im)
        mask = A
        mask_inv = cv2.bitwise_not(mask)

        # Blend two images
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(
            resized_model_im, resized_model_im, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)
        im[sy:ey, sx:ex] = dst

    im = im[:, :, (2, 1, 0)]

    fig = plt.figure(frameon=False)
    DPI = fig.get_dpi()
    if FLAGS.half_size_output:
        DPI = 2 * DPI
    fig.set_size_inches(im_width/float(DPI), im_height/float(DPI))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im, aspect='equal')

    # Draw bounding boxes
    if FLAGS.draw_bboxes and os.path.exists(bbox_file):
        print(bbox_file)
        dets = np.genfromtxt(bbox_file, delimiter=' ', dtype=float)
        bbox = dets[:4]
        score = dets[-1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )

        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(FLAGS.target_class, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                #fontsize=14, color='white')
                fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(
    #     FLAGS.target_class, FLAGS.target_class, thresh), fontsize=14)

    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.composite_dir)):
        os.makedirs(os.path.join(FLAGS.data_dir, FLAGS.composite_dir))
    plt.savefig(output_im_file)
    plt.close(fig)

if __name__ == '__main__':
    FLAGS(sys.argv)

    im_names = [os.path.basename(x) for x in
            glob.glob(os.path.join(FLAGS.data_dir, 'images', '*.png'))]

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        vis_estimations(im_name)

    plt.show()
