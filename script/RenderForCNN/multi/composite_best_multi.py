#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../../3rdparty/RenderForCNN'))
sys.path.append(BASE_DIR)
from global_variables import *
sys.path.append(os.path.join(g_render4cnn_root_folder, 'render_pipeline'))

from joblib import Parallel, delayed
import cnn_utils
import cv2
import gflags
import glob
import multiprocessing
import numpy as np


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('class_name_file', os.path.join(
    BASE_DIR, '../../script/RenderForCNN/multi/class_names.txt'), '')
gflags.DEFINE_string('bbox_file', 'convnet/bboxes.csv', '')
gflags.DEFINE_string('render_dir', 'convnet/render_best', '')
gflags.DEFINE_string('out_composite_dir', 'convnet/composite_best', '')

gflags.DEFINE_bool('composite_rendered', True, '')
gflags.DEFINE_bool('draw_bboxes', True, '')
gflags.DEFINE_bool('half_size_output', True, '')


def composite_rendered_images(im, bbox_idx, row, num_digits):
    render_im_file = os.path.join(FLAGS.data_dir, FLAGS.render_dir,
                                  str(bbox_idx).zfill(num_digits) + '.png')
    # print(render_im_file)
    assert (os.path.exists(render_im_file))

    render_im = cv2.imread(render_im_file, cv2.IMREAD_UNCHANGED)
    model_im_width = np.size(render_im, 1)
    model_im_height = np.size(render_im, 0)

    bbox_width = row['x2'] - row['x1']
    bbox_height = row['y2'] - row['y1']
    scale_width = bbox_width / model_im_width
    scale_height = bbox_height / model_im_height
    resized_model_im = cv2.resize(
        render_im, None, fx=scale_width, fy=scale_height,
        interpolation=cv2.INTER_CUBIC)

    sx = max(int(round(row['x1'])), 0)
    sy = max(int(round(row['y1'])), 0)
    ex = min(sx + np.size(resized_model_im, 1), np.size(im, 1))
    ey = min(sy + np.size(resized_model_im, 0), np.size(im, 0))
    roi = im[sy:ey, sx:ex]

    # Create mask
    (_, _, _, A) = cv2.split(resized_model_im)
    mask = A
    mask_inv = cv2.bitwise_not(mask)

    # Blend two images
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(
        resized_model_im, resized_model_im, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    im[sy:ey, sx:ex] = dst
    return im


def draw_bboxes(im, row, class_names, colors):
    class_idx = row['class_index']
    class_name = class_names[class_idx]

    sx = max(int(round(row['x1'])), 0)
    sy = max(int(round(row['y1'])), 0)
    ex = min(int(round(row['x2'])), np.size(im, 1))
    ey = min(int(round(row['y2'])), np.size(im, 0))

    bbox_thickness = 8
    cv2.rectangle(im, (sx, sy), (ex, ey), colors[class_idx, :],
                  bbox_thickness)

    label = '{:s} {:.3f}'.format(class_name, row['score'])
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    text_thickness = 4
    label_size = cv2.getTextSize(label, fontface, scale, text_thickness)[0]
    # Fill background of label (-1 thickness).
    cv2.rectangle(im, (sx - bbox_thickness / 2,
                       sy - bbox_thickness - label_size[1]),
                  (ex + bbox_thickness / 2, sy), colors[class_idx, :], -1)
    cv2.putText(im, label, (sx, sy - bbox_thickness / 2), fontface, scale,
                (255, 255, 255, 255), text_thickness, cv2.CV_AA)

    return im


def generate_output_images(im_name, df, class_names, colors, num_digits):
    out_im_file = os.path.join(
        FLAGS.data_dir, FLAGS.out_composite_dir, im_name)

    # Load input image
    im_file = os.path.join(FLAGS.data_dir, 'images', im_name)
    im = cv2.imread(im_file)

    # Add alpha channel
    (B, G, R) = cv2.split(im)
    A = np.empty([np.size(im, 0), np.size(im, 1)], dtype=im.dtype)
    A.fill(255)
    im = cv2.merge((B, G, R, A))

    # Select bounding boxes in the given image.
    subset_df = df[df['image_name'] == im_name]

    # print '%d object(s) are detected:' % len(subset_df.index)
    # for bbox_idx, _ in subset_df.iterrows():
    #     print ' - bbox ID: %d' % bbox_idx

    # Composite rendered images.
    if FLAGS.composite_rendered:
        for bbox_idx, row in subset_df.iterrows():
            im = composite_rendered_images(im, bbox_idx, row, num_digits)

    # Draw bounding boxes.
    if FLAGS.draw_bboxes:
        for bbox_idx, row in subset_df.iterrows():
            im = draw_bboxes(im, row, class_names, colors)

    if FLAGS.half_size_output:
        height, width = im.shape[:2]
        im = cv2.resize(im, (width/2, height/2),
                        interpolation = cv2.INTER_CUBIC)

    cv2.imwrite(out_im_file, im)
    print('Saved %s.' % out_im_file)


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read image names.
    im_names = [os.path.basename(x) for x in
                glob.glob(os.path.join(FLAGS.data_dir, 'images', '*.png'))]
    im_names.sort()

    # Read class names to be detected.
    class_names = cnn_utils.read_class_names(FLAGS.class_name_file)

    # Read bounding boxes.
    df, num_digits = cnn_utils.read_bboxes(
        os.path.join(FLAGS.data_dir, FLAGS.bbox_file))

    if not os.path.exists(os.path.join(
            FLAGS.data_dir, FLAGS.out_composite_dir)):
        os.makedirs(os.path.join(
            FLAGS.data_dir, FLAGS.out_composite_dir))

    # Generate random colors.
    np.random.seed(0)
    colors = np.random.rand(len(class_names), 4) * 255
    colors = colors.astype(int)
    colors[:, 3] = 255

    # for im_name in im_names:
    #     generate_output_images(im_name, df, class_names, colors, num_digits)

    # Parallel processing.
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(
        generate_output_images)(im_name, df, class_names, colors, num_digits)
                                         for im_name in im_names)
