#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../'))
sys.path.append(os.path.join(BASE_DIR, 'script'))

from joblib import Parallel, delayed
import cnn_utils
import cv2
import gflags
import glob
import image_list
import multiprocessing
import numpy as np


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('class_name_file', os.path.join(
    BASE_DIR, 'script/RenderForCNN/class_names.txt'), '')
gflags.DEFINE_string('bbox_file', 'convnet/object_bboxes.csv', '')
gflags.DEFINE_string('orientation_file',
    'convnet/object_orientations_fitted.csv', '')
gflags.DEFINE_string('render_dir', 'convnet/object_render_fitted', '')
gflags.DEFINE_string('out_composite_dir', 'convnet/object_composite_fitted', '')

gflags.DEFINE_bool('with_object_index', True, '')
gflags.DEFINE_bool('composite_rendered', True, '')
gflags.DEFINE_bool('draw_bboxes', True, '')
gflags.DEFINE_bool('half_size_output', False, '')


def composite_rendered_images(im, bbox_idx, row, num_digits):
    render_im_file = os.path.join(FLAGS.data_dir, FLAGS.render_dir,
                                  str(bbox_idx).zfill(num_digits) + '.png')
    # print(render_im_file)
    assert (os.path.exists(render_im_file))
    render_im = cv2.imread(render_im_file, cv2.IMREAD_UNCHANGED)

    # Resize the rendered image.
    bbox_size_x = row['x2'] - row['x1']
    bbox_size_y = row['y2'] - row['y1']
    render_im_size_x = render_im.shape[1]
    render_im_size_y = render_im.shape[0]
    resize_ratio = min(bbox_size_x / render_im_size_x,
                       bbox_size_y / render_im_size_y)
    resized_half_size_x = int(round(0.5 * render_im_size_x * resize_ratio))
    resized_half_size_y = int(round(0.5 * render_im_size_y * resize_ratio))
    resized_render_im = cv2.resize(
        render_im, (2 * resized_half_size_x, 2 * resized_half_size_y),
        interpolation=cv2.INTER_CUBIC)

    # Compute offsets.
    im_size_x = im.shape[1]
    im_size_y = im.shape[0]
    bbox_center_x = int(round(0.5 * (row['x1'] + row['x2'])))
    bbox_center_y = int(round(0.5 * (row['y1'] + row['y2'])))
    sx = bbox_center_x - resized_half_size_x
    sy = bbox_center_y - resized_half_size_y
    ex = bbox_center_x + resized_half_size_x
    ey = bbox_center_y + resized_half_size_y
    offset_sx = max(0 - sx, 0)
    offset_sy = max(0 - sy, 0)
    offset_ex = max(ex - im_size_x, 0)
    offset_ey = max(ey - im_size_y, 0)

    resized_render_im = resized_render_im[
                        offset_sy:(2 * resized_half_size_y - offset_ey),
                        offset_sx:(2 * resized_half_size_x - offset_ex), :]
    sx = sx + offset_sx
    sy = sy + offset_sy
    ex = ex - offset_ex
    ey = ey - offset_ey
    if sx >= ex or sy >= ey:
        return im

    # Create mask
    (_, _, _, A) = cv2.split(resized_render_im)
    mask = A
    mask_inv = cv2.bitwise_not(mask)

    # Blend two images
    roi = im[sy:ey, sx:ex]
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(
        resized_render_im, resized_render_im, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    im[sy:ey, sx:ex] = dst
    return im


def draw_bboxes(im, row, pred, class_names):
    class_idx = row['class_index']
    class_name = class_names[class_idx]

    if FLAGS.with_object_index:
        # Use object index for color.
        object_idx = row['object_index']
        np.random.seed(object_idx)
    else:
        # Use class index for color.
        np.random.seed(class_idx)
    bbox_color = np.random.rand(1, 4) * 255
    bbox_color = bbox_color.astype(int)
    bbox_color[:, 3] = 255
    bbox_color = bbox_color[0]

    sx = max(int(round(row['x1'])), 0)
    sy = max(int(round(row['y1'])), 0)
    ex = min(int(round(row['x2'])), np.size(im, 1))
    ey = min(int(round(row['y2'])), np.size(im, 0))

    bbox_thickness = 8
    cv2.rectangle(im, (sx, sy), (ex, ey), bbox_color,
                  bbox_thickness)

    bbox_score = row['score']
    orientation_score = pred[-1]
    assert (len(pred) == 4)

    if FLAGS.with_object_index:
        label = '[{:d}] {:s} ({:.3f}, {:.3f})'.format(
                object_idx, class_name, bbox_score, orientation_score)
    else:
        label = '{:s} ({:.3f}, {:.3f})'.format(
                class_name, bbox_score, orientation_score)

    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    text_thickness = 2
    label_size = cv2.getTextSize(label, fontface, scale, text_thickness)[0]
    # Fill background of label (-1 thickness).
    cv2.rectangle(im, (sx - bbox_thickness / 2,
                       sy - bbox_thickness - label_size[1]),
                  (ex + bbox_thickness / 2, sy), bbox_color, -1)

    (major, minor, _) = cv2.__version__.split(".")
    if major == '2':
        cv_text_option = cv2.CV_AA
    elif major == '3':
        cv_text_option = cv2.LINE_AA
    else:
        print('Error: Unrecognized OpenCV version ({}.{}).'.format(
            major, minor))
        assert(major == '2' or major == '3')
    cv2.putText(im, label, (sx, sy - bbox_thickness / 2), fontface, scale,
                (255, 255, 255, 255), text_thickness, cv_text_option)

    return im


def generate_output_images(im_name, df, preds, class_names, num_digits):
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
            pred = preds[bbox_idx, :]
            im = draw_bboxes(im, row, pred, class_names)

    if FLAGS.half_size_output:
        height, width = im.shape[:2]
        im = cv2.resize(im, (width/2, height/2),
                        interpolation = cv2.INTER_CUBIC)

    cv2.imwrite(out_im_file, im)
    print('Saved %s.' % out_im_file)


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read image names.
    im_names = image_list.get_image_filenames(
            os.path.join(FLAGS.data_dir, 'images', '*.png'))

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
            FLAGS.data_dir, FLAGS.out_composite_dir)):
        os.makedirs(os.path.join(
            FLAGS.data_dir, FLAGS.out_composite_dir))

    # for im_name in im_names:
    #     generate_output_images(im_name, df, class_names, num_digits)

    # Parallel processing.
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(
        generate_output_images)(im_name, df, preds, class_names, num_digits)
                                         for im_name in im_names)
