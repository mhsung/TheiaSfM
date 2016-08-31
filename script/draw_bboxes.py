#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import cv2
import gflags
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import os
import sys


FLAGS = gflags.FLAGS
gflags.DEFINE_string('images_dir', '', '')
gflags.DEFINE_string('image_bboxes_dir', '', '')
gflags.DEFINE_string('output_dir', '', '')


def draw_bboxes(image_filename):
    input_image_file = os.path.join(FLAGS.images_dir, image_filename)
    im = cv2.imread(input_image_file)

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    basename = os.path.splitext(image_filename)[0]
    sub_bboxes_dirs = [os.path.join(FLAGS.image_bboxes_dir, x)
                       for x in os.listdir(FLAGS.image_bboxes_dir)]

    prism = cm = plt.get_cmap('prism')
    cnorm  = colors.Normalize(vmin=0, vmax=len(sub_bboxes_dirs))
    scalar_map = cmx.ScalarMappable(norm=cnorm, cmap=prism)

    for i in range(len(sub_bboxes_dirs)):
        bboxes_dir = sub_bboxes_dirs[i]
        if not os.path.isdir(bboxes_dir):
            continue

        bbox_file = os.path.join(bboxes_dir, basename + '.txt')
        if (not os.path.isfile(bbox_file)):
            continue

        # Read bounding box.
        dets = np.genfromtxt(bbox_file, delimiter=' ', dtype=float)
        bbox = dets[:4]
        score = dets[-1]

        color = scalar_map.to_rgba(i)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=color, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(os.path.basename(bboxes_dir)),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=20, color='white')

    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    output_image_file = os.path.join(FLAGS.output_dir, image_filename)
    plt.savefig(output_image_file, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    FLAGS(sys.argv)

    image_filenames = [os.path.basename(x) for x in glob.glob(os.path.join(
        FLAGS.images_dir, '*.png'))]

    for image_filename in image_filenames:
        draw_bboxes(image_filename)
