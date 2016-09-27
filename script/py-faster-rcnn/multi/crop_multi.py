#!/usr/bin/env python

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../../3rdparty/py-faster-rcnn'))
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, 'tools'))
sys.path.append('../../RenderForCNN/multi')

from joblib import Parallel, delayed
from PIL import Image
from utils.timer import Timer
import cnn_utils
import gflags
import multiprocessing


# 'data_dir' must have 'images' directory including *.png files.
FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('bbox_file', 'convnet/bboxes.csv', '')
gflags.DEFINE_string('out_crop_dir', 'convnet/crop', '')


def crop_image(bbox_idx, row, num_digits):
    im_name = str(row['image_name'])
    x1 = int(round(row['x1']))
    y1 = int(round(row['y1']))
    x2 = int(round(row['x2']))
    y2 = int(round(row['y2']))

    im_file = os.path.join(FLAGS.data_dir, 'images', im_name)
    im = Image.open(im_file)
    crop_im = im.crop((x1, y1, x2, y2))
    crop_im_file = os.path.join(FLAGS.data_dir, FLAGS.out_crop_dir,
                                str(bbox_idx).zfill(num_digits) + '.png')
    crop_im.save(crop_im_file)
    print('Saved %s.' % crop_im_file)


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read bounding boxes.
    df, num_digits = cnn_utils.read_bboxes(
        os.path.join(FLAGS.data_dir, FLAGS.bbox_file))

    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.out_crop_dir)):
        os.makedirs(os.path.join(FLAGS.data_dir, FLAGS.out_crop_dir))

    timer = Timer()
    timer.tic()

    # for bbox_idx, row in df.iterrows():
    #     crop_image(bbox_idx, row, num_digits)

    # Parallel processing.
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(
        crop_image)(bbox_idx, row, num_digits)
                                         for bbox_idx, row in df.iterrows())

    timer.toc()
    print ('Cropping took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, len(df.index))
