#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

# Python 2/3 compatibility
from __future__ import print_function
import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../'))
sys.path.append(os.path.join(BASE_DIR, 'script'))

import gflags
import glob
import image_list
import numpy as np


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir',
                     '/Users/msung/Developer/data/sun3d.cs.princeton.edu/data'
                     '/mit_dorm_mcc_eflr6/dorm_mcc_eflr6_oct_31_2012_scan1_erika', '')
gflags.DEFINE_string('extrinsics_dir', 'extrinsics', '')
gflags.DEFINE_string('out_camera_matrics_dir', 'pose', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read image file names.
    im_names = image_list.get_image_filenames(
        os.path.join(FLAGS.data_dir, 'images', '*.png'))
    num_images = len(im_names)

    # Find extrinsics file.
    extrinsics_file_list = glob.glob(
        os.path.join(FLAGS.data_dir, FLAGS.extrinsics_dir, '*.txt'))
    # Find the most recent one.
    extrinsics_file_list.sort()
    assert (len(extrinsics_file_list) > 0)
    extrinsics_file = extrinsics_file_list[-1]
    print('Extrinsics file: {}'.format(extrinsics_file))

    # Read extrinsics file.
    extrinsics_all = np.genfromtxt(extrinsics_file, delimiter=' ')
    # Each 3 lines are an extrinsics matrix for an image.
    assert (extrinsics_all.shape[0] >= 3 * num_images)

    out_dir = os.path.join(FLAGS.data_dir, FLAGS.out_camera_matrics_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(num_images):
        extrinsics = np.identity(4)
        start_row = 3 * i + 0
        end_row = 3 * i + 3
        extrinsics[0:3, :] = extrinsics_all[start_row:end_row, :]
        print(extrinsics)

        im_name = os.path.splitext(im_names[i])[0]
        out_file = os.path.join(out_dir, im_name + '.txt')
        print(out_file)
        np.savetxt(out_file, extrinsics, delimiter=',')
