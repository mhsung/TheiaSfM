#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

# Python 2/3 compatibility
from __future__ import print_function
import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../../'))
sys.path.append(os.path.join(BASE_DIR, 'script'))
sys.path.append(os.path.join(BASE_DIR, 'script', 'tools'))

import gflags
import image_list
import math
import numpy as np
import transformations


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir', '', '')
gflags.DEFINE_string('extrinsics_file', '', '')
gflags.DEFINE_string('out_camera_matrics_dir', 'pose', '')



def inv_affine(T):
    R = T[:3, :3]
    t = T[:3, 3]
    inv_R = np.transpose(R)
    inv_t = np.dot(inv_R, -t)
    inv_T = np.identity(4)
    inv_T[:3, :3] = inv_R
    inv_T[:3, 3] = inv_t
    return inv_T


if __name__ == '__main__':
    FLAGS(sys.argv)

    # Read image file names.
    im_names = image_list.get_image_filenames(
        os.path.join(FLAGS.data_dir, 'images', '*.png'))
    num_images = len(im_names)

    # Read extrinsics file.
    extrinsics_lines = np.genfromtxt(
        os.path.join(FLAGS.data_dir, FLAGS.extrinsics_file), delimiter=' ')
    assert (len(extrinsics_lines) >= num_images)

    out_dir = os.path.join(FLAGS.data_dir, FLAGS.out_camera_matrics_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(num_images):
        # Format: [timestamp tx ty tz qx qy qz qw]
        # [tx ty tz]
        t = extrinsics_lines[i, 1:4]
        # [qw qx qy qz]
        # NOTE:
        # Use -angle for axes conversion.
        q = np.hstack((-extrinsics_lines[i, -1], extrinsics_lines[i, 4:7]))
        R = transformations.quaternion_matrix(q)[:3, :3]

        # Apply converted axes to translation.
        axes_converter = np.identity(4)
        axes_converter[1, 1] = +1.0
        axes_converter[2, 2] = -1.0
        t = np.dot(axes_converter[:3, :3], t)

        camera_to_scene = np.identity(4)
        camera_to_scene[:3, :3] = R
        camera_to_scene[:3, 3] = t

        scene_to_camera = inv_affine(camera_to_scene)

        im_name = os.path.splitext(im_names[i])[0]
        out_file = os.path.join(out_dir, im_name + '.txt')
        print(out_file)
        np.savetxt(out_file, scene_to_camera, fmt='%f', delimiter=',')
