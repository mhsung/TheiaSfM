#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import gflags
import glob
import os
import shutil
import sys


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_dir',
                     '/afs/cs.stanford.edu/u/mhsung/home/data/sun3d.cs.princeton.edu/', '')
gflags.DEFINE_string('domain_address', 'http://sun3d.cs.princeton.edu', '')
gflags.DEFINE_string('data_address', '/data/hotel_umd/maryland_hotel3', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    cwd = os.getcwd()

    # Move to the data root directory.
    os.chdir(FLAGS.data_dir)
    print(os.getcwd())

    # Download data.
    cmd = "wget -r -nH --no-parent -R index.html*,*.gif "
    cmd += (FLAGS.domain_address + os.path.join(FLAGS.data_address, 'image'))
    print(cmd)
    os.system(cmd)

    # Move to the specific data directory.
    os.chdir(FLAGS.data_dir + '/' + FLAGS.data_address)
    print(os.getcwd())

    image_names = [os.path.splitext(os.path.basename(x))[0]
                   for x in glob.glob(os.path.join('image', '*.jpg'))]
    num_images = len(image_names)

    frames = [x.split('-')[0] for x in image_names]

    shutil.move('image', 'jpg')
    out_dir = 'images'
    os.makedirs(out_dir)

    for i in range(num_images):
        in_file = os.path.join('jpg', image_names[i] + '.jpg')
        out_file = os.path.join('images', frames[i] + '.png')
        cmd = 'convert ' + in_file + ' ' + out_file
        print(cmd)
        os.system(cmd)

    os.chdir(cwd)
