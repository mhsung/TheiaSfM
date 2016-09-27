#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import glob
import os


def get_image_filenames(images_wildcard):
    data_dir = os.path.join(os.path.dirname(images_wildcard), '../')
    out_filename = os.path.join(data_dir, 'image_filenames.txt')

    if os.path.exists(out_filename):
        with open(out_filename, 'r') as f:
            image_filenames = f.read().splitlines()
    else:
        image_filenames = [os.path.basename(x)
                           for x in glob.glob(images_wildcard)]

        # Assume that the image sequence is given in the sorted order.
        image_filenames.sort()

        with open(out_filename, 'w') as f:
            for image_filename in image_filenames:
                f.write('{}\n'.format(image_filename))

    return image_filenames
