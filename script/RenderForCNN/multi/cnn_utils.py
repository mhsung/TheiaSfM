#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import numpy as np
import pandas as pd

def read_class_names(filename):
    with open(filename) as f:
        class_names = f.read().splitlines()
        # Remove empty strings.
        class_names = filter(None, class_names)
    print('== Class names to be detected ==')
    print(class_names)
    return class_names

def create_bbox_data_frame():
    header = ('image_name', 'class_index',
              'x1', 'y1', 'x2', 'y2', 'score')
    df = pd.DataFrame(columns=header)
    return df


def read_bboxes(filename):
    header = ('image_name', 'class_index',
              'x1', 'y1', 'x2', 'y2', 'score')
    df = pd.read_csv(filename, names=header)
    num_bboxes = len(df.index)
    print ('{:d} bounding box(es) are loaded.'.format(num_bboxes))
    num_digits = len(str(num_bboxes))
    return df, num_digits


def read_orientations(filename):
    preds = np.genfromtxt(filename, delimiter=',')
    return preds
