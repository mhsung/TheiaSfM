#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../'))
sys.path.append(os.path.join(BASE_DIR, '3rdparty', 'RenderForCNN'))
sys.path.append(os.path.join(BASE_DIR, '3rdparty', 'RenderForCNN', 'view_estimation'))
from global_variables import *
from caffe_utils import *

import math
import numpy as np
import random
import sys
import tempfile


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def viewpoint_best(img_filenames, class_idxs):
    batch_size = g_test_batch_size
    model_params_file = g_caffe_param_file
    model_deploy_file = g_caffe_deploy_file
    result_keys = g_caffe_prob_keys
    resize_dim = g_images_resize_dim
    image_mean_file = g_image_mean_file

    # ** NETWORK FORWARD PASS **
    probs_lists = batch_predict(model_deploy_file, model_params_file,
                                batch_size, result_keys, img_filenames,
                                image_mean_file, resize_dim)

    # EXTRACT PRED FROM PROBS
    num_images = len(img_filenames)

    # @mhsung
    # Store scores at the end.
    preds = np.zeros((num_images, len(result_keys) + 1))
    for i in range(len(img_filenames)):
        class_idx = class_idxs[i]

        # pred is the class with highest prob within
        # class_idx*360~class_idx*360+360-1
        for k in range(len(result_keys) + 1):
            probs = probs_lists[k][i]
            probs = probs[class_idx * 360:(class_idx + 1) * 360]
            max_idx = probs.argmax()
            pred = max_idx + class_idx * 360
            preds[i, k] = pred % 360
            # Softmax.
            probs = softmax(probs)
            preds[i, -1] = probs[max_idx]

    return preds


def viewpoint_scores(img_filenames, class_idxs, output_result_files):
    batch_size = g_test_batch_size
    model_params_file = g_caffe_param_file
    model_deploy_file = g_caffe_deploy_file
    result_keys = g_caffe_prob_keys
    resize_dim = g_images_resize_dim
    image_mean_file = g_image_mean_file
    assert (len(result_keys) == 3)  # azimuth,elevation,tilt

    # ** NETWORK FORWARD PASS **
    probs_lists = batch_predict(model_deploy_file, model_params_file,
                                batch_size, result_keys, img_filenames,
                                image_mean_file, resize_dim)

    # EXTRACT PRED FROM PROBS
    probs_3dview_all = []
    for k in range(len(img_filenames)):
        probs_3dview_all.append([])
    for i in range(len(img_filenames)):
        class_idx = class_idxs[i]

        # probs_3dview is length-3 list of azimuth, elevation and tilt probs (each is length-360 list)
        probs_3dview = []
        for k in range(len(result_keys)):
            # probs for class_idx:
            # class_idx*360~class_idx*360+360-1
            probs = probs_lists[k][i]
            probs = probs[class_idx * 360:(class_idx + 1) * 360]
            # Softmax.
            probs = softmax(probs)
            probs_3dview.append(probs)
        probs_3dview_all[i] = probs_3dview

    for i in range(len(img_filenames)):
        print 'Saving "' + output_result_files[i] + '.npy"'
        np.save(output_result_files[i], probs_3dview_all[i])

