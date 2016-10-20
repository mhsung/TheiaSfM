#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import gflags
import parallel_exec
import os
import sys


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('dataset_dir', '', '')
gflags.DEFINE_string('image_wildcard', '', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    print('Run batch jobs...')
    cmd_list = []
    for data_name in os.listdir(FLAGS.dataset_dir):
        data_dir = os.path.join(FLAGS.dataset_dir, data_name)
        if not os.path.isdir(data_dir) or \
                not os.path.isdir(os.path.join(data_dir, 'images')):
            continue

        print ('================')
        print ('Data name: {}'.format(data_name))

        # Run convolutional neural networks.
        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_convnet.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--image_wildcard=' + FLAGS.image_wildcard
        print(cmd)
        os.system(cmd)

        # Run TheiaSfM.
        cmd_list = []

        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_exp.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--track_features'
        print(cmd)
        # os.system(cmd)
        cmd_list.append((data_name, cmd))

        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_exp.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--track_features --use_initial_orientations'
        print(cmd)
        # os.system(cmd)
        cmd_list.append((data_name, cmd))

        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_exp.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--track_features --use_gt_orientations'
        print(cmd)
        # os.system(cmd)
        cmd_list.append((data_name, cmd))

        parallel_exec.run_parallel('TheiaSfM', cmd_list)
        print ('================')
        print ('\n')
