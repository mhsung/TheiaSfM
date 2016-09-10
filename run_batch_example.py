#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import gflags
import parallel_exec
import os
import sys


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('dataset_dir',
                     '/Users/msung/Developer/data/7-scenes/office', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    print('Run batch jobs...')
    cmd_list = []
    for data_name in os.listdir(FLAGS.dataset_dir):
        data_dir = os.path.join(FLAGS.dataset_dir, data_name)
        if not os.path.isdir(data_dir) or \
                not os.path.isdir(os.path.join(data_dir, 'images')):
            continue

        if data_name != 'seq-01':
            continue

        '''
        Extract camera params from ground truth.
        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'build', 'bin', 'exp_export_orientations')) + ' \\\n'
        cmd += '--input_data_type=pose' + ' \\\n'
        cmd += '--input_filepath=' + os.path.join(data_dir, 'pose') + ' \\\n'
        cmd += '--output_filepath=' + os.path.join(data_dir, 'camera_params')\
               + ' \\\n'
        cmd += '--logtostderr'
        '''

        # Run TheiaSfM.
        '''
        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_exp.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--track_features --clean'
        print(cmd)
        os.system(cmd)

        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_exp.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--track_features --use_initial_orientations --clean'
        print(cmd)
        os.system(cmd)
        '''

        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_exp.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--track_features'
        print(cmd)
        os.system(cmd)

        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'script', 'run_exp.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--track_features --use_initial_orientations'
        print(cmd)
        os.system(cmd)


        # Run plot.
        cmd = ''
        cmd += os.path.normpath(os.path.join(
            sys.path[0], 'plot_camera_params.py')) + ' \\\n'
        cmd += '--data_dir=' + data_dir
        print(cmd)
        os.system(cmd)
        '''

        # print(cmd)
        # cmd_list.append((data_name, cmd))

    # parallel_exec.run_parallel('TheiaSfM', cmd_list)
