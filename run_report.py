#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import gflags
# import parallel_exec
import os
import sys


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('dataset_dir', '', '')
gflags.DEFINE_string('out_dir', './output', '')


if __name__ == '__main__':
    FLAGS(sys.argv)

    dataset_name = os.path.basename(FLAGS.dataset_dir)
    print(dataset_name)
    print(FLAGS.out_dir)
    tex_dir = os.path.join(FLAGS.out_dir, dataset_name)
    print(tex_dir)
    if not os.path.exists(tex_dir):
        os.makedirs(tex_dir)

    csv_file = os.path.join(FLAGS.out_dir, dataset_name + '.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)
    print(csv_file)

    convnet_file = os.path.join(FLAGS.out_dir, dataset_name + '_convnet.csv')
    if os.path.exists(convnet_file):
        os.remove(convnet_file)
    print(convnet_file)

    cwd = os.getcwd()

    print('Run batch jobs...')
    cmd_list = []
    for data_name in os.listdir(FLAGS.dataset_dir):
        data_dir = os.path.join(FLAGS.dataset_dir, data_name)
        if not os.path.isdir(data_dir) or \
                not os.path.isdir(os.path.join(data_dir, 'images')):
            continue

        print ('================')
        print ('Data name: {}'.format(data_name))

        tex_file = data_name + '.tex'

        cmd = ''
        cmd += './script/run_latex_report.py ' + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--out_file=' + os.path.join(tex_dir, tex_file)
        print(cmd)
        os.system(cmd)

        cmd = ''
        cmd += './script/run_csv_report.py ' + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--out_file=' + csv_file
        print(cmd)
        os.system(cmd)

        cmd = ''
        cmd += './script/run_convnet_report.py ' + ' \\\n'
        cmd += '--data_dir=' + data_dir + ' \\\n'
        cmd += '--out_file=' + convnet_file
        print(cmd)
        os.system(cmd)

        if not os.path.exists(os.path.join(tex_dir, tex_file)):
            print('Warning: generating tex file failed.')
            continue

        os.chdir(tex_dir)
        cmd = ''
        cmd += 'pdflatex ' + os.path.basename(tex_file)
        print(cmd)
        os.system(cmd)
        os.chdir(cwd)

    # Merge PDF files.
    os.chdir(tex_dir)
    cmd = 'pdfunite *.pdf ../' +  dataset_name + '.pdf'
    print(cmd)
    os.system(cmd)
    os.chdir(cwd)

    # Get stats.
    cmd = './script/run_csv_stats.py --csv_file=' + csv_file
    print(cmd)
    os.system(cmd)

