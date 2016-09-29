#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import gflags
import glob
import os
import multiprocessing
import requests
import sys


FLAGS = gflags.FLAGS

# Set input files.
gflags.DEFINE_string('data_root_dir',
                     os.path.expanduser("~") +
                     '/home/data/sun3d.cs.princeton.edu/', '')
gflags.DEFINE_string('domain_address', 'http://sun3d.cs.princeton.edu/', '')
gflags.DEFINE_string('data_address',
                     'data/mit_dorm_mcc_eflr6/dorm_mcc_eflr6_oct_31_2012_scan1_erika', '')


def listFD(url, ext):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a')
            if node.get('href').endswith(ext)]


def download_file(file_url, out_dir):
    cmd = 'wget ' + file_url + ' -P ' + out_dir
    print(cmd)
    os.system(cmd)


def convert_jpg_to_png(image_name):
    frame = image_names.split('-')
    in_file = os.path.join('jpg', image_name + '.jpg')
    out_file = os.path.join('images', frame + '.png')
    cmd = 'convert ' + in_file + ' ' + out_file
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    FLAGS(sys.argv)

    cwd = os.getcwd()

    # Move to the data root directory.
    url = FLAGS.domain_address + FLAGS.data_address
    data_dir = os.path.join(FLAGS.data_root_dir, FLAGS.data_address)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    os.chdir(data_dir)
    print('Source: {}'.format(url))
    print('Destination: {}'.format(os.getcwd()))

    # Get jpg file list.
    jpg_files = listFD(url + '/image', 'jpg')
    print(jpg_files)
    if not os.path.exists('jpg'):
        os.makedirs('jpg')

    # Download data.
    # for f in jpg_files:
    #     download_file(f, 'jpg')
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(
        download_file)(f, 'jpg') for f in jpg_files)

    # Convert jpg to png.
    if not os.path.exists('images'):
        os.makedirs('images')

    image_names = [os.path.splitext(os.path.basename(x))[0]
                   for x in glob.glob(os.path.join('jpg', '*.jpg'))]

    # for image_name in image_names:
    #     convert_jpg_to_png(image_name)
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(
        convert_jpg_to_png)(image_name) for image_name in image_names)

    os.chdir(cwd)
