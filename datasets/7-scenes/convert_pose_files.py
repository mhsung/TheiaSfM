#!/usr/bin/python
# Minhyuk Sung (mhsung@cs.stanford.edu)

import glob
import os
import pandas as pd
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Required arguments:')
        print(' - Data path')
        sys.exit(-1)

    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print("Warning: Data directory does not exist: '{}'.".format(data_dir))
        sys.exit(-1)

    files = glob.glob(os.path.join(data_dir, 'poses', '*.txt'))
    for file in files:
        pose = pd.read_csv(file, sep='\t', header=None)
        if pose.shape[0] >= 4 and pose.shape[1] >= 4:
            pose.to_csv(file, header=None, index=None)
            print('Converted "' + file +'"')

