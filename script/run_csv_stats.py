#!/usr/bin/env python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import gflags
import math
import numpy as np
import os
import pandas as pd
import sys

FLAGS = gflags.FLAGS
gflags.DEFINE_string('csv_file', '', '')

if __name__ == '__main__':
    FLAGS(sys.argv)

    df = pd.read_csv(FLAGS.csv_file)
    df.drop([col for col in df.columns if 'Unnamed' in col],
            axis=1, inplace=True)
    df.drop('Dataset', axis=1, inplace=True)

    for col in df.columns:
        if 'Rotation' in col:
            df[col] *= (180.0 / math.pi)

    # Get count data frame.
    df_count = pd.DataFrame(df.groupby('Name').size().rename('Counts'))
    print(df_count)

    # Get mean data frame.
    df_mean = df.groupby('Name').mean()
    mean_file = os.path.splitext(FLAGS.csv_file)[0] + '_mean.csv'
    df_mean.to_csv(mean_file)
    print("Saved '{}'.".format(mean_file))

    # Get median data frame.
    df_median = df.groupby('Name').median()
    median_file = os.path.splitext(FLAGS.csv_file)[0] + '_median.csv'
    df_median.to_csv(median_file)
    print("Saved '{}'.".format(median_file))

