#!/usr/bin/python
# Author: Minhyuk Sung (mhsung@cs.stanford.edu)

import datetime
import os

def save_and_run_cmd(cmd, filepath):
    print(cmd)
    with open(filepath, 'w') as f:
        f.write(cmd)
        f.write('\n')
    os.system('chmod +x ' + filepath)
    print("Saved '" + filepath + "'.")

    start = datetime.datetime.now()
    os.system(filepath)
    end = datetime.datetime.now()
    elapsed = end - start

    print('Start time: ' + start.strftime("%Y-%m-%d %H:%M:%S"))
    print('End time: ' + end.strftime("%Y-%m-%d %H:%M:%S"))
    print('Elapsed time: ' + str(end - start))
    print('')
