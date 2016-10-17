#!/usr/bin/env bash

# Change variables.
DATA_ADDRESS=data/mit_dorm_mcc_eflr6/dorm_mcc_eflr6_oct_31_2012_scan1_erika
DATA_DIR=$HOME/home/data/sun3d.cs.princeton.edu/$DATA_ADDRESS
./get_sun3d_data.py --data_address=${DATA_ADDRESS}

