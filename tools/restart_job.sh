#!/bin/bash

# Sod
cd /home/wangchen/test/
cp /home/wangchen/sources/CompDetector/tools/restart_comet.py ./
cp /home/wangchen/sources/CompDetector/create_dataset.py ./
python ./restart_comet.py sod_hdf5_chk_ 190 191
