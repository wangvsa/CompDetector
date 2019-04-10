#!/usr/bin/env python
# encoding: utf-8

import sys
import h5py
import glob
import numpy as np
from create_dataset import get_flip_error, hdf5_to_numpy, read_data


def diff(clean, error):
    d1 = read_data(clean)
    d2 = read_data(error)
    bit, x, y, z = error.split("_")[-5:-1]
    bit, x, y, z = int(bit), int(x), int(y), int(z)
    print("bit: %s, x: %s, y: %s, z: %s" %(bit, x, y ,z))
    print("clean: %s, error: %s" %(d1[x,y,z], d2[x,y,z]))
    sum_error = np.sum(np.abs(d1 - d2))
    diff_count = np.sum(d1 != d2)
    diff_max = np.max(np.abs(d1 - d2))
    return sum_error, diff_count, diff_max

sum_error, diff_count, diff_max = diff(sys.argv[1], sys.argv[2])
print("sum of abs(difference)", sum_error)
print("number of different point: ", diff_count)
print("max difference point:", diff_max)
