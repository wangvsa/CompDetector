#!/usr/bin/env python
# encoding: utf-8
import glob, sys
import numpy as np
from create_dataset import get_flip_error, hdf5_to_numpy, read_data


path = sys.argv[1]
end_files = glob.glob(path+"/*_200.npy")

def is_crash(start_file):
    end = start_file.replace("_100.npy", "_200.npy")
    if end not in end_files:
        return True
    return False

def diff(clean, error):
    d1 = read_data(clean)
    d2 = read_data(error)
    mse = (np.square(d1 -d2)).mean(axis=None)
    diff_count = np.sum(d1 != d2)
    diff_max = np.max(np.abs(d1 - d2))
    diff_rel_max = np.max(np.abs((d1 - d2)/d1))
    return mse, diff_count, diff_max, diff_rel_max


# crash rate against bit flipped
def crash_rate():
    path = sys.argv[1]
    for bit in range(0, 64):
        crashed = 0
        start_files = glob.glob(path+"/error_"+str(bit)+"_*_100.npy")
        for f in start_files:
            if is_crash(f):
                crashed += 1
        print("bit: %s, crashed: %s, total: %s" %(bit, crashed, len(start_files)))

def error_impact():
    path = sys.argv[1]
    clean_end_file = sys.argv[2]
    for bit in range(0, 10):
        mse, diff_count, diff_max, diff_rel_max = 0, 0, 0, 0
        error_end_files = glob.glob(path+"/error_"+str(bit)+"_*_200.npy")
        for error_end_file in error_end_files:
            t1, t2, t3, t4 = diff(clean_end_file, error_end_file)
            mse += t1
            diff_count += t2
            diff_max += t3
            diff_rel_max += t4
            print t1, t2, t3, t4
        N = len(error_end_files)
        if N  == 0:
            print("bit: %s, N: 0" %bit)
        else:
            print("bit: %s, N: %s, mse: %s, diff count: %s, diff max: %s, diff max rel: %s" %(bit, N, mse/N, diff_count/N, diff_max/N, diff_rel_max/N) )

error_impact()
