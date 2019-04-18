#!/usr/bin/env python
# encoding: utf-8
import glob, sys, os
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
    diff_count = np.sum(np.abs(d1-d2)>10e-5) + np.sum(np.isnan(d2))
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
    for bit in range(0, 31):
        mse, diff_count, diff_max, diff_rel_max = [], [], [], []
        error_end_files = glob.glob(path+"/error_"+str(bit)+"_*_end.npy")
        error_start_files = glob.glob(path+"/error_"+str(bit)+"_*_start.npy")

        completion = len(error_end_files)
        total = len(error_start_files)
        crashed = total - completion

        explicit_malignant, implicit_malignant = float(crashed), 0.0

        for error_end_file in error_end_files:
            t1, t2, t3, t4 = diff(clean_end_file, error_end_file)
            mse.append(t1)
            diff_count.append(t2)
            diff_max.append(t3)
            diff_rel_max.append(t4)
            if np.isnan(t4):
                explicit_malignant += 1.0
            elif t4 > 0.01:
                implicit_malignant += 1.0
            #print t1, t2, t3, t4
        if completion == 0:
            #print("bit: %s, completion: 0" %bit)
            pass
        else:
            #print("bit: %s, completion: %s, total: %s, mse: %s, diff count: %s, diff max: %s, diff max rel: %s" \
            #        %(bit, completion, len(error_start_files), sum(mse)/completion, sum(diff_count)/completion,\
            #            sum(diff_max)/completion, sum(diff_rel_max)/completion))
            #print("Bit: %s, MSE: %s, MAE: %s, MRE: %s, PAP: %.3f" \
            #        %(bit, sum(mse)/completion, sum(diff_max)/completion, \
            #            sum(diff_rel_max)/completion, sum(diff_count)/262144.0/completion))
            pass
        #print("%s\t%s\t%s, crashed: %s" %((explicit_malignant+implicit_malignant)/total, explicit_malignant/total, implicit_malignant/total, crashed))
        print("%s\t%s\t%s" %(bit, explicit_malignant/total, implicit_malignant/total))


# Get all maglinant error samples
def get_malignant_errors():
    path = sys.argv[1]
    clean_end_file = sys.argv[2]

    count = 0
    for error_end_file in glob.glob(path+"/*_200.npy"):
        start_file = error_end_file.replace("_200.npy", "_100.npy")
        # Validate the end checkpoint
        mse, diff_count, abs_error, rel_error = diff(clean_end_file, error_end_file)
        if rel_error > 0.01 or np.isnan(mse):
            print start_file
            count += 1
            os.system("cp "+start_file+" /home/wangchen/Flash/SC19/StirTurb/malignant/")
        else:
            os.system("cp "+start_file+" /home/wangchen/Flash/SC19/StirTurb/benign/")
    print count
error_impact()
#get_malignant_errors()
