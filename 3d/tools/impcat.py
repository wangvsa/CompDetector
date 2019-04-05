#!/usr/bin/env python
# encoding: utf-8
import glob, sys


path = sys.argv[1]
end_files = glob.glob(path+"/*_200.npy")

def is_crash(start_file):
    end = start_file.replace("_100.npy", "_200.npy")
    if end not in end_files:
        return True
    return False


for bit in range(0, 64):
    crashed = 0
    start_files = glob.glob(path+"/error_"+str(bit)+"_*_100.npy")
    for f in start_files:
        if is_crash(f):
            crashed += 1
    print("bit: %s, crashed: %s, total: %s" %(bit, crashed, len(start_files)))
