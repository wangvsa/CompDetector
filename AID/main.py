import random, math
import os, glob, sys
import numpy as np
from create_dataset import get_flip_error
from aid import AdaptiveDetector

def read_data(prefix, it):
    filename = prefix + ("0000"+str(it))[-4:] + ".npy"
    data = np.load(filename)
    return data

'''
Test the recall of the AID
We insert an error in each frame and see if AID can detect it
'''
def test_0_recall(prefix, total_iterations):
    aid = AdaptiveDetector()
    aid.fp = 42

    d5 = read_data(prefix, 0)
    d4 = read_data(prefix, 1)
    d3 = read_data(prefix, 2)
    d2 = read_data(prefix, 3)
    d1 = read_data(prefix, 4)

    # start from the 6th frame
    recall = 0
    for it in range(5, total_iterations):
        d = read_data(prefix, it)

        # insert an error
        x, y, z = random.randint(0, d.shape[0]-1), random.randint(0, d.shape[1]-1), random.randint(0, d.shape[2]-1)
        org = d[x, y, z]
        truth = False
        if it % 2 == 0:
            truth = True
            d[x, y, z] = get_flip_error(org, 13)

        hasError = aid.detect(d, d1, d2, d3, d4, d5)
        if hasError and truth:      # true positive
            recall += 1
        if hasError and not truth:  # false positive
            aid.fp += 1
        aid.it += 1

        d[x, y, z] = org   # restore the correct value before next detection

        d5, d4, d3, d2, d1  = d4, d3, d2, d1, d
        print("it:", it, " recall:", recall, " fp:", aid.fp)

def evaluate_detection(timestep, clean_prefix, path):
    recall = 0
    aid = AdaptiveDetector()
    aid.order = 2
    aid.it = 5
    aid.fp = 18

    # first read previous clean data
    d5 = read_data(clean_prefix, timestep-5)
    d4 = read_data(clean_prefix, timestep-4)
    d3 = read_data(clean_prefix, timestep-3)
    d2 = read_data(clean_prefix, timestep-2)
    d1 = read_data(clean_prefix, timestep-1)

    for f in glob.glob(path+"/error*.npy"):
        d = read_data(clean_prefix, timestep)
        hasError = aid.detect(d, d1, d2, d3, d4, d5)
        print(f, hasError)
        recall += hasError

    print recall


# Perform the detection on clean data to get false positive rate
def test_fp(prefix):
    aid = AdaptiveDetector()

    d5 = read_data(prefix, 0)
    d4 = read_data(prefix, 1)
    d3 = read_data(prefix, 2)
    d2 = read_data(prefix, 3)
    d1 = read_data(prefix, 4)

    # start from the 6th frame
    for it in range(5, 1001):
        d = read_data(prefix, it)
        aid.fp += aid.detect(d, d1, d2, d3, d4, d5)
        aid.it += 1
        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d
        print("it:", it, " fp:", aid.fp)

if __name__ == "__main__":
    directory = sys.argv[1]
    files = glob.glob(directory+"/*chk*")
    total_iterations = len(files)
    prefix = files[0].split("chk_")[-2] + "chk_"
    test_0_recall(prefix, total_iterations)

    #evaluate_detection(100, "/home/wangchen/Flash/SC19/Sod_3d/clean/sod_hdf5_chk_", sys.argv[1])
    #evaluate_detection(100, "/home/wangchen/Flash/BlastBS_3d/4/clean/blastBS_mhd_hdf5_chk_", sys.argv[1])
