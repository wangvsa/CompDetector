import random, math
import os, glob, sys
import numpy as np
from create_dataset import hdf5_to_numpy, get_flip_error
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
            d[x, y, z] = get_flip_error(org, 20)

        hasError = aid.detect(d, d1, d2, d3, d4, d5)
        if hasError and truth:      # true positive
            recall += 1
        if hasError and not truth:  # false positive
            aid.fp += 1
        aid.it += 1

        d[x, y, z] = org   # restore the correct value before next detection

        d5, d4, d3, d2, d1  = d4, d3, d2, d1, d
        print("it:", it, " recall:", recall, " fp:", aid.fp)

def test_k_recall(restart_iter, delay, clean_prefix, error_prefix):
    recall = np.zeros(delay)

    aid = AdaptiveDetector()
    aid.it = 0
    aid.fp = 200

    # first read previous clean data
    d5 = read_data(clean_prefix, restart_iter-5)
    d4 = read_data(clean_prefix, restart_iter-4)
    d3 = read_data(clean_prefix, restart_iter-3)
    d2 = read_data(clean_prefix, restart_iter-2)
    d1 = read_data(clean_prefix, restart_iter-1)

    for k in range(delay):      # detecting after k iterations
        d = read_data(error_prefix, k)
        recall[k] = aid.detect(d, d1, d2, d3, d4, d5)
        aid.it += 1

        d5 = d4
        d4 = d3
        d3 = d2
        d2 = d1
        d1 = d
    print recall
    return recall


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

#test_0_recall("/home/wangchen/Flash/StirTurb_3d/4/clean/stirturb_3d_hdf5_chk_")
#test_0_recall("/home/wangchen/Flash/Sedov_3d/0/clean/sedov_hdf5_chk_")
#test_0_recall("/home/wangchen/Flash/BlastBS_3d/4/clean/blastBS_mhd_hdf5_chk_")
#test_0_recall("/home/wangchen/Flash/Sod_3d/4/clean/sod_hdf5_chk_")
#test_fp("/home/wangchen/Flash/BlastBS_3d/4/clean/blastBS_mhd_hdf5_chk_")

if __name__ == "__main__":
    directory = sys.argv[1]
    files = glob.glob(directory+"/*chk*")
    total_iterations = len(files)
    prefix = files[0].split("chk_")[-2] + "chk_"
    test_0_recall(prefix, total_iterations)



# test k iterations
'''
if __name__ == "__main__":
    delay = 11
    recall = np.zeros(delay)
    total = 0
    # First find all restarting points
    directory = "/home/wangchen/Flash/OrszagTang/"
    for filename in glob.iglob(directory+"*plt_cnt_0000"):
        last_one = filename[:-4] + "0011"
        if os.path.isfile(last_one):
            restart_iter = int(filename.split("_")[1])
            error_prefix = filename[0:-4]
            clean_prefix = directory + "clean/orszag_mhd_2d_hdf5_plt_cnt_"
            print(restart_iter, error_prefix)
            recall += test_k_recall(restart_iter, delay, clean_prefix, error_prefix)
            total += 1.0
            print(recall)
            print(recall/total)
'''
