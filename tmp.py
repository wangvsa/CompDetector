import os, sys, random, h5py
import bitstring, math
import numpy as np
import glob
from skimage.util.shape import view_as_windows, view_as_blocks

def test_sz(origin_file, decompressed_file):
    origin = np.fromfile(origin_file, dtype=np.double)
    decompressed = np.fromfile(decompressed_file, dtype=np.double)
    print np.mean(np.abs(origin-decompressed))

# Flip a bit of a given position
# val: input, python float(64 bits), equals double in C
def bit_flip(val, pos):
    bin_str = bitstring.BitArray(float=val, length=64).bin
    l = list(bin_str)
    l[pos] = '1' if l[pos] == '0' else '0'
    flipped_str = "".join(l)
    return bitstring.BitArray(bin=flipped_str).float

# Read hdf5 file, return an array shape of (NX, NY)
def hdf5_to_numpy(filename, var_name="dens"):
    f = h5py.File(filename, 'r')
    data = f[var_name][:]
    if data.ndim == 4:
        data = data[0, 0]
    return data

def get_flip_error(val):
    while True :
        pos = random.randint(0, 20)
        error = bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            break
    error = min(10e+3, error)
    error = max(-10e+3, error)
    return error

def split_to_windows(frame, rows, cols, overlap):
    step = cols - overlap
    windows = view_as_windows(frame, (rows, cols), step = step)
    return np.vstack(windows)

def process(filename):
    data = hdf5_to_numpy(filename)
    print("read file finished")

    # Insert an error
    #x, y = random.randint(1, data.shape[0])-1, random.randint(1, data.shape[1])-1
    #data[x, y] = get_flip_error(data[x, y])
    #print("error:", x, y, data[x,y])

    # Save into a binary file
    filename = filename + ".dat"
    data.tofile(filename)
    print("save to", filename)

    # 1. Compress the data, which contains one error
    # The output will be xxx.sz
    #os.system("./sz -z -d -c sz.config -i " + filename + " -2 480 480")

    # 2. Decompress the data
    # The output file name would be xxx.sz.out
    #os.system("./sz -x -d -s " + filename + ".sz -2 480 480")

    #test_sz(filename, filename+".sz.out")

    # 3. Read back the decompressed data
    #decompressed = np.fromfile(filename+".sz.out", dtype=np.double).reshape(480, 480)
    #windows = split_to_windows(decompressed, 60, 60, 20)
    #np.save(filename+".npy", windows)
    #print "save to npy"

def combine_to_one_npy(directory):
    clean_dataset, error_dataset = [], []
    for filename in glob.iglob(directory+"/*plt_cnt_*"):
        if ".dat" not in filename:
            print filename
            clean_dataset.append(hdf5_to_numpy(filename))
        if ".out" in filename:
            print filename
            error_dataset.append(np.fromfile(filename, dtype=np.double).reshape(480, 480))
    clean = np.array(clean_dataset)
    error = np.array(error_dataset)
    print clean.shape, error.shape
    np.save("clean.npy", clean)
    np.save("error.npy", error)


if __name__ == "__main__":
    for filename in glob.iglob(sys.argv[1]+"/*plt_cnt_*"):
        if ".dat" not in filename:
            print filename
            process(filename)
    #combine_to_one_npy(sys.argv[1])
