import os, sys, random, h5py
import bitstring, math
import numpy as np
import glob
from skimage.util.shape import view_as_windows, view_as_blocks
NX, NY, NZ = 16, 16, 16

# See the difference of clean data and decompressed data with one error
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

# Read hdf5 file, return an array shape of (NX, NY, NZ)
def hdf5_to_numpy(filename, var_name="dens"):
    f = h5py.File(filename, 'r')
    data = f[var_name][0]   # shape of NZ, NY, NX
    data = np.swapaxes(data, 0, 2)  # swap NZ, NX axes -> NX, NY, NZ
    return data

def get_flip_error(val, bits = 20, threshold = None):
    attempt = 0
    while True :
        attempt += 1
        pos = random.randint(1, bits)
        error = bit_flip(val, pos)
        if not math.isnan(error) and not math.isinf(error):
            if threshold is None or abs(error-val) > threshold:
                break
        if attempt > 100:
            error = 2 * val
            break
    #if abs(error) <= 1e-3:
    #    error = 0
    error = min(10e1, error)
    error = max(-10e1, error)
    return error

def split_to_windows(frame, rows, cols, overlap):
    step = cols - overlap
    windows = view_as_windows(frame, (rows, cols), step = step)
    return np.vstack(windows)

def split_to_blocks(frame):
    blocks = view_as_blocks(frame, (NX, NY, NZ))    # (8,8,8,16,16,16)
    d = np.vstack(blocks)                           # (64,8,16,16,16)
    d = np.vstack(d)                                # (512,16,16,16)
    return d

def create_error_file(filename):
    # 1. Read checkpoint hdf5 files
    data = hdf5_to_numpy(filename)
    print("1. Read file", filename)

    # 2. Insert an error
    x, y = random.randint(1, data.shape[0])-1, random.randint(1, data.shape[1])-1
    data[x, y] = get_flip_error(data[x, y])
    print("2. Insert error:", x, y, data[x,y])

    # 3. Save into a binary file so we can compress it
    filename = filename + ".error.dat"
    data.tofile(filename)
    print("3. Save to binary", filename)

    # 4. Compress the data, which contains one error
    # The output will be xxx.sz
    os.system("./sz -z -d -c sz.config -i " + filename + " -2 480 480")

    # 5. Decompress the data
    # The output file name would be xxx.sz.out
    os.system("./sz -x -d -s " + filename + ".sz -2 480 480")

    #test_sz(filename, filename+".sz.out")

    # 6. Read back the decompressed data
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

def create_error_dataset(data_dir):
    for filename in glob.iglob(data_dir+"/*plt_cnt_*"):
        if ".dat" not in filename:
            create_error_file(filename)

# Read from split_data directory
# Read window by window and inject one error into each window
def create_split_compressed_error_dataset(data_dir):
    for filename in glob.iglob(data_dir+"/*clean.dat"):
        # 1. Read clean data
        data = np.fromfile(filename, dtype=np.double).reshape(NX, NY)

        # 2. Insert an error
        x, y = random.randint(40, data.shape[0])-20, random.randint(40, data.shape[1])-20
        data[x, y] = get_flip_error(data[x, y])
        #print("2. Insert error:", x, y, data[x,y])

        # 3. Save into a binary file so we can compress it
        # replace error in the path, so later the file will be
        # automatically decompressed to the error directory :)
        filename = filename.replace("clean", "error")
        data.tofile(filename)
        #print("3. Save to binary", filename)

        # 4. Compress the data, which contains one error
        # The output will be xxx.sz
        os.system("./sz -z -d -c sz.config -i " + filename + " -2 "+str(NX)+" "+str(NY))

        # 5. Decompress the data
        # The output file name would be xxx.sz.out
        os.system("./sz -x -d -s " + filename + ".sz -2 "+str(NX)+" "+str(NY))

        os.system("rm " + filename)
        os.system("rm " + filename+".sz")

# Read from unsplit_data/error directory
# where each file contains only one error
def create_split_error_testset(data_dir):
    for filename in glob.iglob(data_dir+"/*plt_cnt_*.sz.out"):
        print filename
        data = np.fromfile(filename, dtype=np.double).reshape(480, 480)
        windows = split_to_windows(data, NX, NY, 20)
        for i in range(windows.shape[0]):
            output_filename = filename + "." + str(i) +".error.dat"
            windows[i].tofile(output_filename)   # Save to binary format

# Read from FLASH directory(hdf5 files) or unsplit_data directory(binary files)
def create_split_clean_dataset(data_dir, output_dir):
    #file_list = glob.glob(data_dir+"/*plt_cnt_*")
    file_list = glob.glob(data_dir+"/*chk*")
    file_list.sort()
    count = 0
    for filename in file_list:
        if count % 20 != 0:
            count += 1
            continue

        dens = hdf5_to_numpy(filename, "dens")
        pres = hdf5_to_numpy(filename, "pres")
        temp = hdf5_to_numpy(filename, "temp")

        dens_blocks = np.squeeze(split_to_blocks(dens))
        pres_blocks = np.squeeze(split_to_blocks(pres))
        temp_blocks = np.squeeze(split_to_blocks(temp))

        # -> (512, NX, NY, NZ, 3)
        #blocks = np.stack((dens_blocks, pres_blocks, temp_blocks), -1)
        blocks = np.expand_dims(dens_blocks, -1)
        print filename, blocks.shape
        for i in range(blocks.shape[0]):
            tmp = filename.split("/")[-1]
            output_filename = output_dir + "/" + tmp + "." + str(i) +".clean.npy"
            np.save(output_filename, blocks[i])
        count += 1


# Read from clean numpy data files and ouput error data files
def create_split_error_dataset(data_dir, output_dir):
    file_list = glob.glob(data_dir+"*.npy")
    file_list.sort()
    for filename in file_list:
        print filename
        data = np.load(filename)

        # Insert an error
        #x, y, z, var = random.randint(1, data.shape[0])-1, random.randint(1, data.shape[1])-1, \
        #                random.randint(1, data.shape[2])-1, random.randint(0, data.shape[3]-1)
        #data[x, y, z, var] = get_flip_error(data[x, y, z, var], 20)
        x, y, var = random.randint(1, data.shape[0])-1, random.randint(1, data.shape[1])-1, random.randint(0, data.shape[2]-1)
        data[x, y, 0] = get_flip_error(data[x, y, 0], 15)
        tmp = filename.split("/")[-1]
        tmp = tmp.replace("clean", "error")
        output_filename = output_dir + "/" + tmp
        np.save(output_filename, data)   # Save to binary format

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    create_split_clean_dataset(input_dir, output_dir)
    #create_split_error_dataset(input_dir, output_dir)
    #create_split_error_testset(sys.argv[1])

