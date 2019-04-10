import sys, os
import numpy as np
import random
import glob
NX, NY = 480, 480

def decompress(filename, tool="sz"):
    if tool == "sz":
        os.system("./sz -x -d -s " + filename + " -2 480 480")
    elif tool == "zip":
        os.system("mv "+filename+" "+filename+".zip")
        os.system("unzip " + filename + ".zip")
    elif tool == "gzip":
        os.system("mv "+filename+" "+filename+".gz")
        os.system("gzip -d " + filename+".gz")
    elif tool == "zfp":
        os.system("./zfp -d -2 480 480 -a 0.1 -z "+filename+" -o "+filename+".out")
    else:
        print "not supported"


def read_binary_by_bytes(filename):
    # Read the sz compressed binary file byte by byte
    # this gives us an array of byte (uint8)
    # e.g. [120, 2, 30, ...]
    data = np.fromfile(filename, dtype = "uint8")

    # Convert data to a bitstring, e.g. "01010111000..."
    # len(bits) = 8 * len(data)
    bits = np.unpackbits(data)

    # Now we can flip a bit and pack it back
    pos = random.randint(1, len(bits)) - 1
    bits[pos] = 0 if bits[pos] == 1 else 1

    data2 = np.packbits(bits)
    print pos, data[pos/8], data2[pos/8]

    # At last, we write it back to the binary file
    corrupted_file = filename + ".corrupted"
    data2.tofile(corrupted_file)

    # Decompress it
    decompress(corrupted_file, "sz")

if __name__ == "__main__":
    for filename in glob.iglob(sys.argv[1]+"/*.sz"):
        print filename
        read_binary_by_bytes(filename)
