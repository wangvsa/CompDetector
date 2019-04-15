#!/usr/bin/env python2
# encoding: utf-8
import numpy as np
import h5py
import os, sys
import random
from create_dataset import get_flip_error, hdf5_to_numpy, read_data
from skimage.util.shape import view_as_windows, view_as_blocks

NX, NY, NZ = 16, 16, 16

def get_random_indices():
    i = random.randint(4, 60)
    j = random.randint(4, 60)
    k = random.randint(4, 60)
    return i, j, k

def restart(checkpoint_file, postfix=0, var_name="dens"):
    timestep = int(checkpoint_file[-4:])
    start_checkpoint_file = checkpoint_file

    os.system("cp ./clean/"+checkpoint_file + " ./"+start_checkpoint_file)
    end_checkpoint_file = start_checkpoint_file[0:-4] + ("0000"+str(timestep+1))[-4:]

    print(start_checkpoint_file, end_checkpoint_file)

    # 1. insert an error into the checkpoint file
    # remeber the error bit and error position (x,y,z)
    f = h5py.File(start_checkpoint_file, "r+")
    x, y, z = get_random_indices()
    error, bit = get_flip_error(f[var_name][0, z, y, x], 0, 30)
    print(bit, x, y, z, f[var_name][0,z,y,x], error)
    f[var_name][0,z,y,x] = error
    f.close()

    # 2. read flash.par and change it so we could restart from the corrupted checkpoint
    # need to specify restart from which iteration
    # and also the end iteration
    # sed can do this easily
    os.system("cp ./flash.par ./flash.par_"+postfix)
    os.system("sed -i \'s/.*checkpointFileNumber.*/checkpointFileNumber = " +str(timestep) + "/g\' ./flash.par_"+postfix)
    #os.system("sed -i \'s/.*basenm.*/basenm = " + postfix+"_sod_" + "/g\' ./flash.par_"+postfix)
    #os.system("sed -i \'s/.*nend.*/nend = " +str(timestep+delay) + "/g\' ./flash.par_"+postfix)

    # 3. restart the flash program
    os.system("mpirun -np 8 ./flash4 -par_file ./flash.par_"+postfix)

    # 4. read the corrupted checkpoint file
    new_start_name = "error_%s_%s_%s_%s_%s" %(bit,x,y,z,timestep)
    new_end_name = "error_%s_%s_%s_%s_200" %(bit,x,y,z)
    np.save(new_start_name+".npy", read_data(start_checkpoint_file, var_name))
    np.save(new_end_name+".npy", read_data(end_checkpoint_file, var_name))
    os.system("rm " + start_checkpoint_file)
    os.system("rm " + end_checkpoint_file)

    # 5. delete unnecessary files
    os.system("rm *.dat *.log *_plt_cnt_*")


basenm = sys.argv[1]
start_iter = int(sys.argv[2])
end_iter = int(sys.argv[3])
for t in range(start_iter, end_iter):
    checkpoint_file = basenm + ("0000"+str(t))[-4:]
    for repeat in range(2):
        #try:
        postfix = (str(start_iter)+"to"+str(end_iter))
        print postfix
        restart(checkpoint_file, postfix=(str(start_iter)+"to"+str(end_iter)))
        #except:
        #    print("app crashs!!!")
