#!/usr/bin/env python
# encoding: utf-8

from mpi4py import MPI
import numpy as np
import glob
from keras_detector_3d import detection, model
from create_dataset import hdf5_to_numpy, split_to_blocks
from timeit import default_timer as timer

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print rank, size
val = rank

sendbuf = np.zeros(1)
recvbuf = np.zeros(1)

# Construct the model
try:
    model = multi_gpu_model(model)
except:
    pass
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("./models/model_keras_sod_15bits.h5")

error = 0
path = "/home/chenw/Flash/Sod_3d/0/10_delay/"
files = glob.glob(path+"/*chk_*")+glob.glob(path+"/*error*")
files.sort()

start = timer()
for filename in files:
    dens = hdf5_to_numpy(filename)
    dens_blocks = np.expand_dims(np.squeeze(split_to_blocks(dens)), -1)

    #print rank, filename
    N = dens_blocks.shape[0] / size
    start_idx, end_idx = N*rank, N*(rank+1)
    hasError = detection(model, dens_blocks[start_idx:end_idx])
    sendbuf[0] = hasError # hasError is Ture/Flase

    comm.Reduce(sendbuf, recvbuf, op=MPI.LOR, root=0)
    if rank == 0:
        error += recvbuf[0]
        print filename, recvbuf

end = timer()

if rank == 0:
    print "Running time: %s seconds." %(end-start)
    print "detected %s error samples, total: %s, recall: %s" %(error, len(files), error*1.0/len(files))
