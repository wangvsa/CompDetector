#!/usr/bin/env python
# encoding: utf-8

from mpi4py import MPI
import numpy as np
import glob, sys
from detector import detection, model
from create_dataset import hdf5_to_numpy, split_to_blocks, read_data
from timeit import default_timer as timer

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

print(rank, name, size)
val = rank

sendbuf = np.zeros(1)
recvbuf = np.zeros(1)

# Construct the model
try:
    model = multi_gpu_model(model)
except:
    pass
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights(sys.argv[2])

error = 0
path = sys.argv[1]
files = glob.glob(path+"/*chk_*")+glob.glob(path+"/*error*")
files.sort()

total_time = 0
start = timer()
for filename in files:
    dens = read_data(filename)
    t1 = timer()
    dens_blocks = np.expand_dims(np.squeeze(split_to_blocks(dens)), -1)

    #print rank, filename
    N = dens_blocks.shape[0] / size
    start_idx, end_idx = N*rank, N*(rank+1)
    hasError = detection(model, dens_blocks[start_idx:end_idx])
    sendbuf[0] = hasError # hasError is Ture/Flase

    comm.Reduce(sendbuf, recvbuf, op=MPI.LOR, root=0)
    if rank == 0:
        error += recvbuf[0]
        #print filename, recvbuf
    t2 = timer()
    total_time += (t2-t1)

end = timer()

if rank == 0:
    print "Running time: %s seconds, training time: %s" %(end-start, total_time)
    print "detected %s error samples, total: %s, recall: %s" %(error, len(files), error*1.0/len(files))
