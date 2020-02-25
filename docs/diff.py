#!/usr/bin/env python
# encoding: utf-8
import os, glob
import h5py



if __name__ == "__main__":
    error_dir = "../error/"
    clean_dir = "../clean/"
    diff_dir = "./"


    for error_filename in glob.glob(error_dir+"*plt_cnt*"):
        os.system("cp "+ error_filename + " " + diff_dir)

        basename = error_filename.split("/")[-1]
        clean_filename = clean_dir + basename
        diff_filename = diff_dir + basename

        clean_file = h5py.File(clean_filename, 'r')
        diff_file = h5py.File(diff_filename, 'r+')

        diff_file['dens'][0,0] = diff_file['dens'][0,0] - clean_file['dens'][0,0]
        diff_file['pres'][0,0] = diff_file['pres'][0,0] - clean_file['pres'][0,0]
        diff_file['temp'][0,0] = diff_file['temp'][0,0] - clean_file['temp'][0,0]
        diff_file['velx'][0,0] = diff_file['velx'][0,0] - clean_file['velx'][0,0]

        clean_file.close()
        diff_file.close()

