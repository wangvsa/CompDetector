#!/usr/bin/env python
# encoding: utf-8

from bokeh.plotting import figure, output_file, show
import sys, glob, h5py
import numpy as np

if __name__ == "__main__":
    diff_dir = sys.argv[1]
    var = sys.argv[2]

    filenames = sorted(glob.glob(diff_dir+"/*plt_cnt*"))

    y = []
    for filename in filenames:
        f = h5py.File(filename)
        y.append( np.abs(np.max(f[var][0,0])) )

    title = {
        'dens': 'Max Error of Density',
        'pres': 'Max Error of Pressure',
        'temp': 'Max Error of Temperature',
        'velx': 'Max Error of Velocity X',
    }


    output_file("line.html")
    p = figure(y_axis_label="abs(error)", x_axis_label="Plot File Number", title=title[var], plot_width=400, plot_height=300)

    p.line(range(len(filenames)), y, line_width=2)


    show(p)
