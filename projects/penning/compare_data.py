from math import sqrt, fsum, pi, exp, cos, sin, floor
from decimal import Decimal
import io
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import h5py as h5
import sys
import traceback
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

from tools.writing import *
from tools.plotting import *

data_root = "./output/"
fig_name = "640"

filenames = {}


filenames["A"] = "sdc_M5K4_A_640.h5"
filenames["B"] = "sdc_M5K4_B_640.h5"
filenames["C"] = "sdc_M5K4_C_640.h5"
filenames["D"] = "sdc_M5K4_D_640.h5"

plot_params = {}
plot_params['legend.fontsize'] = 22
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 24
plot_params['axes.titlesize'] = 24
plot_params['xtick.labelsize'] = 24
plot_params['ytick.labelsize'] = 24
plot_params['lines.linewidth'] = 3
plot_params['axes.titlepad'] = 10
plot_params['legend.loc'] = 'upper right'
plt.rcParams.update(plot_params)
r = 1
b = 1

for key,value in filenames.items():
    file = h5.File(data_root+value,'r')
    dt = file["fields/dt"]
    times = file["fields/t"]
    x = file["fields/pos"]
    v = file["fields/vel"]

    if key == "Velocity-Verlet":
        c = "black"
    if key == "Leapfrog":
        c = "green"
    if key == "Vay":
        c = "orange"
    if key == "HC":
        c = "pink"
    if "Boris-SDC M3" in key:
        sims = 2
        c = (0,0,b)
        b -= 1/sims
    if "Boris-SDC M5" in key:
        sims = 7
        c = (r,0,b)
        r -= np.round(0.7/sims,2)

    label = key
    ## Iso trajectory plot
    plot_params['xtick.labelsize'] = 20
    plot_params['ytick.labelsize'] = 20
    plt.rcParams.update(plot_params)

    fig_isotraj = plt.figure(1)
    ax = fig_isotraj.gca(projection='3d')
    ax.plot3D(x[:,0,0],
              x[:,0,1],
              zs=x[:,0,2],
              label=label)

    ax.view_init(35, 0)
    ax.set_xlabel('$x$')
    ax.set_ylabel('\n $y$')
    ax.set_zlabel('\n $z$')
    ax.legend(loc="upper left")

    ## XY trajectory plot
    plot_params['xtick.labelsize'] = 24
    plot_params['ytick.labelsize'] = 24
    plt.rcParams.update(plot_params)

    fig_xytraj = plt.figure(2)
    ax_xy = fig_xytraj.add_subplot(111)
    ax_xy.plot(x[:,0,0],x[:,0,1],label=label)
    ax_xy.set_xlabel('$x$')
    ax_xy.set_ylabel('$y$')
    ax_xy.legend(loc="lower left")

fig_isotraj.savefig(data_root + 'pen_x_'+ fig_name + '_isotraj.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
fig_xytraj.savefig(data_root + 'pen_x_'+ fig_name + '_xytraj.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
