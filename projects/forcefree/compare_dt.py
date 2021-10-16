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
from pushers.gammafac import gu

from forcefree import config
conf = config()

data_root = conf.data_root
fig_name = ""
testcase = "forcefree"
tend = 5
Nt = [1,2,4,8,10,16,32,100,1000]

scale = 'log'
r = 1
b = 1
gr = 1
am3 = 1
am4 = 1
am5 = 1

filenames = {}

filenames["Velocity-Verlet B"] = "vvb_{0}_te{1}".format(testcase,tend)
filenames["Vay"] = "vay_{0}_te{1}".format(testcase,tend)
filenames["Boris-SDC M3K1"] = "sdc_M3K1_{0}_te{1}".format(testcase,tend)
# filenames["Boris-SDC M3K2"] = "sdc_M3K2_{0}_te{1}".format(testcase,tend)
filenames["Boris-SDC M5K1"] = "sdc_M5K1_{0}_te{1}".format(testcase,tend)
# filenames["Boris-SDC M5K4"] = "sdc_M5K4_{0}_te{1}".format(testcase,tend)


plot_params = {}
plot_params['legend.fontsize'] = 22
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 24
plot_params['axes.titlesize'] = 24
plot_params['xtick.labelsize'] = 24
plot_params['ytick.labelsize'] = 24
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 5
plot_params['legend.loc'] = 'upper left'
plt.rcParams.update(plot_params)


for key,value in filenames.items():
    dt = []
    t = []
    x = []
    vx = []
    for nt in Nt:
        filename = value + "_nt{0}.h5".format(nt)
        file = h5.File(data_root+filename,'r')
        dt.append(10**tend/(10**tend * nt))
        t.append(file["fields/t"][-1])
        x.append(file["fields/pos"][-1,0,0])
        
        u = file["fields/vel"][-1,:,:]
        v = gu(u,c=conf.c)*u
        vx.append(v[0,0])

    x = np.abs(x)
    vx = np.abs(vx)
    print(key)
    print(vx)
    if "Velocity-Verlet B" in key:
        ms = 14
        lw = 7
        marker = "x"
        c = "black"
    if key == "Leapfrog":
        c = "black"
    if key == "Vay":
        marker = "v"
        ms = 12
        lw = 6
        c = "blue"
    if key == "HC":
        c = (r,0,b)
    if key == "Collocation M3":
        c = '0.4'
    if key == "Collocation M4":
        c = '0.6'
    if key == "Collocation M5":
        c = '0.8'
    if "Boris-SDC M3" in key:
        ms = 10
        lw = 5
        sims = 2
        marker = "s"
        c = (r,0,0,am3)
        am3 -= 1/sims
    if "Boris-SDC M4" in key:
        sims = 6
        c = (r,0,b,am4)
        am4 -= 1/sims
    if "Boris-SDC M5" in key:
        ms= 8
        lw = 4
        marker = "o"
        sims = 2
        c = (0,gr,0,am5)
        am5 -= 1/sims
        


    label = key
    ## Error in pos vs. time
    fig_pos = plt.figure(1)
    ax_pos = fig_pos.add_subplot(1, 1, 1)
    ax_pos.plot(dt,x,color=c,label=label,linewidth=lw,marker=marker,markersize=ms)
    ax_pos.set_ylabel(r'$x$')
    ax_pos.set_yscale(scale)
    ax_pos.set_xscale(scale)
    # ax_pos.set_ylim(0,10**-19)
    ax_pos.set_xlabel(r'$\Delta t$')
    # ax_pos.set_xlim(1,10**tend)

    ## Error in vel vs. time
    fig_vel = plt.figure(2)
    ax_vel = fig_vel.add_subplot(1, 1, 1)
    ax_vel.plot(dt,vx,color=c,label=label,linewidth=lw,marker=marker,markersize=ms)
    ax_vel.set_ylabel(r'$v_x$')
    ax_vel.set_yscale(scale)
    ax_vel.set_xscale(scale)
    # ax_vel.set_ylim(0,10**-8)
    ax_vel.set_xlabel(r'$\Delta t$')
    # ax_vel.set_xlim(0,10**tend)



handles, labels = fig_pos.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_pos.legend(by_label.values(), by_label.keys(),loc='upper left')

handles, labels = fig_vel.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_vel.legend(by_label.values(), by_label.keys(),loc='upper left')

fig_pos.savefig(data_root + conf.name + '_pos_te{0}_dt'.format(tend) + fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
fig_vel.savefig(data_root + conf.name + '_vel_te{0}_dt'.format(tend) + fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
