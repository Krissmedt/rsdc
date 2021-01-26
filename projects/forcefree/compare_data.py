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
Nt = 10

filenames = {}

filenames["Velocity-Verlet"] = "vvb_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)
filenames["Vay"] = "vay_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)
filenames["Collocation M3"] = "coll_M3_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)
filenames["Collocation M5"] = "coll_M5_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)
filenames["Boris-SDC M3K1"] = "sdc_M3K1_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)
filenames["Boris-SDC M3K2"] = "sdc_M3K2_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)
filenames["Boris-SDC M5K1"] = "sdc_M5K1_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)
filenames["Boris-SDC M5K4"] = "sdc_M5K4_{0}_te{1}_nt{2}.h5".format(testcase,tend,Nt)


plot_params = {}
plot_params['legend.fontsize'] = 16
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 20
plot_params['axes.titlesize'] = 20
plot_params['xtick.labelsize'] = 16
plot_params['ytick.labelsize'] = 16
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 5
plot_params['legend.loc'] = 'upper right'
plt.rcParams.update(plot_params)

scale = 'log'
r = 1
b = 1
gr = 1
am3 = 1
am4 = 1
am5 = 1

for key,value in filenames.items():
    file = h5.File(data_root+value,'r')
    dt = file["fields/dt"]
    t = file["fields/t"][:]
    x = file["fields/pos"][:]
    u = file["fields/vel"][:]

    v = np.zeros(u.shape,dtype=np.float)
    g = np.zeros((u.shape[0],u[:].shape[1]),dtype=np.float)
    g0 = conf.gamma

    for ts in range(0,u.shape[0]):
        g[ts,:] = gu(u[ts,:,:])
        v[ts,:,:] = g[ts,:]*u[ts,:,:]
    print(key)
    print(v[-1,0,0])
    refg_errors = np.abs(g-g0)/np.abs(g0)
    refg_errors = np.linalg.norm(refg_errors,axis=1)

    if "Velocity-Verlet" in key:
        c = "black"
    if key == "Leapfrog":
        c = "black"
    if key == "Vay":
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
        sims = 2
        c = (r,0,0,am3)
        am3 -= 1/sims
    if "Boris-SDC M4" in key:
        sims = 2
        c = (r,0,b,am4)
        am4 -= 1/sims
    if "Boris-SDC M5" in key:
        sims = 1
        c = (0,gr,0,am5)
        am5 -= 1/sims

    label = key
    ## Error in pos vs. time
    fig_pos = plt.figure(1)
    ax_pos = fig_pos.add_subplot(1, 1, 1)
    ax_pos.plot(t,np.abs(x[:,0,0]),color=c,label=label)
    ax_pos.set_ylabel(r'$x$')
    ax_pos.set_yscale(scale)
    # ax_pos.set_ylim(0,10**-19)
    ax_pos.set_xlabel(r'$t$')
    ax_pos.set_xlim(0,10**tend)

    ## Error in vel vs. time
    fig_vel = plt.figure(2)
    ax_vel = fig_vel.add_subplot(1, 1, 1)
    ax_vel.plot(t,np.abs(v[:,0,0]),color=c,label=label)
    ax_vel.set_ylabel(r'$v_x$')
    ax_vel.set_yscale(scale)
    # ax_vel.set_ylim(0,10**-8)
    ax_vel.set_xlabel(r'$t$')
    ax_vel.set_xlim(0,10**tend)

    # ## Error in gamma vs. time
    # fig_gam = plt.figure(3)
    # ax_gam = fig_gam.add_subplot(1, 1, 1)
    # ax_gam.plot(t,g,color=c,label=label)
    # ax_gam.set_ylabel(r'\gamma')
    # ax_gam.set_yscale(scale)
    # ax_gam.set_ylim(10**(5),10**7)
    # ax_gam.set_xlabel(r'$t$')
    # ax_gam.set_xlim(0,tend)

handles, labels = fig_pos.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_pos.legend(by_label.values(), by_label.keys(),loc='lower left')

handles, labels = fig_vel.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_vel.legend(by_label.values(), by_label.keys(),loc='lower left')

# handles, labels = fig_gam.gca().get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# ax_gam.legend(by_label.values(), by_label.keys(),loc='lower left')

fig_pos.savefig(data_root + conf.name + '_pos_te{0}_nt{1}'.format(tend,Nt) + fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
fig_vel.savefig(data_root + conf.name + '_vel_te{0}_nt{1}'.format(tend,Nt) + fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
# fig_gam.savefig(data_root + conf.name + '_gam_te{0}_nt{1}'.format(tend,Nt) + fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
