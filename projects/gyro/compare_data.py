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

from gyro import config
conf = config()

data_root = conf.data_root
fig_name = "full"
testcase = "gyro"

filenames = {}

# filenames["Leapfrog"] = "lf_{0}_full.h5".format(testcase)
filenames["Velocity-Verlet"] = "vvb_{0}_full.h5".format(testcase)
filenames["Vay"] = "vay_{0}_full.h5".format(testcase)
# filenames["HC"] = "hc_{0}_full.h5".format(testcase)
filenames["Collocation M5"] = "coll_M5_{0}_full.h5".format(testcase)
filenames["Boris-SDC M3K2"] = "sdc_M3K2_{0}_full.h5".format(testcase)
filenames["Boris-SDC M3K4"] = "sdc_M3K4_{0}_full.h5".format(testcase)
filenames["Boris-SDC M5K4"] = "sdc_M5K4_{0}_full.h5".format(testcase)
filenames["Boris-SDC M5K5"] = "sdc_M5K5_{0}_full.h5".format(testcase)
filenames["Boris-SDC M5K6"] = "sdc_M5K6_{0}_full.h5".format(testcase)

plot_params = {}
plot_params['legend.fontsize'] = 22
plot_params['figure.figsize'] = (12,8)
plot_params['axes.labelsize'] = 24
plot_params['axes.titlesize'] = 24
plot_params['xtick.labelsize'] = 24
plot_params['ytick.labelsize'] = 24
plot_params['lines.linewidth'] = 4
plot_params['axes.titlepad'] = 10
plot_params['legend.loc'] = 'upper right'
plt.rcParams.update(plot_params)
r = 1
b = 1
gr = 1
am3 = 1
am5 = 1

for key,value in filenames.items():
    file = h5.File(data_root+value,'r')
    Nt = file["fields/Nt"]
    dt = file["fields/dt"]
    t = file["fields/t"][:]
    x = file["fields/pos"][:]
    u = file["fields/vel"][:]

    v = np.zeros(u.shape,dtype=np.float)
    g = np.zeros((u.shape[0],u[:].shape[1]),dtype=np.float)
    x_sol = np.zeros(u.shape,dtype=np.float)
    u_sol = np.zeros(u.shape,dtype=np.float)
    gamma_sol = conf.gamma
    x0, u0 = conf.prtcl_setup()
    u_sol = u0[:,1]
    Tc = conf.ref_period()

    for ts in range(0,u.shape[0]):
        g[ts,:] = gu(u[ts,:,:])
        x_sol[ts,0,:] = conf.ref_pos(t[ts])

    ref_errors = np.abs(x[:,0,:]-x_sol[:,0,:])
    ref_errors = np.linalg.norm(ref_errors,axis=1)

    refu_errors = np.abs(np.linalg.norm(u,axis=2)+u_sol)/np.abs(u_sol)
    refu_errors = np.linalg.norm(refu_errors,axis=1)

    refg_errors = np.abs(g-gamma_sol)/np.abs(gamma_sol)
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
        sims = 3
        c = (0,gr,0,am5)
        am5 -= 1/sims

    label = key
    ## Error in pos vs. time
    fig_pos = plt.figure(1)
    ax_pos = fig_pos.add_subplot(1, 1, 1)
    ax_pos.plot((t/Tc),ref_errors,color=c,label=label)
    ax_pos.set_ylabel(r'$\Delta x^{\mathrm{rel}}$')
    ax_pos.set_yscale('log')
    # ax_pos.set_ylim(10**(-15),10**(0))
    ax_pos.set_xlabel(r'$t/T_c$')
    ax_pos.set_xlim(0,100)

    ## Error in vel vs. time
    fig_vel = plt.figure(2)
    ax_vel = fig_vel.add_subplot(1, 1, 1)
    ax_vel.plot(t/Tc,refu_errors,color=c,label=label)
    ax_vel.set_ylabel(r'$\Delta v^{\mathrm{rel}}$')
    ax_vel.set_yscale('log')
    ax_vel.set_ylim(10**(-16),10**(-7))
    ax_vel.set_xlabel(r'$t/T_c$')
    ax_vel.set_xlim(0,100)

    ## Error in gamma vs. time
    fig_gam = plt.figure(3)
    ax_gam = fig_gam.add_subplot(1, 1, 1)
    ax_gam.plot(t/Tc,refg_errors,color=c,label=label)
    ax_gam.set_ylabel(r'$\Delta \gamma^{\mathrm{rel}}$')
    ax_gam.set_yscale('log')
    # ax_gam.set_ylim(10**(-16),10**(-10))
    ax_gam.set_xlabel(r'$t/T_c$')
    ax_gam.set_xlim(0,100)

handles, labels = fig_pos.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_pos.legend(by_label.values(), by_label.keys(),loc='lower left')

handles, labels = fig_vel.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_vel.legend(by_label.values(), by_label.keys(),loc='lower left')

handles, labels = fig_gam.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax_gam.legend(by_label.values(), by_label.keys(),loc='lower left')

fig_pos.savefig(data_root + conf.name + '_pos_'+ fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
fig_vel.savefig(data_root + conf.name + '_vel_' + fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
fig_gam.savefig(data_root + conf.name + '_gam_' + fig_name + '.pdf', dpi=150, facecolor='w', edgecolor='w',orientation='portrait',pad_inches=0.05,bbox_inches = 'tight')
