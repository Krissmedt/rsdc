import numpy as np
from tools.writing import wp_dump
from tools.plotting import *
from pushers.coll import coll
from pushers.coll_solve import implicit_coll
from pushers.gauss_legendre import CollGaussLegendre
from pushers.gauss_lobatto import CollGaussLobatto

from uniform_e import config

def plot_residuals(t,xres,vres,conf):
    fig_xres = plt.figure(8)
    ax_xres = fig_xres.add_subplot(111)
    for m in range(0,rx_array.shape[1]):
        ax_xres.plot(t[1:],rx_array[:,m],label="m={0}".format(m))
    ax_xres.set_xlim([0,t[-1]])
    ax_xres.set_yscale('log')
    ax_xres.legend()

    fig_vres = plt.figure(9)
    ax_vres = fig_vres.add_subplot(111)
    for m in range(0,rv_array.shape[1]):
        ax_vres.plot(t[1:],rv_array[:,m],label="m={0}".format(m))
    ax_vres.set_xlim([0,t[-1]])
    ax_vres.set_yscale('log')
    ax_vres.legend()

    filename = "{0}coll_M{1}_{2}".format(conf.data_root,m+1,conf.name)
    fig_vres.savefig(filename+"_vres.pdf")
    fig_xres.savefig(filename+"_xres.pdf")


sims = [640]
tend = 10

M = 5

conf = config()

plot = False
new = True

for Nt in sims:
    dt = tend/Nt
    t = 0
    pos, vel = conf.prtcl_setup()

    rx_array = []
    rv_array = []
    x_array = [pos]
    v_array = [vel]
    t_array = [t]

    col = coll(CollGaussLobatto,dt,conf.nq,M=M,K=1)
    for ti in range(1,Nt+1):
        t = ti*dt

        pos, vel, col = implicit_coll(pos,vel,col,conf,t=t)

        rx_array.append(col.Rx[1,:])
        rv_array.append(col.Rv[1,:])
        x_array.append(np.copy(pos))
        v_array.append(np.copy(vel))
        t_array.append(t)

    rx_array = np.array(rx_array)
    rv_array = np.array(rv_array)
    x_array = np.array(x_array)
    v_array = np.array(v_array)
    t_array = np.array(t_array)

    filename = "{0}coll_M{1}_{2}".format(conf.data_root,M,conf.name)
    wp_dump(t_array,x_array,v_array,dt,filename+"_wp.h5",new=new)
    new = False

    if plot == True:
        plot_isotraj(x_array,filename+"_isotraj")
        plot_vel(t_array,v_array,filename+"_traj")
        plot_residuals(t_array,rx_array,rv_array,conf)
