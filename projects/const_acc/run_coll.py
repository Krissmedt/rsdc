import numpy as np
import math
import time
from tools.writing import data_dump
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
        ax_xres.plot(t,rx_array[:,m],label="m={0}".format(m))
    ax_xres.set_xlim([0,t[-1]])
    ax_xres.set_yscale('log')
    ax_xres.legend()

    fig_vres = plt.figure(9)
    ax_vres = fig_vres.add_subplot(111)
    for m in range(0,rv_array.shape[1]):
        ax_vres.plot(t,rv_array[:,m],label="m={0}".format(m))
    ax_vres.set_xlim([0,t[-1]])
    ax_vres.set_yscale('log')
    ax_vres.legend()

    filename = "{0}coll_M{1}_{2}".format(conf.data_root,m+1,conf.name)
    fig_vres.savefig(filename+"_vres.pdf")
    fig_xres.savefig(filename+"_xres.pdf")


dt = 10.**3
tend = 2.*10.**8
tend = 10**5
Nt = np.int(tend/dt)
samples = 100

plot = False

M = 5

conf = config()
sample_interval = math.floor(Nt/samples)

t = 0
pos, vel = conf.prtcl_setup()

x_array = [pos]
v_array = [vel]
t_array = [t]

col = coll(CollGaussLobatto,dt,conf.nq,K=1,M=M)
rx_array = [np.linalg.norm(col.Rx,axis=1)]
rv_array = [np.linalg.norm(col.Rv,axis=1)]

sim_time = 0
t1 = time.time()
for ti in range(1,Nt+1):
    t = ti*dt
    pos, vel, col = implicit_coll(pos,vel,col,conf,t=t)

    if ti % sample_interval == 0:
        t2 = time.time()
        sim_time += t2-t1
        print("Status: " + time.strftime("%d/%m/%y  %H:%M:%S",time.localtime()) +
        ", step = {0}, t = {1}, time elapsed = {2}s".format(ti,t,np.around(sim_time,decimals=2)))
        t1 = time.time()
        rx_array.append(np.linalg.norm(col.Rx,axis=1))
        rv_array.append(np.linalg.norm(col.Rv,axis=1))
        x_array.append(np.copy(pos))
        v_array.append(np.copy(vel))
        t_array.append(np.copy(t))

rx_array = np.array(rx_array)
rv_array = np.array(rv_array)
x_array = np.array(x_array)
v_array = np.array(v_array)
t_array = np.array(t_array)

filename = "{0}coll_M{1}_{2}".format(conf.data_root,M,conf.name)
data_dump(t_array,x_array,v_array,dt,
          filename+"_full.h5",xres=rx_array,vres=rv_array)

if plot == True:
    plot_residuals(t_array,rx_array,rv_array,conf)
    plot_isotraj(x_array,filename+"_isotraj")
    plot_vel(t_array,v_array,filename+"_traj")
