import numpy as np
import math
import time
from tools.writing import data_dump
from tools.plotting import *
from pushers.boris_sdc import boris_SDC
from pushers.coll import coll
# from pushers.rel_col18 import implicit_coll
from pushers.gauss_legendre import CollGaussLegendre
from pushers.gauss_lobatto import CollGaussLobatto

from uniform_e import config



dt = 10.**3
tend = 10**9
Nt = np.int(tend/dt)
samples = 10000

plot = False
offset = True

M = 5
K_range = [5]

conf = config()
sample_interval = math.floor(Nt/samples)

for K in K_range:
    si = 1
    t = 0
    pos, vel = conf.prtcl_setup()

    x_array = [pos]
    v_array = [vel]
    t_array = [t]

    col = coll(CollGaussLobatto,dt,conf.nq,K=K,M=M,predictor=True)
    rx_array = [np.linalg.norm(col.Rx,axis=1)]
    rv_array = [np.linalg.norm(col.Rv,axis=1)]

    sim_time = 0
    t1 = time.time()

    if offset == True:
        si = 2
        t += dt
        pos, vel, col = boris_SDC(pos,vel,col,conf)
        pos[:,0] = conf.ref_x(t)

        x_array.append(np.copy(pos))
        v_array.append(np.copy(vel))
        t_array.append(np.copy(t))
        rx_array.append(np.linalg.norm(col.Rx,axis=1))
        rv_array.append(np.linalg.norm(col.Rv,axis=1))

    for ti in range(si,Nt+1):
        t = ti*dt
        pos, vel, col = boris_SDC(pos,vel,col,conf)

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

    if col.predictor == True:
        rhs = (M-1)*(K+1)*Nt
    else:
        rhs = (M-1)*K*Nt

    filename = "{0}sdc_M{1}K{2}_{3}".format(conf.data_root,M,K,conf.name)
    data_dump(t_array,x_array,v_array,dt,
              filename+"_full.h5",xres=rx_array,vres=rv_array,rhs=rhs)

    if plot == True:
        plot_xres(t_array,rx_array,filename+"_xres")
        plot_vres(t_array,rv_array,filename+"_vres")
        plot_isotraj(x_array,filename+"_isotraj")
        plot_vel(t_array,v_array,filename+"_traj")
