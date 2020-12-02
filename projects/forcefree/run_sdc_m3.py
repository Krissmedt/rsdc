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

from forcefree import config
conf = config()

tend = 10**5
dt = 0.25
Nt = np.int(tend/dt)
samples = 1000

plot = False

M = 3
K_range = [2]

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

    tend_str = np.int(np.log10(tend))
    nt_str = np.int(Nt/tend)
    filename = "{0}sdc_M{1}K{2}_{3}_te{4}_nt{5}".format(conf.data_root,M,K,conf.name,tend_str,nt_str)
    data_dump(t_array,x_array,v_array,dt,
              filename+".h5",xres=rx_array,vres=rv_array,rhs=rhs)

    if plot == True:
        plot_xres(t_array,rx_array,filename)
        plot_vres(t_array,rv_array,filename)
        plot_traj(x_array,filename)
