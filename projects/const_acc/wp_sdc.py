import numpy as np
from tools.writing import wp_dump
from tools.plotting import *
from pushers.boris_sdc import boris_SDC
from pushers.coll import coll
# from pushers.rel_col18 import implicit_coll
from pushers.gauss_legendre import CollGaussLegendre
from pushers.gauss_lobatto import CollGaussLobatto

from uniform_e import config


sims = [10,20,40,80,160,320,640]
# sims = [1]
tend = 10

M = 5
K_range = [4]

conf = config()

plot = False
new = True

for K in K_range:
    new = True
    for Nt in sims:
        dt = tend/Nt
        t = 0
        pos, vel = conf.prtcl_setup()

        x_array = [pos]
        v_array = [vel]
        t_array = [t]

        col = coll(CollGaussLobatto,dt,conf.nq,K=K,M=M,predictor=True)
        rx_array = [np.linalg.norm(col.Rx,axis=1)]
        rv_array = [np.linalg.norm(col.Rv,axis=1)]

        for ti in range(1,Nt+1):
            t = ti*dt

            pos, vel, col = boris_SDC(pos,vel,col,conf)

            rx_array.append(np.linalg.norm(col.Rx,axis=1))
            rv_array.append(np.linalg.norm(col.Rv,axis=1))
            x_array.append(pos)
            v_array.append(vel)
            t_array.append(t)

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
        wp_dump(t_array,x_array,v_array,dt,filename+"_wp.h5",rhs=rhs,new=new)
        new = False

        if plot == True:
            plot_xres(t_array,rx_array,filename+"_xres")
            plot_vres(t_array,rv_array,filename+"_vres")
            plot_isotraj(x_array,filename+"_isotraj")
            plot_vel(t_array,v_array,filename+"_traj")
