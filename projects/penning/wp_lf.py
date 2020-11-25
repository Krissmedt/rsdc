import numpy as np
from tools.writing import wp_dump
from tools.plotting import *
from pushers.leapfrog import leapfrog
from pushers.leapfrog import sync_leapfrog

from penning import config


sims = [10,20,40,80,160,320,640,1280,2560]
# sims = [1]
tend = 10

conf = config()

plot = False
new = True

for Nt in sims:
    dt = tend/Nt
    t = 0
    pos0, vel0 = conf.prtcl_setup()
    pos, vel = sync_leapfrog(pos0,vel0,dt,conf,t=t) #match initial conditions

    x_array = [pos0,pos]
    v_array = [vel0,vel]
    t_array = [t,t+dt]

    for ti in range(2,Nt+1):
        t = ti*dt

        pos, vel = leapfrog(pos,vel,dt,conf,t=t)

        x_array.append(np.copy(pos))
        v_array.append(np.copy(vel))
        t_array.append(t)

    x_array = np.array(x_array)
    v_array = np.array(v_array)
    t_array = np.array(t_array)

    filename = "{0}lf_{1}".format(conf.data_root,conf.name)
    wp_dump(t_array,x_array,v_array,dt,filename+"_wp.h5",rhs=Nt,new=new)
    new = False

    if plot == True:
        plot_isotraj(x_array,filename+"_isotraj")
        plot_vel(t_array,v_array,filename+"_traj")
