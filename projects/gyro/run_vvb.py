import numpy as np
import math
import time
from tools.writing import data_dump
from tools.plotting import *
from pushers.velocity_verlet import velocity_verlet_B as vvb

from gyro import config
conf = config()

tend = 100*conf.ref_period()
dt = conf.ref_period()/100
Nt = np.int(tend/dt)
samples = 1000

plot = True

sample_interval = math.floor(Nt/samples)

t = 0
si = 1
pos, vel = conf.prtcl_setup()

x_array = [pos]
v_array = [vel]
t_array = [t]

sim_time = 0
t1 = time.time()

for ti in range(si,Nt+1):
    t = ti*dt
    pos, vel = vvb(pos,vel,dt,conf)

    if ti % sample_interval == 0:
        t2 = time.time()
        sim_time += t2-t1
        print("Status: " + time.strftime("%d/%m/%y  %H:%M:%S",time.localtime()) +
        ", step = {0}, t = {1}, time elapsed = {2}s".format(ti,t,np.around(sim_time,decimals=2)))
        t1 = time.time()

        x_array.append(np.copy(pos))
        v_array.append(np.copy(vel))
        t_array.append(np.copy(t))

x_array = np.array(x_array)
v_array = np.array(v_array)
t_array = np.array(t_array)

filename = "{0}vvb_{1}".format(conf.data_root,conf.name)
data_dump(t_array,x_array,v_array,dt,filename+"_full.h5")

if plot == True:
    plot_traj(x_array,filename)
    plot_isotraj(x_array,filename)
    plot_vel(t_array,v_array,filename)
