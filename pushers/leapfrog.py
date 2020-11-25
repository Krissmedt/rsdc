import numpy as np
from pushers.gammafac import G, gu
from pushers.boris import boris_classic


def leapfrog(pos,vel,dt,conf,t=0):
    E = conf.E
    B = conf.B

    c = conf.c
    q = conf.q

    vel = boris_classic(vel,E(t,pos),B(t,pos),dt,c=c,q=q)

    pos = pos + G(vel,c=c)*dt

    return pos, vel


def sync_leapfrog(pos,vel,dt,conf,t=0):
    E = conf.E
    B = conf.B

    c = conf.c
    q = conf.q


    vel = boris_classic(vel,E(t,pos),B(t,pos),dt/2,c=c,q=q)

    pos  = pos + G(vel,c=c)*dt

    return pos, vel
