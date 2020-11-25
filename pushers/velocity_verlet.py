import numpy as np
from pushers.gammafac import G, gu
from pushers.boris import boris_classic, boris_verbon, boris_trick


def velocity_verlet_A(pos,vel,dt,conf,t=0):
    E = conf.E
    B = conf.B
    F = conf.F

    c = conf.c
    q = conf.q

    En = E(t,pos)
    Bn = B(t,pos)

    uhalf = vel + F(vel,En,Bn)*dt/2
    pos =  pos + G(uhalf,c=c)*dt

    Eh = (E(t,pos) + En)/2
    Bh = (B(t,pos) + Bn)/2

    vel = boris_verbon(vel,Eh,Bh,dt,q=q,c=c)

    return pos, vel



def velocity_verlet_B(pos,vel,dt,conf,t=0):
    E = conf.E
    B = conf.B
    F = conf.F

    c = conf.c
    q = conf.q

    En = E(t,pos)
    Bn = B(t,pos)

    uhalf = vel + F(vel,En,Bn)*dt/2
    pos =  pos + G(uhalf,c=c)*dt

    Eh = (E(t,pos) + En)/2
    Bh = (B(t,pos) + Bn)/2

    vel = boris_classic(vel,Eh,Bh,dt,q=q,c=c)

    return pos, vel


def velocity_verlet_C(pos,vel,dt,conf,t=0):
    E = conf.E
    B = conf.B
    F = conf.F

    c = conf.c
    q = conf.q

    En = E(t,pos)
    Bn = B(t,pos)

    uhalf = vel + F(vel,En,Bn)*dt/2
    pos =  pos + G(uhalf,c=c)*dt

    Eh = (E(t,pos) + En)/2
    Bh = (B(t,pos) + Bn)/2

    gamma = gu(vel + dt/2 * (q*Eh/1),c=c)
    vel = boris_freeform(vel,Eh,Bh,dt,gamma,q=q)

    return pos, vel


def velocity_verlet_D(pos,vel,dt,conf,t=0):
    # First order, doesn't work as intended
    E = conf.E
    B = conf.B
    F = conf.F

    c = conf.c
    q = conf.q

    uhalf = vel + F(vel,E(t,pos),B(t,pos))*dt/2
    pos =  pos + G(uhalf,c=c)*dt

    gamma = gu(uhalf,c=c)
    ck = -(0.5*dt/gamma[:,np.newaxis])*np.cross(uhalf,B(t,pos))
    vel = boris_freeform(uhalf,0.5*E(t,pos),B(t,pos),dt,gamma,ck=ck,q=q)

    return pos, vel
