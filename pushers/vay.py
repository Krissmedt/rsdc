import numpy as np
from pushers.gammafac import G, gu

def vay(pos,vel,dt,conf,t=0):
    # Synchronised Vay pusher following Ripperda (2018)

    E = conf.E
    B = conf.B
    F = conf.F

    c = conf.c
    q = conf.q
    m = 1.

    xhalf = pos + G(vel,c=c)*dt/2

    # field contribution
    uhalf = vel + dt/2 * F(vel,E(t,xhalf),B(t,xhalf))

    # auxiliary quantities
    udash = uhalf + E(t,xhalf)*q*dt/(2*m)

    tau = B(t,xhalf) * q*dt/(2*m)
    tau_sq = np.linalg.norm(tau**2,axis=1)[:,np.newaxis]
    ustar = arraydot(udash,tau)/c
    gdash = gu(udash,c=c)[:,np.newaxis]
    sigma = gdash**2 - tau_sq
    gnew = np.sqrt((sigma+np.sqrt(sigma**2+4*(tau_sq+ustar**2)))/2)
    t = tau/gnew
    s = 1/(1+np.linalg.norm(t**2,axis=1)[:,np.newaxis])
    # rotation step

    vel = s*(udash + arraydot(udash,t)*t + np.cross(udash,t))

    pos = xhalf + G(vel,c=c)*dt/2

    return pos, vel


def arraydot(a,b):
    return np.sum(a*b,axis=1)[:,np.newaxis]
