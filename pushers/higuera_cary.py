import numpy as np
from pushers.gammafac import G, gu

def higuera_cary(pos,vel,dt,conf,t=0):
    # Synchronised Higuera-Cary pusher following Ripperda (2018)

    E = conf.E
    B = conf.B
    F = conf.F

    c = conf.c
    q = conf.q
    m = 1.

    xhalf = pos + G(vel,c=c)*dt/2

    # first half electric acceleration
    um = vel + q*dt/(2*m) * E(t,xhalf)
    gminus = gu(um,c=c)[:,np.newaxis]

    # auxiliary quantities
    tau = B(t,xhalf) * q*dt/(2*m)
    tau_sq = np.linalg.norm(tau**2,axis=1)[:,np.newaxis]
    ustar = arraydot(um,tau)/c
    sigma = gminus**2 - tau_sq
    gplus = np.sqrt((sigma+np.sqrt(sigma**2+4*(tau_sq+ustar**2)))/2)
    t = tau/gplus
    s = 1/(1+np.linalg.norm(t**2,axis=1)[:,np.newaxis])

    # rotation step
    uplus = s*(um + arraydot(um,t)*t + np.cross(um,t))

    # second half electric acceleration
    vel = uplus + q*dt/(2*m) * E(t,xhalf) + np.cross(um,t)

    pos = xhalf + G(vel,c=c)*dt/2

    return pos, vel


def arraydot(a,b):
    return np.sum(a*b,axis=1)[:,np.newaxis]


def hc_velocity_avg(x,u,E,B,dt,q=-1,ck=0):
    # Higuera-Cary variant of Boris leapfrog integration, using different
    # velocity average in Lorentz force discretisation.
    um = u + dt/2 * (q*E/1 + ck)

    t = q*B/(2*1) * dt
    g2 = g_biquad(um,t)
    t = t/g2
    ud = um + np.cross(um,t)

    up = um + np.cross(ud,2*t/(1+np.linalg.norm(t**2,axis=1)[:,np.newaxis]))

    vel = up + dt/2 * (q*E/1 + ck)

    return vel


def g_biquad(um,beta):
    b = np.linalg.norm(beta,axis=1)[:,np.newaxis]
    gm = gu(um)[:,np.newaxis]

    dot = np.linalg.norm(b*um**2,axis=1)
    inner_sq = np.sqrt((gm**2+b**2)**2 + 4*(b**2+dot[:,np.newaxis]))
    g2 = np.sqrt(0.5*(gm**2-b**2+inner_sq))

    return g2
