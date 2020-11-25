import numpy as np
from pushers.gammafac import *

def boris_verbon(u,E,B,dt,ck=0,q=-1,c=1):
    # Classic Boris following Verbonceour (2015) particle-in-cell review
    # where magnetic vector t includes tan evaluation of the angle directly.
    # Option for Tretiak (2019) style extra constant term "ck".
    # Treat q as charge-mass ratio if needed.

    um = u + dt/2 * (q*E/1 + ck)
    g = gu(um,c=c)[:,np.newaxis]
    Bmag = np.linalg.norm(B,axis=1)[:,np.newaxis]
    Bdir = np.nan_to_num(B/Bmag)

    t = Bdir * np.tan(q*dt*Bmag/(2*g*1))
    ud = um + np.cross(um,t)

    up = um + np.cross(ud,2*t/(1+np.linalg.norm(t**2,axis=1)[:,np.newaxis]))
    # up = um + np.cross(ud,2*t/(1+t*t))
    vel = up + dt/2 * (q*E/1 + ck)

    return vel

def boris_classic(u,E,B,dt,ck=0,q=-1,c=1):
    # Classic Boris following Birdsall and Langdon ())
    # Option for Tretiak (2019) style extra constant term "ck".
    # Treat q as charge-mass ratio if needed.

    um = u + dt/2 * (q*E/1 + ck)
    g = gu(um,c=c)[:,np.newaxis]

    t = q*dt*B/(2*g*1)

    ud = um + np.cross(um,t)

    up = um + np.cross(ud,2*t/(1+np.linalg.norm(t**2,axis=1)[:,np.newaxis]))
    # up = um + np.cross(ud,2*t/(1+t*t))
    vel = up + dt/2 * (q*E/1 + ck)

    return vel

def boris_trick(u,E,B,dt,gamma,ck=0,q=-1):
    # Freeform Boris following Paper (2021), with Boris used as a general
    # algorithm to solve implicit velocity equation.
    # Relativistic factor (gamma) estimate must be provided seperately here.
    # Treat q as charge-mass ratio if needed.

    t = 0.5*dt*q*B/gamma[:,np.newaxis]
    s = 2.0*t/(1.0 + np.linalg.norm(t**2,axis=1)[:,np.newaxis])
    u_min = u + 0.5*dt*q*E + 0.5*ck
    u_star  = u_min + np.cross(u_min, t)
    u_plus  = u_min + np.cross(u_star, s)

    return u_plus + 0.5*dt*q*E + 0.5*ck






def boris_nr(vel, E, B, dt, alpha=1, ck=0):
    """
    Applies Boris' trick for given velocity, electric and magnetic
    field for vector data in the shape (N x 3), i.e. particles as rows
    and x,y,z components for the vector as the columns.
    k = delta_t * alpha / 2
    """

    k = dt*alpha/2

    tau = k*B

    vMinus = vel + dt/2 * (alpha*E + ck)

    tauMag = np.linalg.norm(tau,axis=1)
    vDash = vMinus + np.cross(vMinus,tau)

    tm = 2/(1+tauMag**2)

    for col in range(0,3):
        vDash[:,col] = tm[:] * vDash[:,col]

    vPlus = vMinus + np.cross(vDash,tau)

    vel_new = vPlus + dt/2 * (alpha*E + ck)

    return vel_new




# u = np.array([[40.,20.,10.]])
# B = np.array([[0.,0.,25.]])
# E = np.array([[240.1,0.,0.]])
# dt = 0.01
#
# vel1 = boris_classic(u,E,B,dt)
# vel2 = boris_classic2(u,E,B,dt)
# gamma = gu(u+dt/2*(-1*E/1))
# vel3 = boris_freeform(u,E,B,dt,gamma)
#
#
# print(vel1)
# print(vel2)
# print(vel3)
