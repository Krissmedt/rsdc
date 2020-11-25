import numpy as np
import scipy.optimize as scop
from pushers.gammafac import G, gu
from pushers.gauss_legendre import CollGaussLegendre
from pushers.gauss_lobatto import CollGaussLobatto

def implicit_coll(pos,vel,coll,conf,t=0):
    M = coll.M
    nq = coll.nq
    conf.t = t

    #Remap collocation weights from [0,1] to [tn,tn+1]
    weights =  coll.weights

    for m in range(0,M+1):
        coll.x[m,:,:] = pos
        coll.u[m,:,:] = vel

    Id = coll.Id
    Ix = coll.Ix
    Iv = coll.Iv

    U0 = np.append(coll.x[1:,:,:].ravel(),coll.u[1:,:,:].ravel())
    FU = FXV(U0,coll,conf)
    sol = scop.root(rootF,U0,args=(coll,U0,conf),tol=coll.tolerance,method='krylov',jac=False)
    U = sol.x
    md = M*coll.nq*3
    coll.x[1:,:,:] = U[0:md].reshape((M,coll.nq,3))
    coll.u[1:,:,:] = U[md:].reshape((M,coll.nq,3))
    coll.calc_residual_2018(1,c=conf.c)

    pos = coll.x[-1,:,:]
    vel = coll.u[-1,:,:]

    return pos, vel, coll


def rootF(U,*args):
    coll = args[0]
    U0 = args[1]
    conf = args[2]

    f = U - coll.Q @ FXV(U,coll,conf) - U0

    return f


def FXV(U,coll,conf):
    M = coll.M
    Id = coll.Id
    Ix = coll.Ix
    Iv = coll.Iv
    md = M*coll.nq*3

    F = conf.F
    E = conf.E
    B = conf.B

    x = U[0:md].reshape((M,coll.nq,3))
    v = U[md:].reshape((M,coll.nq,3))
    Fx = np.zeros((M,coll.nq,3),dtype=np.float)
    Fv = np.zeros((M,coll.nq,3),dtype=np.float)
    for m in range(0,M):
        Fx[m,:,:] = G(v[m,:,:],c=conf.c)
        Fv[m,:,:] = F(v[m,:,:],E(conf.t,x[m,:,:]),B(conf.t,x[m,:,:]))

    coll.F[1:,:,:] = Fv
    FXV = np.append(Fx.ravel(),Fv.ravel())
    return FXV
