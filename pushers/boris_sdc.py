import numpy as np
import scipy.optimize as scop
from pushers.gammafac import G, gu
from pushers.boris import boris_trick
from pushers.gauss_legendre import CollGaussLegendre
from pushers.gauss_lobatto import CollGaussLobatto

def boris_SDC(pos,vel,coll,conf,t=0):
    E = conf.E
    B = conf.B
    F = conf.F
    c = conf.c
    qe = conf.q

    M = coll.M
    K = coll.K
    nq = coll.nq

    #Remap collocation weights from [0,1] to [tn,tn+1]
    weights =  coll.weights

    q =  coll.Qmat

    dm =  coll.delta_m

    ## Populate node solutions with x0, v0, F0 ##
    coll.x[0,:,:] = pos
    coll.u[0,:,:] = vel
    coll.F[0,:,:] = F(vel,E(t,pos),B(t,pos))

    for m in range(1,M+1):
        coll.x[m,:,:] = coll.x[0,:,:]
        coll.u[m,:,:] = coll.u[0,:,:]
        coll.F[m,:,:] = coll.F[0,:,:]

    for m in range(0,2):
        coll.xn[m,:,:] = coll.x[0,:,:]
        coll.un[m,:,:] = coll.u[0,:,:]
        coll.Fn[m,:,:] = coll.F[0,:,:]

    coll.IV = 0
    coll.IF = 0

    predictor = simple_predict
    if coll.predictor == True:
        predictor = verlet_predict

    ###### Initial Step #########################
    coll.x, coll.u, coll.F = predictor(0,coll.x,coll.u,coll.F,coll,conf)

    ############################################
    ######## Predictor Step ####################
    for m in range(1,M):
        coll.x, coll.u, coll.F = predictor(m,coll.x,coll.u,coll.F,coll,conf)

    coll.calc_residual_2018(0,c=c)

    for k in range(1,K+1):
        for m in range(coll.ssi,M):
            # Calculate collocation terms required for pos update
            coll.IV = 0
            for j in range(1,M+1):
                coll.IV += (q[m+1,j]-q[m,j])*G(coll.u[j,:,:],c=c)

            v_half = coll.u[m,:,:] + 0.5*dm[m]*coll.F[m,:,:]
            vn_half = coll.un[m,:,:] + 0.5*dm[m]*coll.Fn[m,:,:]

            ### POSITION UPDATE FOR NODE m/SWEEP k ###
            coll.xn[m+1,:,:] = coll.xn[m,:,:]
            coll.xn[m+1,:,:] += dm[m]* (G(vn_half,c=c)-G(v_half,c=c))
            coll.xn[m+1,:,:] += coll.IV

            # Calculate collocation terms required for pos update
            coll.IF = 0
            for j in range(1,M+1):
                coll.IF += (q[m+1,j]-q[m,j])*coll.F[j,:,:]

            En         = 0.5*(E(t,coll.xn[m+1,:,:]) + E(t,coll.xn[m,:,:]))*qe
            Bn         = B(t,coll.xn[m+1,:,:])*qe
            gamma      = gu(coll.u[m+1,:,:],c=c)

            c_1        = 0.5*dm[m]*np.cross(G(coll.un[m,:,:],c=c), B(t,coll.xn[m,:,:]))*qe
            c_1       += -0.5*dm[m]* (coll.F[m+1,:,:] + coll.F[m,:,:])
            c_1       += coll.IF
            c_2        = -(0.5*dm[m]/gamma)*np.cross(coll.un[m,:,:], Bn) + c_1
            coll.un[m+1,:,:] = boris_trick(coll.un[m,:,:],En,Bn,dm[m],gamma,ck=c_2,q=qe)
            coll.Fn[m+1,:,:] = F(coll.un[m+1,:,:],E(t,coll.xn[m+1,:,:]),B(t,coll.xn[m+1,:,:]))

        coll.F = np.copy(coll.Fn[:,:,:])
        coll.x = np.copy(coll.xn[:,:,:])
        coll.u = np.copy(coll.un[:,:,:])
        coll.calc_residual_2018(k,c=c)

    pos = np.copy(coll.xn[m+1,:,:])
    vel = np.copy(coll.un[m+1,:,:])

    return pos, vel, coll


def simple_predict(m,xarr,uarr,Farr,coll,conf,t=0):
    # Dummy method for predictors
    # default is to simply let the arrays be populated by initial conditions

    return xarr,uarr,Farr


def verlet_predict(m,xarr,uarr,Farr,coll,conf,t=0):
    E = conf.E
    B = conf.B
    F = conf.F
    c = conf.c
    q = conf.q
    dm = coll.delta_m

    v_half = uarr[m,:,:] + 0.5*dm[m]*Farr[m,:,:]
    xarr[m+1,:,:] = xarr[m,:,:] + dm[m]*G(v_half,c=c)

    En         = 0.5*(E(t,xarr[m,:,:]) + E(t,xarr[m+1,:,:]))*q
    Bn         = B(t,xarr[m,:,:])*q
    gamma      = gu(uarr[m,:,:],c=c)
    c_1        = 0.5*dm[m]*np.cross(G(uarr[m,:,:],c=c), B(t,xarr[m,:,:]))*q
    c_2        = -(0.5*dm[m]/gamma)*np.cross(uarr[m,:,:], Bn) + c_1
    uarr[m+1,:,:] = boris_trick(uarr[m,:,:],En,Bn,dm[m],gamma,ck=c_2,q=q)
    Farr[m+1,:,:] = F(uarr[m+1,:,:],E(t,xarr[m+1,:,:]),B(t,xarr[m+1,:,:]))

    return xarr,uarr,Farr
