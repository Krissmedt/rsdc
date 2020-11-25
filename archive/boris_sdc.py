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

    coll.xn[0,:,:] = coll.x[0,:,:]
    coll.un[0,:,:] = coll.u[0,:,:]
    coll.Fn[0,:,:] = coll.F[0,:,:]

    coll.IV = 0
    coll.IF = 0

    ###### Initial Step #########################
    v_half = vel + 0.5*dm[0]*F(vel,E(t,pos),B(t,pos))
    coll.x[1,:,:] = pos + dm[0]*G(v_half,c=c)

    En         = 0.5*(E(t,pos) + E(t,coll.x[1,:,:]))*qe
    Bn         = B(t,coll.x[1,:,:])*qe
    gamma      = gu(coll.u[0,:,:],c=c)
    c_1        = 0.5*dm[0]*np.cross(G(coll.u[0,:,:],c=c), B(t,coll.x[0,:,:]))*qe
    c_2        = -(0.5*dm[0]/gamma)*np.cross(coll.u[0,:,:], Bn) + c_1
    coll.u[1,:,:] = boris_trick(coll.u[0,:,:],En,Bn,dm[0],gamma,ck=c_2,q=qe)
    coll.F[1,:,:] = F(coll.u[0,:,:],E(t,coll.x[0,:,:]),B(t,coll.x[0,:,:]))

    if coll.predictor == False:
        coll.x[1,:,:] = coll.x[0,:,:]
        coll.u[1,:,:] = coll.u[0,:,:]
        coll.F[1,:,:] = coll.F[0,:,:]

    ############################################
    ######## Predictor Step ####################
    for m in range(1,M):
        v_half = coll.u[m,:,:] + 0.5*dm[m]*coll.F[m,:,:]
        coll.x[m+1,:,:] = coll.x[m,:,:] + dm[m]*G(v_half,c=c)

        En         = 0.5*(E(t,coll.x[m,:,:]) + E(t,coll.x[m+1,:,:]))*qe
        Bn         = B(t,coll.x[m,:,:])*qe
        gamma      = gu(coll.u[m,:,:],c=c)
        c_1        = 0.5*dm[m]*np.cross(G(coll.u[m,:,:],c=c), B(t,coll.x[m,:,:]))*qe
        c_2        = -(0.5*dm[m]/gamma)*np.cross(coll.u[m,:,:], Bn) + c_1
        coll.u[m+1,:,:] = boris_trick(coll.u[m,:,:],En,Bn,dm[m],gamma,ck=c_2,q=qe)
        coll.F[m+1,:,:] = F(coll.u[m+1,:,:],E(t,coll.x[m+1,:,:]),B(t,coll.x[m+1,:,:]))

        if coll.predictor == False:
            coll.x[m+1,:,:] = coll.x[m,:,:]
            coll.u[m+1,:,:] = coll.u[m,:,:]
            coll.F[m+1,:,:] = coll.F[m,:,:]

    coll.calc_residual_2018(0,c=c)

    for k in range(1,K+1):
        coll.IV = 0
        coll.IF = 0

        coll.xn[1,:,:] = coll.x[0,:,:] + coll.IV

        En         = 0.5*(E(t,coll.x[0,:,:]) + E(t,coll.xn[1,:,:]))*qe
        Bn         = B(t,coll.xn[1,:,:])*qe
        gamma      = gu(coll.u[1,:,:],c=c)
        c_1        = 0.5*dm[0]*np.cross(G(coll.u[0,:,:],c=c), B(t,coll.x[0,:,:]))*qe
        c_1       += -0.5*dm[0]* (coll.F[0,:,:] + coll.F[1,:,:])
        c_1       += coll.IF
        c_2        = -(0.5*dm[0]/gamma)*np.cross(coll.u[0,:,:], Bn)
        coll.un[1,:,:] = boris_trick(coll.u[0,:,:],En,Bn,dm[0],gamma,ck=c_2,q=qe)
        coll.Fn[1,:,:] = F(coll.un[1,:,:],E(t,coll.xn[1,:,:]),B(t,coll.xn[1,:,:]))

        if coll.predictor == False:
            coll.xn[1,:,:] = coll.x[1,:,:]
            coll.un[1,:,:] = coll.u[1,:,:]
            coll.Fn[1,:,:] = coll.F[1,:,:]

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
