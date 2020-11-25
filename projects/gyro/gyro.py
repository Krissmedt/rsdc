import numpy as np
import copy as cp

from gammafac import gu, vg

##### Problem configuration class
## Use:
## Copy, modify and use in driver script
## OR
## Import to driver script and overwrite E,B,F via the kwarg input

class config:
    def __init__(self,**kwargs):
        #elf.c = 299792458
        self.c = 1.
        self.q = 1.
        self.m = 1.
        self.nq = 1
        self.name = "gyro"
        self.data_root = "./output/"

        self.Bz = 10**6
        self.gamma = 10**6
        self.larmor = 1.
        self.gcentre = np.array([0.,0.,0.])

        self.x0 = np.zeros((self.nq,3),dtype=np.float)
        self.u0 = np.zeros((self.nq,3),dtype=np.float)

        ## Iterate through keyword arguments and store all in object (self)
        for key, value in kwargs.items():
            setattr(self,key,value)


    ###### Electric field call for pusher
    def E(self,t,x):
        E = np.zeros((x.shape[0],3),dtype=np.float)

        return E


    ###### Magnetic field call for pusher
    def B(self,t,x):
        B = np.zeros((x.shape[0],3),dtype=np.float)
        B[:,2] = self.Bz

        return B


    ###### Lorentz force call for pusher
    def F(self,vel,E,B):
        F = self.q/1 * (E + np.cross(vel/(gu(vel,c=self.c)[:,np.newaxis]),B))
        return F

    ###### Particle initialisation for driver
    def prtcl_setup(self):
        x0 = cp.deepcopy(self.x0+self.gcentre)
        u0 = cp.deepcopy(self.u0)

        vperp = vg(self.gamma,c=self.c)

        x0[:,0] += self.larmor
        u0[:,1] = -self.gamma*vperp

        return x0,u0

    ###### Analytical cyclotron freq
    def ref_cyclo(self):
        wc = self.q*self.Bz/(self.gamma*self.m)
        return wc

    def ref_period(self):
        Tc = 2*np.pi*self.gamma*self.m/(self.q*self.Bz)
        return Tc

    def ref_phase(self,t):
        theta = -self.ref_cyclo()*t
        return theta

    def ref_phase_corr(self,t):
        Tc = self.ref_period(t)
        theta_n = -self.ref_cyclo()*t/Tc
        n = np.floor(theta_n)
        theta = -(theta_n-n)*2*np.pi
        return theta

    def phase(self,x,y):
        thetasemi = np.arctan2(y,x)
        if self.gcentre[1] - y >= 0:
            theta = thetasemi
        else:
            theta = -np.pi - (np.pi-thetasemi)
        return theta

    def ref_pos(self,t):
        phase = self.ref_phase(t)
        x = self.larmor * np.cos(phase)
        y = self.larmor * np.sin(phase)
        pos = np.array([x,y,0])
        return pos
