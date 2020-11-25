import numpy as np
import copy as cp

from gammafac import gu

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
        self.nq = 1
        self.name = "acc"
        self.data_root = "./output/"

        self.x0 = np.zeros((self.nq,3),dtype=np.float)
        self.v0 = np.zeros((self.nq,3),dtype=np.float)

        self.Ex = 1.

        ## Iterate through keyword arguments and store all in object (self)
        for key, value in kwargs.items():
            setattr(self,key,value)


    ###### Electric field call for pusher
    def E(self,t,x):
        E = np.zeros((x.shape[0],3),dtype=np.float)
        E[:,0] = self.Ex

        return E


    ###### Magnetic field call for pusher
    def B(self,t,x):
        B = np.zeros((x.shape[0],3),dtype=np.float)
        return B


    ###### Lorentz force call for pusher
    def F(self,vel,E,B):
        F = self.q/1 * (E + np.cross(vel/(gu(vel,c=self.c)[:,np.newaxis]),B))
        return F

    ###### Particle initialisation for driver
    def prtcl_setup(self):
        x0 = cp.deepcopy(self.x0)
        v0 = cp.deepcopy(self.v0)

        return x0,v0

    ###### Analytical relativistic factor
    def gamma(self,t):
        m = 1.
        gamma = np.sqrt(1+(self.q*self.Ex*t)**2/(m*self.c)**2)
        return gamma

    ###### Analytical position
    def ref_x(self,t):
        m = 1.
        x = m*self.c**2/(self.q*self.Ex) * (self.gamma(t)-1)
        return x

    ###### Analytical velocity
    def ref_vx(self,t):
        m = 1.
        vx = self.q*self.Ex/m * t/self.gamma(t)
        return vx
