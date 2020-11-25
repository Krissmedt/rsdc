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
        self.name = "forcefree"
        self.data_root = "./output/"

        self.Bz = 1.
        self.gamma = 10.**6
        self.vy = -vg(self.gamma,c=self.c)
        self.Ex = -self.vy*self.Bz

        self.x0 = np.zeros((self.nq,3),dtype=np.float)
        self.u0 = np.zeros((self.nq,3),dtype=np.float)

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
        B[:,2] = self.Bz

        return B


    ###### Lorentz force call for pusher
    def F(self,vel,E,B):
        F = self.q/1 * (E + np.cross(vel/(gu(vel,c=self.c)[:,np.newaxis]),B))
        return F

    ###### Particle initialisation for driver
    def prtcl_setup(self):
        x0 = cp.deepcopy(self.x0)
        u0 = cp.deepcopy(self.u0)

        u0[0,1] = self.gamma*self.vy

        return x0,u0
