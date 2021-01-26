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
        self.c = 126.491
        self.q = 1
        self.nq = 1
        self.name = "B"
        self.data_root = "./output/"

        self.x0 = np.zeros((self.nq,3),dtype=np.float)
        self.v0 = np.zeros((self.nq,3),dtype=np.float)

        ## Iterate through keyword arguments and store all in object (self)
        for key, value in kwargs.items():
            setattr(self,key,value)


    ###### Electric field call for pusher
    def E(self,t,x):
        E = np.zeros((x.shape[0],3),dtype=np.float)
        nq = x.shape[0]
        E_mat = np.array([[1,0,0],[0,1,0],[0,0,-2]])
        E_mag = 1*(4.9**2 / self.q)
        # E_mag = -0.1
        for pii in range(0,nq):
            E[pii,:] = np.dot(E_mat,x[pii,:]) * E_mag

        #
        #E[:,2] = -0.1
        # E[:,2] = -0.1*x[:,2]
        #E[:,2] = 0.

        return E


    ###### Magnetic field call for pusher
    def B(self,t,x):
        B = np.zeros((x.shape[0],3),dtype=np.float)
        B[:,2] = 1.

        B = B * 25.0/self.q
        return B


    ###### Lorentz force call for pusher
    def F(self,vel,E,B):
        F = self.q/1 * (E + np.cross(vel/(gu(vel,c=self.c)[:,np.newaxis]),B))
        return F


    def prtcl_setup(self):
        self.x0[0,:] = np.array([[10.,0.,0.]])
        self.v0[0,:] = np.array([[100.,0.,100.]])

        x0 = cp.deepcopy(self.x0)
        v0 = cp.deepcopy(self.v0)

        return x0,v0
