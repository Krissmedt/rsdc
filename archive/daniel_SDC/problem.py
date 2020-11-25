'''
Class that contains all parameters defining the Penning trap example.
Defines the magnetic and external electric field via getB and getE functions.
'''
import numpy as np

class problem(object):

  def __init__(self):
    self.e_z = np.array([0, 0, 1])
    self.omega_b = 25.0
    self.alpha   = 1.0
    self.epsilon = -1.0
    self.omega_e = 4.9
    self.E_mat = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, -2] ])
    self.H_mat = np.eye(6)
    self.H_mat[0,0] = self.epsilon*self.omega_e**2
    self.H_mat[1,1] = self.epsilon*self.omega_e**2
    self.H_mat[2,2] = -2*self.epsilon*self.omega_e**2
    self.x0 = np.array([10, 0, 0])
    self.v0 = np.array([100, 0, 100])

    # Parameter for analytical solution
    self.omega       = np.sqrt(-2.0*self.epsilon)*self.omega_e
    self.Omega_plus  = 0.5*(self.omega_b + np.sqrt(self.omega_b**2 + 4.0*self.epsilon*self.omega_e**2))
    self.Omega_minus = 0.5*(self.omega_b - np.sqrt(self.omega_b**2 + 4.0*self.epsilon*self.omega_e**2))
    self.R_minus     = (self.Omega_plus*self.x0[0] + self.v0[1])/(self.Omega_plus - self.Omega_minus)
    self.R_plus      = self.x0[0] - self.R_minus
    self.I_minus     = (self.Omega_plus*self.x0[1] - self.v0[0])/(self.Omega_plus - self.Omega_minus)
    self.I_plus      = self.x0[1] - self.I_minus

    # For relativistic case
#    self.speed_of_light = 299792458
    self.speed_of_light = 29979

  def getB(self,x):
    return (self.omega_b/self.alpha)*self.e_z

  def getE(self,x):
    return -self.epsilon*(self.omega_e**2/self.alpha)*self.E_mat.dot(x)

  def getEnergy(self, x, v):
    u = np.zeros(6)
    u[0:3] = x
    u[3:6] = v
    return np.transpose(u).dot( self.H_mat.dot(u) )

  def pos_exact(self,t):
    w = (self.R_plus + 1j*self.I_plus)*np.exp(-1j*self.Omega_plus*t) + (self.R_minus + 1j*self.I_minus)*np.exp(-1j*self.Omega_minus*t)
    z = self.x0[2]*np.cos(self.omega*t) + (self.v0[2]/self.omega)*np.sin(self.omega*t)
    return w.real, w.imag, z

  def v(self,t):
    return np.ones(3) # complete later

  '''
  Compute the Lorentz factor gamma = sqrt(1 + u^2/c^2) where v is the relativistic velocity.
  '''
  def getLorentzFactor(self, v):
    #return 1.0
    return np.sqrt(1.0 + np.dot(v,v)/self.speed_of_light**2)

  '''
  Define the right hand side function f and g for the SDC formalism
  '''
  def f(self, x, v):
    return (self.getE(x) + np.cross(self.g(v), self.getB(x)))*self.alpha

  def g(self, v):
    return v/self.getLorentzFactor(v)
