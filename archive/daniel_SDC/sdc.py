'''
The parent class for all SDC implementations. Contains all functionality shared by Verlet Boris-SDC in normal and matrix formulation and GMRES Boris-SDC
'''
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from computeWeights import computeWeights

from problem import problem

class sdc(object):

  def __init__(self, dt, kiter, nsteps, type, integrate=True):

    self.dt     = dt
    self.kiter  = kiter
    self.nsteps = nsteps

    self.S, self.q, self.delta_tau = computeWeights(type)
    self.S  *= dt
    self.q  *= dt
    self.delta_tau *= dt

    self.integrate = integrate
    if type=='legendre':
      if not self.integrate:
        print("Overiding integrate=False... for Legendre nodes, integrate needs to be set to True")
        self.integrate = True
    if (self.kiter==0) and (self.integrate):
      print("Setting kiter=0 but integrate=True will NOT reproduce the standard Boris integrator")

    # Define the problem
    self.P = problem()

    self.stats = {}
    self.stats['residuals']     = np.zeros((kiter+1,nsteps))
    self.stats['increments']    = np.zeros((kiter,nsteps))
    self.stats['energy_errors'] = np.zeros(nsteps)

    # Buffers to store approximations at time steps
    self.positions       = np.zeros((3,nsteps+1))
    self.velocities      = np.zeros((3,nsteps+1))
    self.positions[:,0]  = self.P.x0
    self.velocities[:,0] = self.P.v0

    self.stats['exact_energy'] = self.P.getEnergy(self.positions[:,0], self.velocities[:,0])

    self.stats['errors']      = np.zeros((2,nsteps+1))
    self.stats['errors'][0,0] = 0.0
    self.stats['errors'][1,0] = 0.0

  def F(self,x,v):
    return self.P.f(x,v)

  def G(self,v):
    return self.P.g(v)

  def getEnergyError(self, x, v):
    return np.abs( self.P.getEnergy(x, v) - self.stats['exact_energy'])/np.abs(self.stats['exact_energy'])

  def updateIntegrals(self, x, v):
    F = np.zeros((3,3))
    G = np.zeros((3,3))
    for jj in range(3):
      F[:,jj] = self.F(x[:,jj], v[:,jj]) # F[:,jj] = F_j
      G[:,jj] = self.G(v[:,jj])

    # Set integral terms to zero
    self.I_m_mp1    = np.zeros((3,3))
    self.IV_m_mp1   = np.zeros((3,3))

    for jj in range(3):
      # self.I_m_mp1[:,jj] equals I_j^j+1
      for kk in range(3):
        self.I_m_mp1[:,jj]    += self.S[jj,kk]*F[:,kk]
        self.IV_m_mp1[:,jj]   += self.S[jj,kk]*G[:,kk]

  def getResiduals(self, x0, v0, x, v):

    self.updateIntegrals(x, v)

      # Compute residuals
    res_x = np.zeros((3,3))
    res_v = np.zeros((3,3))

    res_v[:,0] = v[:,0] - v0     - self.I_m_mp1[:,0]
    res_v[:,1] = v[:,1] - v[:,0] - self.I_m_mp1[:,1]
    res_v[:,2] = v[:,2] - v[:,1] - self.I_m_mp1[:,2]

    res_x[:,0] = x[:,0] - x0     - self.IV_m_mp1[:,0]
    res_x[:,1] = x[:,1] - x[:,0] - self.IV_m_mp1[:,1]
    res_x[:,2] = x[:,2] - x[:,1] - self.IV_m_mp1[:,2]

    return max(np.linalg.norm(res_v, np.inf), np.linalg.norm(res_x, np.inf))

  def finalUpdateStep(self, x, v, x0, v0):
    self.integrate = False
    if self.integrate:
      F = np.zeros((3,3))
      G = np.zeros((3,3))
      for jj in range(3):
        F[:,jj] = self.F(x[:,jj], v[:,jj]) # F[:,jj] = F_j
        G[:,jj] = self.G(v[:,jj])
      x_final = np.copy(x0)
      v_final = np.copy(v0)
      for jj in range(3):
        x_final += self.q[jj]*G[:,jj]
        v_final += self.q[jj]*F[:,jj]
    else:
      x_final = x[:,2]
      v_final = v[:,2]
    return x_final, v_final
