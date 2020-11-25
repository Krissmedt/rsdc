'''
Class implementing Boris-SDC based on Verlet integrator in normal node-to-node form
'''
import numpy as np
from sdc import sdc
from boris import boris_trick, get_boris_trick_residual, boris_exact

class boris_sdc(sdc):

  def __init__(self, dt, kiter, nsteps, type, integrate=True):

    super(boris_sdc, self).__init__(dt, kiter, nsteps, type, integrate)

  def boris_solve(self, v, B, E, dtB, dtE, c, gamma):
    #return boris_exact(v, B, E, dtB, dtE, c, self.P)
    return boris_trick(v, B, E, dtB/gamma, dtE, c, self.P)

  def run(self):

    print (" ==== New Boris SDC time step ===== \n ")
    # Buffers for k+1 and k solution at integer step
    x_old = np.zeros((3,3))
    v_old = np.zeros((3,3))
    x_new = np.zeros((3,3))
    v_new = np.zeros((3,3))

    for nn in range(self.nsteps):

      '''
      First Velocity Half Step
      '''
      v_half = self.velocities[:,nn] + 0.5*self.delta_tau[0]*self.F(self.positions[:,nn], self.velocities[:,nn])

      '''
      Position Step
      '''
      x_old[:,0] = self.positions[:,nn] + self.delta_tau[0]*self.G(v_half)

      '''
      Second Velocity Half Step using Boris trick
      '''
      E          = 0.5*( self.P.getE(self.positions[:,nn]) + self.P.getE(x_old[:,0]) )*self.P.alpha
      B          = self.P.getB(x_old[:,0])*self.P.alpha
      gamma      = self.P.getLorentzFactor(self.velocities[:,nn])
      c_1        = 0.5*self.delta_tau[0]*np.cross(self.G(self.velocities[:,nn]), self.P.getB(self.positions[:,nn]))*self.P.alpha
      c_2        = -(0.5*self.delta_tau[0]/gamma)*np.cross(self.velocities[:,nn], B) + c_1
      v_old[:,0] = self.boris_solve(self.velocities[:,nn], B, E, self.delta_tau[0], self.delta_tau[0], c_2, gamma)
      res = get_boris_trick_residual(v_old[:,0], self.velocities[:,nn], B, E, self.delta_tau[0], self.delta_tau[0], c_2, self.P)
      # print("Boris trick residual: %7.5e" % res )

      '''
      Predictor step: populate self.x_old and self.v_old
      '''
      for j in range(1,3): # note: we use M=3 nodes here and this is hardcoded
        '''
        First Velocity Half Step
        '''
        v_half     = v_old[:,j-1] + 0.5*self.delta_tau[j]*self.F(x_old[:,j-1], v_old[:,j-1])

        '''
        Position Step
        '''
        x_old[:,j] = x_old[:,j-1] + self.delta_tau[j]*self.G(v_half)

        '''
        Second Velocity Half Step using Boris
        '''
        E          = 0.5*(self.P.getE(x_old[:,j-1]) +  self.P.getE(x_old[:,j]) )*self.P.alpha
        B          = self.P.getB(x_old[:,j])*self.P.alpha
        gamma      = self.P.getLorentzFactor(v_old[:,j-1])
        c_1        = 0.5*self.delta_tau[j]*np.cross(self.G(v_old[:,j-1]), self.P.getB(x_old[:,j-1]))*self.P.alpha
        c_2        = -(0.5*self.delta_tau[j]/gamma)*np.cross(v_old[:,j-1], B) + c_1
        v_old[:,j] = self.boris_solve(v_old[:,j-1], B, E, self.delta_tau[j], self.delta_tau[j], c_2, gamma)
        res = get_boris_trick_residual(v_old[:,j], v_old[:,j-1], B, E, self.delta_tau[j], self.delta_tau[j], c_2, self.P)
        # print("Boris trick residual: %7.5e" % res )

      '''
      SDC iteration
      '''
      for kk in range(self.kiter):
        # Update integral terms... this also computes the residuals
        self.stats['residuals'][kk,nn] = self.getResiduals(self.positions[:,nn], self.velocities[:,nn], x_old, v_old)

        # First value at new iteration is equal to starting value (Lobatto nodes)
        x_new[:,0]      = self.positions[:,nn] + self.IV_m_mp1[:,0]
        gamma           = self.P.getLorentzFactor(v_old[:,0])
        E               = 0.5*(self.P.getE(self.positions[:,nn]) + self.P.getE(x_new[:,0]) )*self.P.alpha
        B               = self.P.getB(x_new[:,0])*self.P.alpha
        c_1             = 0.5*self.delta_tau[0]*np.cross( self.G(self.velocities[:,nn]), self.P.getB(self.positions[:,nn] ))
        c_1            += -0.5*self.delta_tau[0]*( self.F(self.positions[:,nn],self.velocities[:,nn]) + self.F(x_old[:,0], v_old[:,0]))
        c_1            += self.I_m_mp1[:,0]
        c_2             = -(0.5*self.delta_tau[0]/gamma)*np.cross(self.velocities[:,nn], B)
        ### c_2             = -(0.5*self.delta_tau[0])*np.cross(self.G(self.velocities[:,nn]), B)

        v_new[:,0]      = self.boris_solve(self.velocities[:,nn], B, E, self.delta_tau[0], self.delta_tau[0], c_2, gamma)
        res = get_boris_trick_residual(v_new[:,0], self.velocities[:,nn], B, E, self.delta_tau[0], self.delta_tau[0], c_2, self.P)
        # print("Boris trick residual: %7.5e" % res )
        # print(x_new[:,0])
        for j in range(1,3):
          '''
          First Velocity Half Step
          '''
          v_old_half = v_old[:,j-1] + 0.5*self.delta_tau[j]*self.F(x_old[:,j-1],v_old[:,j-1])
          v_new_half = v_new[:,j-1] + 0.5*self.delta_tau[j]*self.F(x_new[:,j-1],v_new[:,j-1])

          '''
          Position Step
          '''
          x_new[:,j] = x_new[:,j-1] + self.delta_tau[j]*( self.G(v_new_half) - self.G(v_old_half) ) + self.IV_m_mp1[:,j]

          '''
          Second Velocity Half Step
          '''
          E          = 0.5*( self.P.getE(x_new[:,j-1]) + self.P.getE(x_new[:,j]) )*self.P.alpha
          B          = self.P.getB(x_new[:,j])*self.P.alpha
          gamma      = self.P.getLorentzFactor(v_old[:,j])

          c_1             = 0.5*self.delta_tau[j]*np.cross( self.G(v_new[:,j-1]), self.P.getB(x_new[:,j-1]) )
          c_1            += -0.5*self.delta_tau[j]*( self.F(x_old[:,j],v_old[:,j]) + self.F(x_old[:,j-1], v_old[:,j-1]))
          c_1            += self.I_m_mp1[:,j]
          c_2             = -(0.5*self.delta_tau[j]/gamma)*np.cross(v_new[:,j-1], B) + c_1
          ### c_2             = -(0.5*self.delta_tau[j])*np.cross(self.G(v_new[:,j-1]), B) + c_1

          v_new[:,j] = self.boris_solve(v_new[:,j-1], B, E, self.delta_tau[j], self.delta_tau[j], c_2, gamma)
          res = get_boris_trick_residual(v_new[:,j], v_old[:,j], B, E, self.delta_tau[j]/gamma, self.delta_tau[j], c_2, self.P)
          # print("Boris trick residual: %7.5e" % res)


        ### Prepare next iteration
        self.stats['increments'][kk,nn] = max( np.linalg.norm(x_new - x_old, np.inf) , np.linalg.norm(v_new - v_old, np.inf) )
        x_old      = np.copy(x_new)
        v_old      = np.copy(v_new)

      '''
      Prepare next time step
      '''
      # Compute residual after final iteration
      self.stats['residuals'][self.kiter,nn] = self.getResiduals(self.positions[:,nn], self.velocities[:,nn], x_old, v_old)

      self.positions[:,nn+1], self.velocities[:,nn+1] = self.finalUpdateStep(x_old, v_old, self.positions[:,nn], self.velocities[:,nn])
      self.stats['energy_errors'][nn] = self.getEnergyError(self.positions[:,nn+1], self.velocities[:,nn+1])
      # print(self.positions[:,nn+1])
      # print(self.velocities[:,nn+1])
    return self.positions, self.velocities, self.stats
