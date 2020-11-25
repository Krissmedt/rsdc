import numpy as np
from problem import problem
from boris import boris_trick, get_boris_trick_residual
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from boris_sdc import boris_sdc

def variant_a_step(x_n, v_n, dt, prob):
  B = prob.getB(x_n)
  E = 0.5*prob.getE(x_n)
  gamma = prob.getLorentzFactor(v_n)
  c = -(0.5*dt/gamma)*np.cross(v_n, B)
  v_half = boris_trick(v_n, B, E, dt/gamma, dt, c, prob)
  res = get_boris_trick_residual(v_half, v_n, B, E, dt, dt, c, prob)
  print("Variant A Boris trick residual: %7.5e" % res)
  x_np1 = x_n + dt*prob.g(v_half)
  v_np1 = v_half + 0.5*dt*prob.f(x_np1, v_half)
  return x_np1, v_np1

def variant_b_step(xn, vn, dt, prob):
  x_half = xn + dt*vn
  E = prob.getE(x_half)
  B = prob.getB(x_half)
  v_np1 = boris_trick(vn, B, E, dt, dt, 0*x_half, prob)
  x_np1 = x_half + 0.5*dt*prob.g(v_np1)
  return x_np1, v_np1

def our_vel_verlet(x_n, v_n, dt, prob):
  v_half = v_n + 0.5*dt*prob.f(x_n, v_n)
  x_np1  = x_n + dt*prob.g(v_half)
  B = prob.getB(x_np1)
  gamma = prob.getLorentzFactor(v_half)
  c = -(0.5*dt/gamma)*np.cross(v_half, B)
  E = 0.5*prob.getE(x_np1)
  v_np1 = boris_trick(v_half, B, E, dt/gamma, dt, c, prob)
  res = get_boris_trick_residual(v_np1, v_half, B, E, dt, dt, c, prob)
  print("Variant C Boris trick residual: %7.5e" % res)
  print(x_np1)
  return x_np1, v_np1

tend = 0.5
nsteps = 5
dt = tend/float(nsteps)
taxis = np.linspace(0, tend, nsteps+1)

prob = problem()


positions_a = np.zeros((3,nsteps+1))
velocities_a = np.zeros((3,nsteps+1))

positions_b = np.zeros((3,nsteps+1))
velocities_b = np.zeros((3,nsteps+1))

positions_c = np.zeros((3,nsteps+1))
velocities_c = np.zeros((3,nsteps+1))

positions_a[:,0]  = prob.x0
velocities_a[:,0] = prob.v0

positions_b[:,0]  = prob.x0
velocities_b[:,0] = prob.v0


positions_c[:,0]  = prob.x0
velocities_c[:,0] = prob.v0

for n in range(nsteps):
  # positions_a[:,n+1], velocities_a[:,n+1] = variant_a_step(positions_a[:,n], velocities_a[:,n], dt, prob)
  # positions_b[:,n+1], velocities_b[:,n+1] = variant_b_step(positions_b[:,n], velocities_b[:,n], dt, prob)
  positions_c[:,n+1], velocities_c[:,n+1] = our_vel_verlet(positions_c[:,n], velocities_c[:,n], dt, prob)

# kiter = 5
# sdc = boris_sdc(dt = dt, kiter = kiter, nsteps = nsteps, type = 'lobatto', integrate = True)
# positions_sdc, velocities_sdc, stats_sdc = sdc.run()

# energy_a = np.zeros(nsteps+1)
# #energy_b = np.zeros(nsteps+1)
# energy_c = np.zeros(nsteps+1)
# for n in range(nsteps+1):
  # energy_a[n] = prob.getEnergy(positions_a[:,n],velocities_a[:,n])
 # energy_b[n] = prob.getEnergy(positions_b[:,n],velocities_b[:,n])
  # energy_c[n] = prob.getEnergy(positions_c[:,n],velocities_c[:,n])

# fig = plt.figure(1)
# ax = fig.gca(projection='3d')
# ax.plot(positions_a[0,:], positions_a[1,:], positions_a[2,:], color='g')
# # ax.plot(positions_b[0,:], positions_b[1,:], positions_b[2,:], color='y')
# ax.plot(positions_c[0,:], positions_c[1,:], positions_c[2,:], color='b')
# ax.plot(positions_sdc[0,:], positions_sdc[1,:], positions_sdc[2,:], color='r')
# plt.xlim([-20, 20])
# plt.ylim([-20, 20])
# plt.zlim([-15, 15])
# fig.savefig("trajectory.png")

# print("Final energy error a:   %7.5e" % (abs(energy_a[-1] - energy_a[0])/energy_a[0]))
# #print("Final energy error b:   %7.5e" % (abs(energy_b[-1] - energy_b[0])/energy_b[0]))
# print("Final energy error c:   %7.5e" % (abs(energy_c[-1] - energy_c[0])/energy_c[0]))
# print("Final energy error SDC: %7.5e" % stats_sdc['energy_errors'][-1])
# print("Final Residual SDC: %7.5e" % stats_sdc['residuals'][kiter,nsteps-1])
# print("Final Increment SDC: %7.5e" % stats_sdc['increments'][kiter-1,nsteps-1])
# print("Difference SDC to Variant (A): %7.5e" % np.linalg.norm(positions_a[-1,:] - positions_sdc[-1,:], np.inf))
# fig = plt.figure(2)
# plt.semilogy(taxis, abs(energy_a - energy_a[0])/energy_a[0], color='r')
# #plt.semilogy(taxis, abs(energy_b - energy_b[0])/energy_b[0], color='g')
# plt.semilogy(taxis, abs(energy_c - energy_c[0])/energy_c[0], color='b')
# plt.semilogy(taxis[1:], stats_sdc['energy_errors'])
#
# fig = plt.figure(3)
# plt.semilogy(taxis[1:], stats_sdc['residuals'][0,:], 'b', label='Residual')
# plt.semilogy(taxis[1:], stats_sdc['residuals'][1,:], 'b')
# plt.semilogy(taxis[1:], stats_sdc['residuals'][2,:], 'b')
# plt.semilogy(taxis[1:], stats_sdc['residuals'][3,:], 'b')
# plt.semilogy(taxis[1:], stats_sdc['residuals'][4,:], 'b')
# plt.semilogy(taxis[1:], stats_sdc['increments'][0,:], 'r', label='Increment')
# plt.semilogy(taxis[1:], stats_sdc['increments'][1,:], 'r')
# plt.semilogy(taxis[1:], stats_sdc['increments'][2,:], 'r')
# plt.semilogy(taxis[1:], stats_sdc['increments'][3,:], 'r')
# plt.semilogy(taxis[1:], stats_sdc['increments'][4,:], 'r')
# plt.xlabel('Time')
# plt.legend()
# plt.xlim([taxis[1], taxis[-1]])
#
# plt.show()
