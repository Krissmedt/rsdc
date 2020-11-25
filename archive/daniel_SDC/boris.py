import numpy as np
from scipy import optimize
from scipy.optimize import show_options

'''
Solves the implicit equation approximately using Boris trick
v = w + dt_E*E + 0.5*dt_B*[ (g(v) + g(w)) x B ] + c
'''
def boris_trick(v_old, B, E, dtB, dtE, c, prob):
  t       = 0.5*dtB*B
  s       = 2.0*t/(1.0 + np.dot(t,t))
  v_min   = v_old + 0.5*dtE*E + 0.5*c
  v_star  = v_min + np.cross(v_min, t)
  v_plus  = v_min + np.cross(v_star, s)
  return v_plus + 0.5*dtE*E + 0.5*c

def get_boris_defect(v_new, v_old, B, E, dtB, dtE, c, prob):
  return v_new - v_old - dtE*E - dtB*0.5*np.cross( prob.g(v_old) + prob.g(v_new), B ) - c

def get_boris_trick_residual(v_new, v_old, B, E, dtB, dtE, c, prob):
  defect = get_boris_defect(v_new, v_old, B, E, dtB, dtE, c, prob)
  return np.linalg.norm(defect, np.inf)

def boris_exact(v_old, B, E, dtB, dtE, c, prob):
  fun = lambda x : get_boris_defect(x, v_old, B, E, dtB, dtE, c, prob)
  #show_options(solver="root", method="hybr")
  sol = optimize.root(fun, v_old, jac=None, method='krylov', options={'xtol':1e-14, 'maxfev':500})
  assert sol.success, "Root finding method failed to converge"
  print("quality of root finder: %7.5e" % np.linalg.norm(sol.fun, np.inf))
  return sol.x
