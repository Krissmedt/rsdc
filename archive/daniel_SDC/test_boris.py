import numpy as np

from boris import boris_trick, get_boris_trick_residual, boris_exact
from problem import problem

prob = problem()

for n in range(1000):
  B = np.random.rand(3)
  E = np.random.rand(3)
  dtE = 100*np.random.rand(1)
  dtB = 100*np.random.rand(1)
  c = np.random.rand(3)
  v_old = np.random.rand(3)

  v_new = boris_trick(v_old, B, E, dtB, dtE, c, prob)
  res = get_boris_trick_residual(v_new, v_old, B, E, dtB, dtE, c, prob)
  if res>1e-15:
    print("Residual from Boris trick: %7.5e" % res)
  v_new = boris_exact(v_old, B, E, dtB, dtE, c, prob)
  res = get_boris_trick_residual(v_new, v_old, B, E, dtB, dtE, c, prob)
  if res>1e-15:
    print("Residual from exact solve: %7.5e" % res)
print("Passed")
