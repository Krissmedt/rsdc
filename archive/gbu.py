import numpy as np
from rvv_functions import *

c = 1
u = 145

# gamma = gu(np.array([[u,0,0]]),c=c)

gamma = 150

u1 = 100
u2 = 100

c = np.sqrt((u1**2+u2**2)/(gamma**2-1))
beta = np.sqrt(1-1./gamma**2.)
v = beta*c

print(beta)
print(c)
print(v)


# gamma = np.linspace(1.005,10,100,dtype=np.float)
# gamma = np.array([1.0005,1.00005,1.000005,1.0000005,1.00000005])
#
# for i in range(0,gamma.shape[0]):
#     print("gamma = {0}".format(gamma[i]))
#     beta = np.sqrt(1-1./gamma[i]**2.)
#     u = beta*c
#
#     print("beta = {0}".format(beta))
