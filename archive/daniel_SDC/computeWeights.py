'''
Class to compute weights for collocation formula and Q matrix
'''
import numpy as np

def computeWeights(type):
    
  if type=='lobatto':
    nodes      = [0.0, 0.5, 1.0]
  elif type=='legendre':
    nodes       = [0.5*(1.0 - np.sqrt(3.0/5.0)), 0.5, 0.5*(1.0 + np.sqrt(3.0/5.0))]
  elif type=='radau':
    nodes = [0.15505102572168216745752, 0.64494897427831787695141, 1.0]
  else:
    raise("Invalid type for quadrature nodes")

  # delta_tau stores the distances between the nodes; delta_tau[0] is the distance from left endpoint to first node (zero for Lobatto nodes)
  delta_tau   = np.array([ nodes[0], nodes[1] - nodes[0], nodes[2] - nodes[1] ])
  
  # First lagrange polynomial: Lagrange polynomial j is 1 at node j and zero at all other nodes
  p1 = np.polyfit(nodes, [1.0, 0.0, 0.0], 2)
  p2 = np.polyfit(nodes, [0.0, 1.0, 0.0], 2)
  p3 = np.polyfit(nodes, [0.0, 0.0, 1.0], 2)

  # Compute the integrals of the three Lagrange polynomials: evaluating those at some time t gives int_tn^t l_j(s)~ds
  intp1 = np.polyint(p1)
  intp2 = np.polyint(p2)
  intp3 = np.polyint(p3)
  '''
  @Krasymyr: storing those integrals allows very easily to compute weights w_i to find a value at some arbitrary time tn <= s <= tn+1.
  Simply compute w_j = intpj(s) and then form u0 + w_1*f(u_1) + ... + w_M*f(u_M)
  This should be useful for the bisection algorithm!
  '''

  # Weights for the final integration step correspond to integrals from tn to tnp1
  q     = np.zeros(3)
  q[0] = np.polyval(intp1,1.0)
  q[1] = np.polyval(intp2,1.0)
  q[2] = np.polyval(intp3,1.0)

  # Integrals in S matrix are node to node, so compute int_tj^tj1 = int_t^tj1 - int_t^tj
  S = np.zeros((3,3))
  for j in range(3):
    if j==0:
      S[j,0] = np.polyval(intp1, nodes[j]) - np.polyval(intp1, 0.0)
      S[j,1] = np.polyval(intp2, nodes[j]) - np.polyval(intp2, 0.0)
      S[j,2] = np.polyval(intp3, nodes[j]) - np.polyval(intp3, 0.0)
    else:
      S[j,0] = np.polyval(intp1, nodes[j]) - np.polyval(intp1, nodes[j-1])
      S[j,1] = np.polyval(intp2, nodes[j]) - np.polyval(intp2, nodes[j-1])
      S[j,2] = np.polyval(intp3, nodes[j]) - np.polyval(intp3, nodes[j-1])

  # THIS IS ONLY NEEDED FOR THE STAGGERED SDC VERSION WHICH TURNED OUT NOT TO BE PARTICULARLY USEFUL
  #S1 = np.zeros((3,3))
  #for j in range(3):
  #  if j==0:
  #    pass
  #  else:
  #    S1[j,0] = np.polyval(intp1, half_nodes[j]) - np.polyval(intp1, nodes[j-1])
  #    S1[j,1] = np.polyval(intp2, half_nodes[j]) - np.polyval(intp2, nodes[j-1])
  #    S1[j,2] = np.polyval(intp3, half_nodes[j]) - np.polyval(intp3, nodes[j-1])

  #S2 = np.zeros((3,3))
  #for j in range(3):
  #  if j==0:
  #    pass
  #  else:
  #    S2[j,0] = np.polyval(intp1, nodes[j]) - np.polyval(intp1, half_nodes[j])
  #    S2[j,1] = np.polyval(intp2, nodes[j]) - np.polyval(intp2, half_nodes[j])
  #    S2[j,2] = np.polyval(intp3, nodes[j]) - np.polyval(intp3, half_nodes[j])

  # Generate a polynomial with random coefficient and order = number of nodes and make sure it gets integrated exactly.
  ptest    = np.random.rand(3)
  ptestval = np.polyval(ptest, nodes)
  ptestint = np.polyint(ptest)

  for j in range(3):

    if j==0:
      intexS = np.polyval(ptestint, nodes[0]) - np.polyval(ptestint, 0.0)
      assert np.abs( np.dot(S[0,:], ptestval) - intexS) < 1e-14, "Failed in S."
    else:
      intexS = np.polyval(ptestint, nodes[j]) - np.polyval(ptestint, nodes[j-1])
      assert np.abs( np.dot(S[j,:], ptestval) - intexS) < 1e-14, "Failed in S."

  return S, q, delta_tau
