#!/usr/bin/env python
from __future__ import print_function
from scipy.optimize import linprog
from SetCoverPy import setcover as scp
import numpy as np
import time

##############################################################################################################
### Solve the geometric Set Cover Problem (geoSCP)
# geoSCP input: 
#   * floor_validids_per_tar, floor_validids, targets_reachable: output of mobotscp.connection
# SCP input:
#   * U = {e1, e2, ..., en}: the universe is a set of n elements
#   * S = {S1, S2, ..., Sm}: the collection of m sets, each is a subset of U
#   * E = {E1, E2, ..., En}: the collection of n sets Ei, each records the indices of Sj's that cover ei
# SCP output:
#   * sol = minimum subset of indices {1, 2, ..., m} such that the corresponding sets cover all elements of U
# geoSCP output:
#   * floor_chosenids: indices of chosen points on the floor that together cover all targets 
#   * cost: minimum number of points on the floor to cover all targets 
#   * SCPtime: time used to solve the Set Cover Problem (SCP)
##############################################################################################################


def get_math_model(floor_validids_per_tar, floor_validids, targets_reachable):
  U = targets_reachable
  n = len(U)
  m = len(floor_validids)
  E = []
  for i in targets_reachable:
    elements_per_set = []
    for j in range(len(floor_validids_per_tar[i])):
      elements_per_set.append(floor_validids.index(floor_validids_per_tar[i][j]))
    E.append(elements_per_set)
  print("Size of universe: n = {}".format(n))
  print("Total number of sets in collection: m = {}".format(m))
  return n, m, E


def solver_LPr(n, m, E, maxiters=20):
  """
  Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
  min_x c^T . x
  subject to:
    A_ub . x <= b_ub
    A_eq . x == b_eq
    lb <= x <= ub
  """
  print("--Solving SCP using 'Linear Programming relaxation & rounding (LPr)' solver...")
  starttime = time.time()
  # matrix and vectors for LP
  b_ub = -np.ones(n)
  c = np.ones(m)
  A_ub = np.zeros((n,m))
  for i in range(n):
    for j in E[i]:
      A_ub[i,j] = -1
  # solve LP relaxation
  options = {"disp": True, "maxiter": maxiters*200}
  bounds = [(0,1) for i in range(m)]
  res = linprog(c, A_ub=A_ub/1000, b_ub=b_ub/1000, bounds=bounds, options=options)
  if not res.success:
    raise ValueError("Linear Programming relaxation & rounding (LPr) solver failed: {}".format(res.message))
  raw_sol = res.x
  min_mincost = res.fun
  # rouding
  f = max(abs(np.sum(A_ub,1)))
  print("LP deterministic rounding factor f = {}".format(f))
  sol = np.zeros(len(raw_sol))
  for i in range(len(sol)):
    if raw_sol[i] > 1/f:
      sol[i] = 1
  mincost = np.sum(sol)
  SCPtime = time.time() - starttime
  return sol, mincost, SCPtime


def solver_SCPy(n, m, E, maxiters=20):
  """
  Based on: https://github.com/guangtunbenzhu/SetCoverPy
  Input: 
    -- a_matrix[nrows, mcols], the binary relationship matrix
       a_matrix[irow, jcol] = True if jcol covers irow
    -- cost[mcols], the cost of columns. 
  """
  print("--Solving SCP using 'SetCoverPy (SCPy)' solver...")
  starttime = time.time()
  c = np.ones(m)
  A = np.zeros((n,m), dtype=bool)
  for i in range(n):
    for j in E[i]:
      A[i,j] = True
  g = scp.SetCover(amatrix=A, cost=c, maxiters=maxiters, subg_maxiters=200)
  res, solvertime = g.SolveSCP()
  if res is None:
    raise ValueError("SetCoverPy (SCPy) solver failed!")
  sol = g.s.astype(int)
  mincost = g.total_cost
  SCPtime = time.time() - starttime
  return sol, mincost, SCPtime


def solver_greedy(n, m, E, maxiters=20):
  """
  Based on: https://github.com/guangtunbenzhu/SetCoverPy
  Input: 
    -- a_matrix[nrows, mcols], the binary relationship matrix
       a_matrix[irow, jcol] = True if jcol covers irow
    -- cost[mcols], the cost of columns. 
  """
  print("--Solving SCP using 'greedy' solver...")
  starttime = time.time()
  c = np.ones(m)
  A = np.zeros((n,m), dtype=bool)
  for i in range(n):
    for j in E[i]:
      A[i,j] = True
  g = scp.SetCover(amatrix=A, cost=c, maxiters=maxiters)
  g.greedy()
  sol = g.s.astype(int)
  mincost = g.total_cost
  SCPtime = time.time() - starttime
  return sol, mincost, SCPtime


def solve_geoSCP(floor_validids_per_tar, floor_validids, targets_reachable, solver='SCPy', maxiters=20):
  n, m, E = get_math_model(floor_validids_per_tar, floor_validids, targets_reachable)
  if solver=='SCPy':
    sol, cost, SCPtime = solver_SCPy(n, m, E, maxiters)
  elif solver=='greedy':
    sol, cost, SCPtime = solver_greedy(n, m, E, maxiters)
  elif solver=='LPr':
    sol, cost, SCPtime = solver_LPr(n, m, E, maxiters)
  else:
    raise ValueError("The specified SCP solver is not supported. Valid values: solver='SCPy','greedy','LPr'")
  floor_chosenids = np.array(floor_validids)[np.flatnonzero(sol)]
  print("--Solving SCP finished, results:")
  print("SCP solution I = \n{} \n  (1 means the index is chosen, 0 means not chosen)".format(sol))
  print("  corresponds to floor's chosen indices = {}".format(floor_chosenids))
  print("SCP min total cost = {} points".format(cost))
  print("SCP solver time used = {} s".format(SCPtime))
  return floor_chosenids, cost, SCPtime

# END