#!/usr/bin/env python
from __future__ import print_function
from mobotscp import utils
from scipy.optimize import linprog
from SetCoverPy import setcover as scp
import copy
import math
import numpy as np

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
  n = len(targets_reachable)
  m = len(floor_validids)
  # U: universe of n elements
  U = targets_reachable
  # E: collection of sets of indices of sets that cover each element
  E = []
  for i in targets_reachable:
    sets_per_element = []
    for j in range(len(floor_validids_per_tar[i])):
      sets_per_element.append(floor_validids.index(floor_validids_per_tar[i][j]))
    E.append(sets_per_element)
  # A: matrix representing the relationship between U and S
  A = np.zeros((n,m))
  for i in range(n):
    for j in E[i]:
      A[i,j] = 1
  # S: collection of sets of elements that are covered by each set
  S = []
  for i in range(m):
    elements_per_set = np.flatnonzero(A[:,i]).tolist()
    S.append(elements_per_set)
  return n, m, E, S


def solver_LPr(n, m, E, maxiters=20):
  """
  Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
  min_x c^T . x
  subject to:
    A_ub . x <= b_ub
    A_eq . x == b_eq
    lb <= x <= ub
  """
  print("  Running 'LPr (Linear Programming relaxation & rounding)' solver...")
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
  return sol, mincost


def solver_SCPy(n, m, E, maxiters=20):
  """
  Based on: https://github.com/guangtunbenzhu/SetCoverPy
  Input: 
    -- a_matrix[nrows, mcols], the binary relationship matrix
       a_matrix[irow, jcol] = True if jcol covers irow
    -- cost[mcols], the cost of columns. 
  """
  print("  Running 'SCPy (SetCoverPy)' solver...")
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
  return sol, mincost


def solver_greedy(n, m, E, maxiters=20):
  """
  Based on: https://github.com/guangtunbenzhu/SetCoverPy
  Input: 
    -- a_matrix[nrows, mcols], the binary relationship matrix
       a_matrix[irow, jcol] = True if jcol covers irow
    -- cost[mcols], the cost of columns. 
  """
  print("  Running 'greedy' solver...")
  c = np.ones(m)
  A = np.zeros((n,m), dtype=bool)
  for i in range(n):
    for j in E[i]:
      A[i,j] = True
  g = scp.SetCover(amatrix=A, cost=c, maxiters=maxiters)
  g.greedy()
  sol = g.s.astype(int)
  mincost = g.total_cost
  return sol, mincost


def solve_geoSCP(targets, targets_reachids, floor_allpoints, floor_validids_per_tar, arm_ori_wrt_base=[0.,0.,0.], \
                 max_phidiff=np.pi/6, solver='SCPy', SCP_maxiters=20, maxiters=200):
  ### Cluster points: solve SCP to find the least number of points on floor to cover all targets
  floor_validids = np.unique(np.concatenate(floor_validids_per_tar)).astype(int).tolist()
  n, m, E, S = get_math_model(floor_validids_per_tar, floor_validids, targets_reachids)
  print("--geoSCP: Solving SCP using '{}' solver...".format(solver))
  print("  * Size of universe: n = {}".format(n))
  print("  * Total number of sets in collection: m = {}".format(m))
  if solver=='SCPy':
    sol, cost = solver_SCPy(n, m, E, SCP_maxiters)
  elif solver=='greedy':
    sol, cost = solver_greedy(n, m, E, SCP_maxiters)
  elif solver=='LPr':
    sol, cost = solver_LPr(n, m, E, SCP_maxiters)
  else:
    raise ValueError("The specified SCP solver is not supported. Valid values: solver='SCPy','greedy','LPr'")
  floor_chosenids = np.array(floor_validids)[np.flatnonzero(sol)]
  floor_chosenpoints = floor_allpoints[floor_chosenids].tolist()
  print("  * chosen set indices = {}".format(floor_chosenids))
  print("  * min total cost = {}".format(cost))

  ### Assign targets into clusters based on orientations
  print("--geoSCP: Assigning targets into clusters...")
  # > get targets' indices & phi for each chosen floor point
  tarids_per_chosenpt = []
  tarphi_per_chosenpt = []
  phidiff_per_chosenpt = []
  targets_reacharray = targets[targets_reachids]
  for i in range(len(floor_chosenids)):
    tarids_per_chosenpt_i = S[np.flatnonzero(sol)[i]]
    tarphi_per_chosenpt_i = np.arctan2( \
      targets_reacharray[tarids_per_chosenpt_i,-2] / np.sqrt(targets_reacharray[tarids_per_chosenpt_i,-3]**2 + \
        targets_reacharray[tarids_per_chosenpt_i,-2]**2), \
      targets_reacharray[tarids_per_chosenpt_i,-3] / np.sqrt(targets_reacharray[tarids_per_chosenpt_i,-3]**2 + \
        targets_reacharray[tarids_per_chosenpt_i,-2]**2) )
    phidiff_per_chosenpt_i = max(tarphi_per_chosenpt_i) - min (tarphi_per_chosenpt_i)
    tarids_per_chosenpt.append(tarids_per_chosenpt_i)
    tarphi_per_chosenpt.append(tarphi_per_chosenpt_i.tolist())
    phidiff_per_chosenpt.append(phidiff_per_chosenpt_i)
  # > assign targets into clusters
  clusters = []
  arm_oris = []
  base_poses = []
  i_check = 0
  i = 0
  count = 0
  length = len(phidiff_per_chosenpt)
  while i < length and i_check < maxiters:
    i_check += 1
    if phidiff_per_chosenpt[i] <= max_phidiff and tarids_per_chosenpt[i] and count < (length-i):
      # add current set into clusters
      cluster_i = tarids_per_chosenpt[i]
      clusters.append(cluster_i)
      phi_cen = np.round( min(tarphi_per_chosenpt[i]) + 0.5*phidiff_per_chosenpt[i], 3)
      base_point_i = np.array(floor_chosenpoints[i]) - utils.z_rotation(arm_ori_wrt_base, phi_cen)[:2]
      arm_oris.append(floor_chosenpoints[i])
      base_poses.append( np.append(base_point_i, phi_cen) )
      # remove the added elements from other sets
      for j in range(len(tarids_per_chosenpt)-i-1):
        for k in range(len(cluster_i)):
          if cluster_i[k] in tarids_per_chosenpt[j+i+1]:
            elementid = tarids_per_chosenpt[j+i+1].index(cluster_i[k])
            tarids_per_chosenpt[j+i+1].pop(elementid)
            tarphi_per_chosenpt[j+i+1].pop(elementid)
        if tarphi_per_chosenpt[j+i+1]:
          phidiff_per_chosenpt[j+i+1] = max(tarphi_per_chosenpt[j+i+1]) - min (tarphi_per_chosenpt[j+i+1])
      i +=  1
      count = 0
    elif count >= (length-i):
      # divide clusters with phidiff > max_phidiff
      div = int( math.ceil(phidiff_per_chosenpt[i]/max_phidiff) )
      minphi = min(tarphi_per_chosenpt[i]) - 0.01
      maxphi = max(tarphi_per_chosenpt[i]) + 0.01
      stepphi = (maxphi-minphi)/div
      for j in range(div):
        new_phi = []
        new_ids = []
        for k in range(len(tarphi_per_chosenpt[i])):
          if minphi+j*stepphi < tarphi_per_chosenpt[i][k] <= minphi+(j+1)*stepphi:
            new_phi.append(tarphi_per_chosenpt[i][k])
            new_ids.append(tarids_per_chosenpt[i][k])
        new_point = floor_chosenpoints[i]
        new_phidiff = max(new_phi) - min(new_phi)
        tarids_per_chosenpt.append(new_ids)
        tarphi_per_chosenpt.append(new_phi)
        phidiff_per_chosenpt.append(new_phidiff)
        floor_chosenpoints.append(new_point)
      print("The cluster associated with point {} has max azimuthal angle difference of {} deg (> {} deg). " \
            .format(np.array(floor_chosenpoints[i]), np.rad2deg(phidiff_per_chosenpt[i]), np.rad2deg(max_phidiff)), \
            "Split it into {} clusters.".format(div) )
      length += div
      i += 1
      count = 0
    elif not tarids_per_chosenpt[i]:
      print("Redundant point found, skipped point {} from SCP results.".format(floor_chosenpoints[i]))
      i +=  1
      count = 0
    else:
      # swap current set with last set, as well as corresponding floor points
      copyset = copy.copy(tarids_per_chosenpt[i])
      tarids_per_chosenpt[i] = tarids_per_chosenpt[-1]
      tarids_per_chosenpt[-1] = copyset
      copyphi = copy.copy(tarphi_per_chosenpt[i])
      tarphi_per_chosenpt[i] = tarphi_per_chosenpt[-1]
      tarphi_per_chosenpt[-1] = copyphi
      copyphidiff = copy.copy(phidiff_per_chosenpt[i])
      phidiff_per_chosenpt[i] = phidiff_per_chosenpt[-1]
      phidiff_per_chosenpt[-1] = copyphidiff
      copypoint = copy.copy(floor_chosenpoints[i])
      floor_chosenpoints[i] = floor_chosenpoints[-1]
      floor_chosenpoints[-1] = copypoint
      count += 1
  # > check maxiters
  if i_check >= maxiters:
    raise ValueError(("geoSCP failed: max iterations ({}) reached during assigning targets into clusters. " \
                      "Please increase 'maxiters' value.").format(maxiters))
  # > check results
  tar_check = 0
  for i in range(len(clusters)):
    tar_check += len(clusters[i])
  if tar_check != len(targets_reachids):
    raise ValueError(("geoSCP failed: number of targets in clusters ({}) does not match " \
                      "number of reachable targets ({}).").format(tar_check, len(targets_reachids)))
  # > finalize results
  clusters = np.array(clusters)
  base_poses = np.array(base_poses)
  arm_oris = np.array(arm_oris)
  print("--geoSCP finished successfully:")
  print("  * number of clusters = {}".format(len(clusters)))
  print("  * clusters of targets = \n{}".format(clusters))
  print("  * corresponding arm's origin at (x[m], y[m]) = \n{}".format(arm_oris))
  print("  * corresponding base poses (x[m], y[m], yaw[rad]) = \n{}".format(base_poses))
  return clusters, arm_oris, base_poses

# END