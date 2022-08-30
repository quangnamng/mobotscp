#!/usr/bin/env python
from __future__ import print_function
from mobotscp import utils
from scipy.optimize import linprog
from SetCoverPy import setcover as scp
import copy
import math
import numpy as np
import tf.transformations as tr

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


class ConnectTargets2Floor(object):
  """
  Connect each target with valid points on the floor that can reach the target.

  Parameters
  ----------
  * alltargets: (array_like) array of orpy.Ray elements representing 5D targets
  * floor grid: X, Y = numpy.mgrid[xmin:xmax, ymin:ymax]
  * floor.allpoints = numpy.c_[X.flat, Y.flat] * gridsize
  * reach_param (fkreach.FocusedKinematicReachability.calculate_reach_limits)

  Returns
  -------
  * floor_validinds_i = numpy.flatnonzero(conditions): indices of floor's points that can reach target i
  * floor_validids_per_tar.append(floor_validids_i): sets of floor's valid indices for all targets
  * targets_reachable: lists of targets having at least 1 floor's valid index
  * targets_unreachable: lists of targets having 0 floor's valid index
  """
  def __init__(self, targets_array, floor, reach_param):
    # targets
    self.targets = copy.copy(targets_array)
    self.targets_seedir = [ self.targets[:,-3]/np.sqrt(self.targets[:,-3]**2+self.targets[:,-2]**2), \
                            self.targets[:,-2]/np.sqrt(self.targets[:,-3]**2+self.targets[:,-2]**2) ]
    self.targets_phi = np.arctan2( self.targets_seedir[1], self.targets_seedir[0] )
    # floor
    self.floor_allpoints = copy.copy(floor.floor_allpoints)
    self.floor_z = floor.floor_z
    # reach parameters
    self.Xmin_wrt_arm = reach_param.Xmin_wrt_arm
    self.Zmin_wrt_arm = reach_param.Zmin_wrt_arm
    self.Zmax_wrt_arm = reach_param.Zmax_wrt_arm
    self.spheres_center_wrt_arm = np.array(reach_param.spheres_center_wrt_arm)
    self.arm_ori_wrt_base = np.array(reach_param.arm_ori_wrt_base)
    self.Rmin = reach_param.Rmin
    self.Rmax = reach_param.Rmax

  def connect(self):
    print("--geoSCP: Connecting each target with valid points on the floor...")
    # Sets of floor's valid indices
    floor_validids_per_tar = []
    targets_reachable = []
    targets_unreachable = []
    for i in range(len(self.targets)):
      spheres_center_wrt_floor = self.targets[i,:2] - utils.z_rotation(self.spheres_center_wrt_arm, \
                                                                       self.targets_phi[i])[:2]
      z_tar_wrt_z_arm = self.targets[i,2] - self.arm_ori_wrt_base[2] - self.floor_z
      z_tar_wrt_z_center = z_tar_wrt_z_arm - self.spheres_center_wrt_arm[2]
      rmin2 = self.Rmin**2 - z_tar_wrt_z_center**2
      rmax2 = self.Rmax**2 - z_tar_wrt_z_center**2
      floor_validids_cond1 = np.flatnonzero( \
        np.sum((self.floor_allpoints-spheres_center_wrt_floor)**2, 1) <= rmax2 )
      floor_validids_cond2 = np.flatnonzero( \
        np.sum((self.floor_allpoints[floor_validids_cond1]-spheres_center_wrt_floor)**2, 1) >= rmin2 )
      floor_validids_cond12 = floor_validids_cond1[floor_validids_cond2]
      r_tar_to_pt = self.targets[i,:2]-self.floor_allpoints[floor_validids_cond12]
      floor_validids_cond3 = np.flatnonzero( \
        ( r_tar_to_pt[:,0]*self.targets_seedir[0][i] + \
          r_tar_to_pt[:,1]*self.targets_seedir[1][i] ) >= self.Xmin_wrt_arm )
      floor_validids_i = floor_validids_cond12[floor_validids_cond3]
      if len(floor_validids_i)==0 or z_tar_wrt_z_arm>self.Zmax_wrt_arm or z_tar_wrt_z_arm<self.Zmin_wrt_arm:
        targets_unreachable.append(i)
        floor_validids_per_tar.append([])
      else:
        targets_reachable.append(i)
        floor_validids_per_tar.append(floor_validids_i)
    print("  * Number of targets reachable = {}/{}".format(len(targets_reachable), len(self.targets)))
    print("  * Number of targets unreachable = {}/{}".format(len(targets_unreachable), len(self.targets)))
    return floor_validids_per_tar, targets_reachable, targets_unreachable


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
  * a_matrix[nrows, mcols], the binary relationship matrix
    a_matrix[irow, jcol] = True if jcol covers irow
  * cost[mcols], the cost of columns. 
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
  * a_matrix[nrows, mcols], the binary relationship matrix
    a_matrix[irow, jcol] = True if jcol covers irow
  * cost[mcols], the cost of columns. 
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


def calculate_phidiff_phicen(target_phi):
  # sort
  phi = copy.copy(target_phi)
  phi.sort()
  # calculate phidiff
  deltaphi = []
  for i in range(len(phi)-1):
    deltaphi += [phi[i+1] - phi[i]]
  deltaphi += [phi[0] + 2*np.pi - phi[-1]]
  phidiff = 2*np.pi - max(deltaphi)
  # calculate phicen
  ids = np.argmax(deltaphi)
  if ids < len(phi)-1:
    phi_lower = phi[ids+1]
  else:
    phi_lower = phi[0]
  phicen = phi_lower + 0.5*phidiff
  if phicen > np.pi:
    phicen -= 2*np.pi
  elif phicen <= -np.pi:
    phicen += 2*np.pi
  return phidiff, phicen


def solve_geoSCP(targets_array, targets_reachids, floor, floor_validids_per_tar, arm_ori_wrt_base=[0.,0.,0.], \
                 max_phidiff=np.pi/6, solver='SCPy', point_maxiters=20, orient_maxiters=200):
  ### Cluster points: solve SCP to find the least number of points on floor to cover all targets
  floor_validids = np.unique(np.concatenate(floor_validids_per_tar)).astype(int).tolist()
  n, m, E, S = get_math_model(floor_validids_per_tar, floor_validids, targets_reachids)
  print("--geoSCP: Solving SCP using '{}' solver...".format(solver))
  print("  * Size of universe: n = {}".format(n))
  print("  * Total number of sets in collection: m = {}".format(m))
  if solver=='SCPy':
    sol, cost = solver_SCPy(n, m, E, point_maxiters)
  elif solver=='greedy':
    sol, cost = solver_greedy(n, m, E, point_maxiters)
  elif solver=='LPr':
    sol, cost = solver_LPr(n, m, E, point_maxiters)
  else:
    raise ValueError("The specified SCP solver is not supported. Valid values: solver='SCPy','greedy','LPr'")
  floor_chosenids = np.array(floor_validids)[np.flatnonzero(sol)]
  floor_chosenpoints = floor.floor_allpoints[floor_chosenids].tolist()
  print("  * chosen set indices = {}".format(floor_chosenids))
  print("  * min total cost = {}".format(cost))

  ### Assign targets into clusters based on orientations
  print("--geoSCP: Assigning targets into clusters...")
  # > get targets' indices & phi for each chosen floor point
  tarids_per_chosenpt = []
  tarphi_per_chosenpt = []
  phidiff_per_chosenpt = []
  phicen_per_chosenpt = []
  targets_reacharray = targets_array[targets_reachids]
  for i in range(len(floor_chosenids)):
    tarids_per_chosenpt_i = S[np.flatnonzero(sol)[i]]
    tarphi_per_chosenpt_i = np.arctan2( \
      targets_reacharray[tarids_per_chosenpt_i,-2] / np.sqrt(targets_reacharray[tarids_per_chosenpt_i,-3]**2 + \
        targets_reacharray[tarids_per_chosenpt_i,-2]**2), \
      targets_reacharray[tarids_per_chosenpt_i,-3] / np.sqrt(targets_reacharray[tarids_per_chosenpt_i,-3]**2 + \
        targets_reacharray[tarids_per_chosenpt_i,-2]**2) \
      ).tolist()
    tarids_per_chosenpt += [tarids_per_chosenpt_i]
    tarphi_per_chosenpt += [tarphi_per_chosenpt_i]
    # calculate max phi difference for each set
    phidiff_per_chosenpt_i, phicen_per_chosenpt_i = calculate_phidiff_phicen(tarphi_per_chosenpt_i)
    phidiff_per_chosenpt += [phidiff_per_chosenpt_i]
    phicen_per_chosenpt += [phicen_per_chosenpt_i]
  # > assign targets into clusters
  clusters = []
  arm_oris = []
  base_poses = []
  i_check = 0
  i = 0
  count = 0
  length = len(phidiff_per_chosenpt)
  while i < length and i_check < orient_maxiters:
    i_check += 1
    if phidiff_per_chosenpt[i] <= max_phidiff and tarids_per_chosenpt[i] and count < (length-i):
      # add current set into clusters
      cluster_i = tarids_per_chosenpt[i]
      clusters.append(cluster_i)
      phi_cen = phicen_per_chosenpt[i]
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
          phidiff_per_chosenpt[j+i+1], phicen_per_chosenpt[j+i+1] = calculate_phidiff_phicen(tarphi_per_chosenpt[j+i+1])
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
        if len(new_phi)==0:
          div -= 1
        else:
          tarids_per_chosenpt += [new_ids]
          tarphi_per_chosenpt += [new_phi]
          phidiff_per_chosenpt_i, phicen_per_chosenpt_i = calculate_phidiff_phicen(new_phi)
          phidiff_per_chosenpt += [phidiff_per_chosenpt_i]
          phicen_per_chosenpt += [phicen_per_chosenpt_i]
          floor_chosenpoints += [floor_chosenpoints[i]]
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
      tarids_per_chosenpt[i], tarids_per_chosenpt[-1] = tarids_per_chosenpt[-1], tarids_per_chosenpt[i]
      tarphi_per_chosenpt[i], tarphi_per_chosenpt[-1] = tarphi_per_chosenpt[-1], tarphi_per_chosenpt[i]
      phidiff_per_chosenpt[i], phidiff_per_chosenpt[-1] = phidiff_per_chosenpt[-1], phidiff_per_chosenpt[i]
      phicen_per_chosenpt[i], phicen_per_chosenpt[-1] = phicen_per_chosenpt[-1], phicen_per_chosenpt[i]
      floor_chosenpoints[i], floor_chosenpoints[-1] = floor_chosenpoints[-1], floor_chosenpoints[i]
      count += 1
  # > check maxiters
  if i_check >= orient_maxiters:
    raise ValueError(("geoSCP failed: max iterations ({}) reached during assigning targets into clusters. " \
                      "Please increase 'maxiters' value.").format(orient_maxiters))

  ### Results
  # > check results
  tar_check = 0
  for i in range(len(clusters)):
    tar_check += len(clusters[i])
  if tar_check != len(targets_reachids):
    raise ValueError(("geoSCP failed: number of targets in clusters ({}) does not match " \
                      "number of reachable targets ({}).").format(tar_check, len(targets_reachids)))
  # > calculate base transform
  Tbase = []
  for i in range(len(base_poses)):
    T = tr.euler_matrix(0, 0, base_poses[i][2], 'sxyz')
    T[:3,3] = np.array(list(base_poses[i][:2])+[floor.floor_z])
    Tbase.append(T)
  # > finalize results
  clusters = np.array(clusters)
  base_poses = np.array(base_poses)
  arm_oris = np.array(arm_oris)
  print("  * number of clusters = {}".format(len(clusters)))
  print("  * clusters of targets = \n{}".format(clusters))
  print("  * corresponding arm's origin at (x[m], y[m]) = \n{}".format(arm_oris))
  print("  * corresponding base poses (x[m], y[m], yaw[rad]) = \n{}".format(base_poses))
  print("--geoSCP finished successfully.")
  return clusters, arm_oris, base_poses, Tbase

# END