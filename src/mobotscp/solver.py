#!/usr/bin/env python
from __future__ import print_function
from mobotscp import connection, geoscp
from collections import defaultdict
import copy
import numpy as np
import robotsp as rtsp
import time

##############################################################################################################
### The main MoboTSCP solver
##############################################################################################################


class SCPparameters(object):
  def __init__(self, SCP_solver='SCPy', SCP_maxiters=20, cluster_maxiters=100):
    """
    SCP_solver (str): method used for solving SCP
      - 'SCPy'(default): SetCoverPy method (https://github.com/guangtunbenzhu/SetCoverPy)
      - 'greedy': greedy method
      - 'LPr': Linear Programming relaxation & rounding method
    SCP_maxiters (int): maximum number of iterations for SCP solver
    cluster_maxiters (int): maximum number of iterations for the clustering process 
    """
    self.SCP_solver = SCP_solver
    self.SCP_maxiters = SCP_maxiters
    self.cluster_maxiters = cluster_maxiters


class TSPparameters(object):
  def __init__(self, Thome, qhome, phome):
    """
    Register TSP parameters
    """
    self.Thome = Thome
    self.qhome = qhome
    self.phome = phome


class MoboTSCP(object):
  def __init__(self, robot, targets_array, floor, reach_param, scp_param, tsp_param):
    self.robot = robot
    self.env = self.robot.GetEnv()
    self.manip = self.robot.GetActiveManipulator()
    self.targets = copy.copy(targets_array)
    self.floor = copy.deepcopy(floor)
    self.reach_param = copy.deepcopy(reach_param)
    self.scp_param = copy.deepcopy(scp_param)
    self.tsp_param = copy.deepcopy(tsp_param)
    self.output = defaultdict(set)

  def get_base_tour(self, Tbase):
    base_xyz = [b[:3,3] for b in Tbase]
    base_xyz = np.insert(base_xyz, 0, self.tsp_param.Thome[:3,3], axis=0)
    bgraph = rtsp.construct.from_coordinate_list(base_xyz,distfn=rtsp.metric.euclidean_fn,args=())
    base_tour = rtsp.tsp.two_opt(bgraph)
    base_tour = rtsp.tsp.rotate_tour(base_tour,start=0)[1:]
    base_tour = np.array(base_tour)-1
    return base_tour

  def prepare_output(self, targets_reachids, targets_unreachids, clusters, cluster_time, arm_oris, base_poses, base_tour):
    self.output["targets_reachids"] = targets_reachids
    self.output["targets_unreachids"] = targets_unreachids
    self.output["clusters"] = clusters
    self.output["clusters_no"] = len(clusters)
    self.output["cluster_time"] = cluster_time
    self.output["arm_oris"] = arm_oris
    self.output["base_poses"] = base_poses
    self.output["base_tour"] = base_tour
    # self.output["cgraph"] = cgraph
    # self.output["hole_tour"] = hole_tour
    # self.output["ctour"] = ctour
    # self.output["trajs"] = trajs
    # self.output["htour_cpu_time"] = htour_cpu_time
    # self.output["dijkstra_cpu_time"] = dijkstra_cpu_time
    # self.output["solver_cpu_time"] = solver_cpu_time
    # self.output["traj_cpu_time"] = traj_cpu_time

  def solve(self):
    # Connect targets to floor points
    tar2floor = connection.ConnectTargets2Floor(self.targets, self.floor, self.reach_param)
    floor_validids_per_tar, targets_reachids, targets_unreachids = tar2floor.connect()

    # SCP: cluster all reachable targets into clusters 
    starttime = time.time()
    clusters, arm_oris, base_poses, Tbase = \
      geoscp.solve_geoSCP(self.targets, targets_reachids, self.floor, floor_validids_per_tar, \
                          self.reach_param.arm_ori_wrt_base, self.reach_param.max_phidiff, \
                          self.scp_param.SCP_solver, self.scp_param.SCP_maxiters, self.scp_param.cluster_maxiters)
    cluster_time = time.time() - starttime

    # TSP:
    # > find base tour
    base_tour = self.get_base_tour(Tbase)
    # TODO: > find IK solution for targets in each cluster


    self.prepare_output(targets_reachids, targets_unreachids, clusters, cluster_time, arm_oris, base_poses, base_tour)
    return self.output