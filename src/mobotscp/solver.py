#!/usr/bin/env python
from __future__ import print_function
from mobotscp import connection, geoscp
from collections import defaultdict
import copy
import numpy as np
import time

##############################################################################################################
### The main MoboTSCP solver
##############################################################################################################


class MoboTSCP(object):
  def __init__(self, targets_array, floor_allpoints, reach_param):
    self.targets = copy.copy(targets_array)
    self.floor_allpoints = copy.copy(floor_allpoints)
    self.reach_param = copy.deepcopy(reach_param)
    self.output = defaultdict(set)

  def prepare_output(self, targets_reachids, targets_unreachids, clusters, arm_oris, base_poses, cluster_time):
    self.output["targets_reachids"] = targets_reachids
    self.output["targets_unreachids"] = targets_unreachids
    self.output["clusters"] = clusters
    self.output["arm_oris"] = arm_oris
    self.output["base_poses"] = base_poses
    self.output["clusters_no"] = len(clusters)
    self.output["cluster_time"] = cluster_time
    # self.output["cgraph"] = cgraph
    # self.output["hole_tour"] = hole_tour
    # self.output["ctour"] = ctour
    # self.output["trajs"] = trajs
    # self.outv["htour_cpu_time"] = htour_cpu_time
    # self.output["dijkstra_cpu_time"] = dijkstra_cpu_time
    # self.output["solver_cpu_time"] = solver_cpu_time
    # self.output["traj_cpu_time"] = traj_cpu_time

  def solve(self, SCP_solver='SCPy', SCP_maxiters=20, maxiters=200):
    # Connect targets to floor points
    tar2floor = connection.ConnectTargets2Floor(self.targets, self.floor_allpoints, self.reach_param)
    floor_validids_per_tar, targets_reachids, targets_unreachids = tar2floor.connect()

    # Cluster all targets into clusters 
    starttime = time.time()
    clusters, arm_oris, base_poses = geoscp.solve_geoSCP(self.targets, targets_reachids, self.floor_allpoints, \
                                               floor_validids_per_tar, self.reach_param.arm_ori_wrt_base, \
                                               self.reach_param.max_phidiff, SCP_solver, SCP_maxiters, maxiters)
    cluster_time = time.time() - starttime

    self.prepare_output(targets_reachids, targets_unreachids, clusters, arm_oris, base_poses, cluster_time)
    return self.output