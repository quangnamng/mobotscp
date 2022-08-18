#!/usr/bin/env python
from __future__ import print_function
from collections import defaultdict
from mobotscp import geoscp, utils
import copy
import itertools
import networkx as nx
import numpy as np
import openravepy as orpy
import progressbar as pbar
import raveutils as ru
import robotsp as rtsp
import time


##############################################################################################################
### The main MoboTSCP solver
##############################################################################################################


class SCPparameters(object):
  def __init__(self, SCP_solver='SCPy', point_maxiters=20, orient_maxiters=100):
    """
    Register SCP parameters.

    Parameters
    ----------
    * SCP_solver: str
      Method used for solving SCP
      - 'SCPy'(default): SetCoverPy method (https://github.com/guangtunbenzhu/SetCoverPy)
      - 'greedy': greedy method
      - 'LPr': Linear Programming relaxation & rounding method
    * SCP_maxiters: int
      Maximum number of iterations for SCP solver
    * cluster_maxiters: int
      Maximum number of iterations for the clustering process 
    """
    self.SCP_solver = SCP_solver
    self.point_maxiters = point_maxiters
    self.orient_maxiters = orient_maxiters



class TSPparameters(object):
  def __init__(self, Thome, qhome, phome):
    """
    Register TSP parameters.

    Parameters
    ----------
    * Thome: 4x4 matrix
      Absolute tramsformation of of the robot (origin at its base) at home position
    * qhome: 9D array
      Home configuration of the robot
    * phome: 3D array
      Absolute position (xyz) of the robot's end-effector at home configuration
    """
    self.Thome = Thome
    self.qhome = qhome
    self.phome = phome



class MoboTSCP(object):
  def __init__(self, robot, targets, floor, reach_param, scp_param, tsp_param):
    """
    MoboTSCP solver.

    Parameters
    ----------
    * robot: orpy.Robot
    * targets: mobotscp.utils.RegisterTargets()
    * floor: mobotscp.utils.RectangularFloor()
    * reach_param: mobotscp.fkreach.FocusedKinematicReachability.calculate_reach_limits()
    * scp_param: mobotscp.solver.SCPparameters()
    * tsp_param: mobotscp.solver.TSPparameters()
    """
    self.robot = robot
    self.env = self.robot.GetEnv()
    self.manip = self.robot.GetActiveManipulator()
    self.targets_array = copy.copy(targets.targets_array)
    self.targets_ray = copy.copy(targets.targets_ray)
    self.floor = copy.deepcopy(floor)
    self.reach_param = copy.deepcopy(reach_param)
    self.scp_param = copy.deepcopy(scp_param)
    self.tsp_param = copy.deepcopy(tsp_param)
    self.output = defaultdict(set)


  def get_base_tour(self, Tbase):
    starttime = time.time()
    base_xyz = [b[:3,3] for b in Tbase]
    base_xyz = np.insert(base_xyz, 0, self.tsp_param.Thome[:3,3], axis=0)
    bgraph = rtsp.construct.from_coordinate_list(base_xyz,distfn=rtsp.metric.euclidean_fn,args=())
    base_tour = rtsp.tsp.two_opt(bgraph)
    base_tour = rtsp.tsp.rotate_tour(base_tour,start=0)[1:]
    base_tour = np.array(base_tour)-1
    btour_time = time.time() - starttime
    print("  * btour_time = {} s".format(btour_time))
    return base_tour, btour_time


  def get_configurations(self, clusters, base_tour, base_poses, Tbase):
    starttime = time.time()
    targets_taskiks = []
    targets_taskiks += [[self.tsp_param.qhome]]
    targets_taskids = []
    with self.env:
      for i in base_tour:
        # Spawn robot in a base pose
        self.robot.SetTransform(Tbase[i])
        # Find IK solutions for each target in each the cluster
        for j in clusters[i]:
          # > end-effector is at self.tsp_param.standoff [m] from the target
          tar_ray = self.targets_ray[j]
          ee_pos = tar_ray.pos() - self.tsp_param.standoff*tar_ray.dir()
          ee_ray = orpy.Ray(ee_pos, tar_ray.dir())
          # > find arm config solution
          ik_solutions = ru.kinematics.find_ik_solutions(self.robot, ee_ray, self.tsp_param.iktype, \
                                                         collision_free=True, freeinc=self.tsp_param.step_size)
          if len(ik_solutions) == 0:
            raise Exception('Failed to find IK solution for target {}'.format(j))
          # > add base pose into the arm solutions
          for s,q in enumerate(ik_solutions):
             ik_solutions[s] = np.insert(q, 6, base_poses[i])
          targets_taskiks.append(ik_solutions)
          # > targets_taskids
          targets_taskids.append(j)
      self.robot.SetTransform(self.tsp_param.Thome)
    ik_time = time.time() - starttime
    print("  * ik_time = {} s".format(ik_time))
    return targets_taskiks, targets_taskids, ik_time


  def stack_cluster(self, clusters, Tbase, base_tour):
    def get_stacked_xyz(Tbase, s, target_idxs, targets):
      R_o_base = Tbase[:3,:3]
      # R_base_o rotates base frame to align with coordinate frame
      R_base_o = np.linalg.inv(R_o_base)
      # T to rotate base frame until the axes directions align with coordinate frame
      T = np.eye(4)
      T[:3,:3] = R_base_o
      T_o_base_ = np.dot(T,Tbase)
      # p_displace will be used to displaced all target points as if the base is placed at the origin
      p_displace = T_o_base_[:3,3]
      stargets = []
      for idx in target_idxs:
        p = targets[idx].pos()
        # > rotate p similarly to base
        p_ = np.dot(R_base_o, p)
        # > displace as if the base will be in coordinate frame
        p_ -= p_displace
        # > add offset along x-axis
        p_ += np.array([s*self.tsp_param.stack_offset,0,0])
        stargets += [p_]
      return stargets
    # Get stack-of-clusters
    stargets_xyz = []
    phome_ray = [orpy.Ray(self.tsp_param.phome, np.ones(3))]
    stargets_xyz += get_stacked_xyz(self.tsp_param.Thome, 0, np.array([0]), phome_ray)
    for i,s in zip(base_tour,range(1,len(base_tour)+1)):
      stargets_xyz += get_stacked_xyz(Tbase[i], s, clusters[i], self.targets_ray)
    stargets_xyz += get_stacked_xyz(self.tsp_param.Thome, len(base_tour)+1, np.array([0]), phome_ray)
    return stargets_xyz


  def get_task_tour(self, clusters, Tbase, base_tour):
    starttime = time.time()
    stargets_xyz = self.stack_cluster(clusters, Tbase, base_tour)
    tgraph = rtsp.construct.from_coordinate_list(stargets_xyz, distfn=rtsp.metric.euclidean_fn, args=())
    tgraph.add_edge(0, len(stargets_xyz)-1, weight=0)
    task_tour = rtsp.tsp.two_opt(tgraph)
    task_tour = rtsp.tsp.rotate_tour(task_tour, start=0)
    ttour_time = time.time() - starttime
    print("  * ttour_time = {} s".format(ttour_time))
    return task_tour, ttour_time


  def from_sorted_setslist(self, setslist, distfn, args=(), verbose=False):
    if distfn is None:
      distfn = rtsp.metric.euclidean_fn
    set_sizes = [len(s) for s in setslist]
    num_sets = len(setslist)
    graph = nx.Graph()
    # Generate the list of nodes ids
    num_edges = 0
    start = 0
    sets = []
    for i,size in enumerate(set_sizes):
      stop = start+size
      sets.append(range(start, stop))
      start = stop
      if i < len(set_sizes)-1:
        num_edges += set_sizes[i]*set_sizes[i+1]
    # Configure the status bar
    if verbose:
      widgets = ['Populating graph edges: ', pbar.SimpleProgress()]
      widgets += [' ', pbar.Bar(), ' ', pbar.Timer()]
      bar = pbar.ProgressBar(widgets=widgets, maxval=num_edges).start()
      count = 0
    # Add nodes and edges
    for i in range(num_sets-1):
      j = i+1
      set_i_indices = range(set_sizes[i])
      set_j_indices = range(set_sizes[j])
      for k,l in itertools.product(set_i_indices, set_j_indices):
        if verbose:
          bar.update(count)
          count += 1
        x = setslist[i][k]
        y = setslist[j][l]
        u = sets[i][k]
        v = sets[j][l]
        graph.add_node(u, value=x)
        graph.add_node(v, value=y)
        graph.add_edge(u, v, weight=distfn(x[:6], y[:6], *args))
    if verbose:
      bar.finish()
    return graph, sets


  def get_configuration_tour(self, targets_taskiks, task_tour):
    starttime = time.time()
    # Create sorted configurations setslist
    setslist = []
    for n in task_tour[:-1]:
      setslist.append(targets_taskiks[n])
    setslist += [[self.tsp_param.qhome]]
    cgraph, bins = self.from_sorted_setslist(setslist, distfn=self.tsp_param.cspace_metric, \
                                             args=self.tsp_param.cspace_metric_args)
    # Get configuration tour
    ctour = nx.dijkstra_path(cgraph, source=0, target=cgraph.number_of_nodes()-1)
    ctour_time = time.time() - starttime
    print("  * ctour_time = {} s".format(ctour_time))
    return cgraph, ctour, ctour_time


  def compute_cspace_trajectories(self, robot, cgraph, ctour, params):
    trajectories = []
    for idx in range(len(ctour)-1):
      u = ctour[idx]
      v = ctour[idx+1]
      qstart = cgraph.node[u]['value']
      qgoal = cgraph.node[v]['value']
      with robot:
        robot.SetActiveDOFValues(qstart)
        traj = ru.planning.plan_to_joint_configuration(robot, qgoal, params.planner, params.max_iters, \
                                                       params.max_ppiters, try_swap=params.try_swap)
        if traj is None:
          raise Exception("Failed to compute trajectories, please try lower 'tsp_param.step_size'.")
      trajectories.append(traj)
    return trajectories


  def get_trajectories(self, cgraph, ctour, retimer=None):
    starttime = time.time()
    # Find trajectories 
    trajectories = self.compute_cspace_trajectories(self.robot, cgraph, ctour, self.tsp_param)
    # Retime trajectories to satisfy velocity and acceleration limits
    # > 'parabolicsmoother'
    if retimer == 'parabolicsmoother':
      i = 0
      for traj in trajectories:
        status = orpy.planningutils.SmoothAffineTrajectory(traj, maxvelocities=self.tsp_param.velocity_limits, \
                                                           maxaccelerations=self.tsp_param.acceleration_limits, \
                                                           plannername='parabolicsmoother')
        if status != orpy.PlannerStatus.HasSolution:
          raise Exception("  * parabolicsmoother failed for trajectory {}.".format(i))
        i += 1
    # > 'trapezoidalretimer'
    if retimer == 'trapezoidalretimer':
      retimed_trajs = []
      for traj in trajectories:
        trapezoidalretimer = utils.RetimeOpenraveTrajectory(self.robot, traj, timestep=self.tsp_param.timestep, \
                                                            Vmax=self.tsp_param.velocity_limits, \
                                                            Amax=self.tsp_param.acceleration_limits)
        retimed_traj, success = trapezoidalretimer.retime()
        if not success:
          raise Exception("  * trapezoidalretimer failed for trajectory {}.".format(len(retimed_trajs)))
        retimed_trajs.append(retimed_traj)
      trajectories = retimed_trajs
    # Results
    traj_time =  time.time() - starttime
    print("  * traj_time = {} s".format(traj_time))
    return trajectories, traj_time


  def prepare_output(self, targets_reachids, targets_unreachids, clusters, base_poses, scp_time, \
                      base_tour, btour_time, targets_taskiks, targets_taskids, ik_time, \
                      task_tour, ttour_time, cgraph, config_tour, ctour_time, trajs, traj_time, tsp_time):
    self.output["targets_reachids"] = targets_reachids
    self.output["targets_unreachids"] = targets_unreachids
    self.output["clusters"] = clusters
    self.output["clusters_no"] = len(clusters)
    self.output["base_poses"] = base_poses
    self.output["scp_time"] = scp_time
    self.output["base_tour"] = base_tour
    self.output["btour_time"] = btour_time
    self.output["target_taskiks"] = targets_taskiks
    self.output["target_taskids"] = targets_taskids
    self.output["ik_time"] = ik_time
    self.output["task_tour"] = task_tour
    self.output["ttour_time"] = ttour_time
    self.output["cgraph"] = cgraph
    self.output["config_tour"] = config_tour
    self.output["ctour_time"] = ctour_time
    self.output["trajs"] = trajs
    self.output["traj_time"] = traj_time
    self.output["tsp_time"] = tsp_time
    self.output["mobotscp_time"] = scp_time + tsp_time


  def solve(self):
    # geoSCP: cluster all reachable targets into clusters
    starttime = time.time()
    # > connect targets to floor points
    tar2floor = geoscp.ConnectTargets2Floor(self.targets_array, self.floor, self.reach_param)
    floor_validids_per_tar, targets_reachids, targets_unreachids = tar2floor.connect()
    # > assign all reachable targets into clusters
    clusters, arm_oris, base_poses, Tbase = \
      geoscp.solve_geoSCP(self.targets_array, targets_reachids, self.floor, floor_validids_per_tar, \
                          self.reach_param.arm_ori_wrt_base, self.reach_param.max_phidiff, \
                          self.scp_param.SCP_solver, self.scp_param.point_maxiters, self.scp_param.orient_maxiters)
    scp_time = time.time() - starttime
    print("  * scp_time = {} s".format(scp_time))

    # stackTSP: find optimal sequence to visit all targets
    starttime = time.time()
    print("--stackTSP: Solving sequencing...")
    # > find base tour
    base_tour, btour_time = self.get_base_tour(Tbase)
    # > find IK solution for targets in each cluster
    targets_taskiks, targets_taskids, ik_time = self.get_configurations(clusters, base_tour, base_poses, Tbase)
    # > find task-space tour
    task_tour, ttour_time = self.get_task_tour(clusters, Tbase, base_tour)
    # > find configuration-space tour
    cgraph, config_tour, ctour_time = self.get_configuration_tour(targets_taskiks, task_tour)
    # > compute trajectories
    trajs, traj_time = self.get_trajectories(cgraph, config_tour, self.tsp_param.retimer)
    # > record time used
    tsp_time = time.time() - starttime
    print("--stackTSP finished successfully.")
    print("  * tsp_time = {} s".format(tsp_time))

    # Results
    self.prepare_output(targets_reachids, targets_unreachids, clusters, base_poses, scp_time, \
                        base_tour, btour_time, targets_taskiks, targets_taskids, ik_time, \
                        task_tour, ttour_time, cgraph, config_tour, ctour_time, trajs, traj_time, tsp_time)
    print("--MoboTSCP solver finished successfully.")
    print("  * mobotscp_time = {} s".format(self.output["mobotscp_time"]))
    return self.output