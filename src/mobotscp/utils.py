#!/usr/bin/env python
import copy
import math
import numpy as np
import openravepy as orpy
import raveutils as ru
import tf.transformations as tr

##############################################################################################################
# Useful utility functions 
##############################################################################################################


def z_rotation(xyz, yaw_rad):
  rot_matrix = [ [np.cos(yaw_rad), -np.sin(yaw_rad), 0.],
                 [np.sin(yaw_rad), np.cos(yaw_rad),  0.],
                 [0.,              0.,               1.] ]
  rotated_xyz = np.matmul(rot_matrix, np.transpose(xyz))
  return np.transpose(rotated_xyz)


class RegisterTargets(object):
  def __init__(self, links, targets_name='hole', max_targets=100, add_azimuth=[0]*100):
    """
    Register targets into appropriate formats:

    Parameters
    ----------
    * links: list of orpy.Link
      The links of the objects that contain the targets
    * targets_name: str
      The common part of the names of the targets
    * max_targets: int
      Maximum number of targets
    * add_azimuth: list
      List of azimuthaal angles added to the targets, must be of size max_targets*1

    Returns
    -------
    * targets_ray: list of orpy.Ray
      List of target Rays for OpenRAVE usage
    * targets_array: 2D numpy.array
      Array of target arrays for computation usage
    """
    targets_ray = []
    targets_array = []
    i = 0
    for link in links:
      lname = link.GetName()
      if lname.startswith(targets_name) and i < max_targets:
        transform = self.add_azimuth_to_targets(link.GetTransform(), add_azimuth[i])
        targets_ray.append( ru.conversions.to_ray(transform) )
        targets_array.append( self.to_array(transform) )
        i += 1
    targets_array = np.vstack(targets_array)
    self.targets_ray = targets_ray
    self.targets_array = targets_array

  def to_array(self, transform):
    """
    Convert a homogeneous transformation into an `numpy.array`.

    Parameters
    ----------
    * transform:  array_like
      The input homogeneous transformation

    Returns
    -------
    * array: numpy.array
      The resulting array
    """
    array = np.ravel( [transform[:3,3], transform[:3,2]] )
    return array

  def add_azimuth_to_targets(self, transform, yaw):
    new_tran = copy.copy(transform)
    orient = np.array(transform[:3,2])
    new_orient = z_rotation(orient, yaw)
    new_tran[:3,2] = new_orient
    return new_tran


class RectangularFloor(object):
  def __init__(self, floor_gridsize=0.1, floor_xrange=[-1., 0.], floor_yrange=[-1., 1.], floor_z=0.):
    """
    Discretize the floor into points in a rectangular region.

    Parameters
    ----------
    * floor_gridsize:  float
      The grid size of the discretized floor
    * floor_xrange:  [float, float]
      [x_min, x_max] of the rectangular region
    * floor_yrange:  [float, float]
      [y_min, y_max] of the rectangular region
    * floor_z:  float
      The height of the floor w.r.t. world frame

    Returns
    -------
    * self.floor_allpoints: 2D numpy.array
      The resulting 2D array of discrete points inside the specified region on the floor
    * self.floor_z:  float
      The height of the floor w.r.t. world frame
    """
    [floor_Xmin, floor_Xmax] = np.array(floor_xrange)//floor_gridsize
    [floor_Ymin, floor_Ymax] = np.array(floor_yrange)//floor_gridsize
    X, Y = np.mgrid[floor_Xmin:floor_Xmax, floor_Ymin:floor_Ymax]
    self.floor_allpoints = np.c_[X.flat, Y.flat] * floor_gridsize
    self.floor_z = floor_z


class VisualizeSolution(object):
  def __init__(self, targets, clusters, base_tour):
    self.colors = [] 
    self.colors += [np.array([255,215,0])/255.]     # gold
    self.colors += [np.array([0,139,139])/255.]     # dark cyan
    self.colors += [np.array([173,255,47])/255.]    # green yellow
    self.colors += [np.array([147,112,219])/255.]   # medium purple
    self.colors += [np.array([245,222,179])/255.]   # wheat
    self.colors += [np.array([144,238,144])/255.]   # light green
    self.colors += [np.array([219,112,147])/255.]   # pale violet red
    self.colors += [np.array([139,69,19])/255.]     # saddle brown
    self.colors += [np.array([255,140,0])/255.]     # dark orange
    self.colors += [np.array([255,255,255])/255.]   # white
    self.arrows = []
    self.points = []
    self.poses = []
    self.axes = []
    self.tour = []
    self.targets_ray = targets.targets_ray
    self.targets_array = targets.targets_array
    self.clusters = clusters
    self.base_tour = base_tour

  def visualize_clusters(self, env, draw_arrows=False, arrow_len=0.07):
    for k in range(len(self.clusters)):
      i = self.base_tour[k]
      for j in range(len(self.clusters[i])):
        # > draw points
        targets_xyz = self.targets_array[self.clusters[i][j]][:3]
        self.points.append( ru.visual.draw_point(env=env, point=targets_xyz, size=5, \
                                                 color=self.colors[i%len(self.colors)]) )
        # > draw arrows on targets
        if draw_arrows and arrow_len>0:
          tar_ray = self.targets_ray[self.clusters[i][j]]
          tar_ray = orpy.Ray(tar_ray.pos()-arrow_len*tar_ray.dir(), tar_ray.dir())
          self.arrows.append( ru.visual.draw_ray(env=env, ray=tar_ray, dist=arrow_len, linewidth=0., \
                                                 color=self.colors[i%len(self.colors)]) )

  def visualize_base_tour(self, env, base_poses, base_home, floor_z):
    base_xyz_home = np.array(list(base_home[:2])+[floor_z])
    for k in range(len(self.base_tour)):
      i = self.base_tour[k]
      # > draw points at the base poses
      base_xyz = np.array(list(base_poses[i][:2])+[floor_z])
      self.poses.append( ru.visual.draw_point(env=env, point=base_xyz, size=20, \
                                              color=self.colors[i%len(self.colors)]) )
      # > draw axes at the base poses
      base_trans = tr.euler_matrix(0, 0, base_poses[i][2], 'sxyz')
      base_trans[:3,3] = base_xyz
      self.axes.append( ru.visual.draw_axes(env=env, transform=base_trans, dist=0.2, linewidth=4) )
      # > draw arrows representing base tour
      if k == 0:                       # from home to first point
        tour_len = np.linalg.norm(base_xyz-base_xyz_home)
        tour_dir = (base_xyz-base_xyz_home)/tour_len
        tour_ray = orpy.Ray(base_xyz_home, tour_dir)
        self.tour.append( ru.visual.draw_ray(env=env, ray=tour_ray, dist=tour_len, linewidth=2, \
                                             color=self.colors[i%len(self.colors)]) )
      if k < len(self.base_tour)-1:    # from current point to next point
        base_xyz_next = np.array(list(base_poses[self.base_tour[k+1]][:2])+[floor_z])
        tour_len = np.linalg.norm(base_xyz_next-base_xyz)
        tour_dir = (base_xyz_next-base_xyz)/tour_len
        tour_ray = orpy.Ray(base_xyz, tour_dir)
        self.tour.append( ru.visual.draw_ray(env=env, ray=tour_ray, dist=tour_len, linewidth=2, \
                                             color=self.colors[i%len(self.colors)]) )
      elif k == len(self.base_tour)-1: # from last poin back home
        tour_len = np.linalg.norm(base_xyz_home-base_xyz)
        tour_dir = (base_xyz_home-base_xyz)/tour_len
        tour_ray = orpy.Ray(base_xyz, tour_dir)
        self.tour.append( ru.visual.draw_ray(env=env, ray=tour_ray, dist=tour_len, linewidth=2, \
                                             color=self.colors[i%len(self.colors)]) )


class RetimeOpenraveTrajectory(object):
  '''
  Time-parameterize straight-line OpenRAVE trajectories.
  @type   traj: orpy.Trajectory
  @param  traj: The input OpenRAVE trajectory
  @type   timestep: float
  @param  timestep: time interval between 2 successive waypoints, should be set at 1./{controller rate}
  @type   Vmax: list
  @param  Vmax: velocity constraints of all active DOFs
  @type   Amax: list
  @param  Amax: acceleration constraints of all active DOFs
  '''
  def __init__(self, robot, traj, timestep=None, Vmax=None, Amax=None):
    self.env = robot.GetEnv()
    self.robotname = robot.GetName()
    self.DOF = robot.GetActiveDOF() # total DOFs = joint DOFs + affine DOFs
    self.joint_DOF = len(robot.GetActiveDOFIndices())
    self.affine_DOF = self.DOF - self.joint_DOF
    self.input_traj = traj
    self.timestep = timestep
    self.Vmax = Vmax
    self.Amax = Amax

  def check_inputs(self):
    valid = True
    # check input trajectory
    if self.input_traj is not None:
      self.spec = self.input_traj.GetConfigurationSpecification()
    else:
      print("[RetimeOpenraveTrajectory] [ERROR] Invalid inputs: traj is None.")
      valid = False
    # check constraints
    if (self.Vmax is None) or (self.Amax is None) or (self.timestep is None):
      print("[RetimeOpenraveTrajectory] [ERROR] Invalid inputs: Arguments timestep, Vmax, Amax must be specified.")
      valid = False
    if (min(self.Vmax) <= 0.) or (min(self.Vmax) <= 0.) or (self.timestep <= 0.):
      print("[RetimeOpenraveTrajectory] [ERROR] Invalid inputs: timestep and all elements of Vmax, Amax must be positive.")
      valid = False
    if len(self.Vmax) != self.DOF:
      print("[RetimeOpenraveTrajectory] [ERROR] Invalid inputs: Vmax's dof ({}) does not match robot's total active dof ({}).".format(len(self.Vmax), self.DOF))
      valid = False
    if len(self.Amax) != self.DOF:
      print("[RetimeOpenraveTrajectory] [ERROR] Invalid inputs: Amax's dof ({}) does not match robot's total active dof ({}).".format(len(self.Amax), self.DOF))
      valid = False
    return valid

  def init_output_traj(self, env, spec):
    output_traj = orpy.RaveCreateTrajectory(env, '')
    output_traj.Init(spec)
    return output_traj

  def extract_offsets(self, spec, robotname, joint_DOF, affine_DOF):
    if joint_DOF > 0:
      values_group = spec.GetGroupFromName('joint_values {0}'.format(robotname))
      velocities_group = spec.GetGroupFromName('joint_velocities {0}'.format(robotname))
      deltatime_group = spec.GetGroupFromName('deltatime')
      values_offset = values_group.offset
      velocities_offset = velocities_group.offset
      deltatime_offset = deltatime_group.offset
    else:
      values_offset = 0
      velocities_offset = affine_DOF
      deltatime_offset = 2 * affine_DOF
    return values_offset, velocities_offset, deltatime_offset

  def get_start_end_points(self, traj, DOF, values_offset):
    start_waypoint = traj.GetWaypoint(0).tolist()
    end_waypoint = traj.GetWaypoint(-1).tolist()
    qstart = start_waypoint[values_offset:values_offset+DOF]
    qend = end_waypoint[values_offset:values_offset+DOF]
    return qstart, qend

  def calculate_constraints(self, qstart, qend, Vmax, Amax):
    vmax_list = []
    amax_list = []
    delta_q = np.absolute(np.array(qend)-np.array(qstart))
    for i in range(len(delta_q)):
      if not np.isclose(delta_q[i], 0):
        vmax_list.append(Vmax[i] / delta_q[i])
        amax_list.append(Amax[i] / delta_q[i])
    if len(vmax_list)==0:
      raise Exception("[RetimeOpenraveTrajectory] Start and end points are too close.")
    vmax = min(vmax_list)
    amax = min(amax_list)
    return vmax, amax
  
  def generate_trapezoidal_profile(self, timestep, vmax, amax):
    '''
    Trapezoidal profile generator, generates time-parameterization s(t): [0,T] -> [0,1]
    which minimizes the traversal time T while respecting vmax and amax constraints.
    @type   vmax: float
    @param  vmax: velocity constraint: v(t) = ds/dt, |v(t)| <= vmax for all t in [0,T]
    @type   amax: float
    @param  amax: acceleration constraint: a(t) = dv/dt, |a(t)| <= amax for all t in [0,T]
    @type   timestep: float
    @param  timestep: time interval between 2 successive waypoints, should be set at 1./{controller rate}
    '''
    success = True
    # initialize time, position, velocity, and acceleration arrays
    times = [] # [sec]
    s = []     # [sec^0]
    v = []     # [sec^-1]
    a = []     # [sec^-2]
    # check inputs
    if (vmax is None) or (amax is None) or (timestep is None):
      raise Exception("[generate_trapezoidal_profile] All arguments (timestep, vmax, amax) must be specified.")
    if (vmax <= 0.) or (amax <= 0.) or (timestep <= 0.):
      raise Exception("[generate_trapezoidal_profile] All arguments (timestep, vmax, amax) must be positive.")
    # generate profile
    if vmax*vmax/amax >= 1: # triangle
      Ta = 1./np.sqrt(amax)
      T = 2./np.sqrt(amax)
      num_points = int(math.ceil(T/timestep))
      if num_points < 5:
        print("[generate_trapezoidal_profile] [WARN] timestep is too big.")
        success = False
      for i in range(num_points):
        t = i * timestep
        times.append(t)
        # start point
        if i == 0:
          a.append(amax)
          v.append(0.)
          s.append(0.)
        # segment 1
        elif t > 0 and t < Ta:
          a.append(amax)
          v.append(amax*t)
          s.append(amax*(t**2.)/2.)
        # segment 2
        elif t >= Ta and t < T:
          ti = t - Ta
          a.append(-amax)
          v.append(amax*Ta-amax*ti)
          s.append(amax*(Ta**2.)/2. + amax*Ta*ti - amax*(ti**2.)/2.)
        # end point
        else:
          a.append(0.)
          v.append(0.)
          s.append(1.)
    else: # trapezoidal
      Ta = vmax/amax
      Tv = 1./vmax - vmax/amax
      T = 1./vmax + vmax/amax
      num_points = int(math.ceil(T/timestep)) + 1
      if num_points < 5:
        print("[generate_trapezoidal_profile] [WARN] timestep is too big.")
        success = False
      for i in range(num_points):
        t = i * timestep
        times.append(t)
        # start point
        if i == 0:
          a.append(amax)
          v.append(0.)
          s.append(0.)
        # segment 1
        elif t > 0 and t < Ta:
          a.append(amax)
          v.append(amax*t)
          s.append(amax*(t**2.)/2.)
        # segment 2
        elif t >= Ta and t < (Ta+Tv):
          ti = t - Ta
          a.append(0.)
          v.append(vmax)
          s.append(vmax*Ta/2. + vmax*ti)
        # segment 3
        elif t >= (Ta+Tv) and t < T:
          ti = t - (Ta+Tv)
          a.append(-amax)
          v.append(vmax - amax*ti)
          s.append(vmax*(Tv+Ta/2.) + vmax*ti - amax*(ti**2.)/2.)
        # end point
        else:
          a.append(0.)
          v.append(0.)
          s.append(1.)
    return s, v, a, times, success

  def retime(self):
    # check input arguments
    valid = self.check_inputs()
    if not valid:
      exit()
    # initilize output trajectory
    output_traj = self.init_output_traj(self.env, self.spec)
    # extract offsets
    [values_offset, velocities_offset, deltatime_offset] = self.extract_offsets(self.spec, self.robotname, self.joint_DOF, self.affine_DOF)
    # get start and end configurations
    [qstart, qend] = self.get_start_end_points(self.input_traj, self.DOF, values_offset)
    # time-parameterization
    [vmax, amax] = self.calculate_constraints(qstart, qend, self.Vmax, self.Amax)
    [s, v, a, times, success] = self.generate_trapezoidal_profile(self.timestep, vmax, amax)
    # add waypoints to the output trajectory
    if success:
      num_waypoints = len(s)
      waypoint_dof = len(self.input_traj.GetWaypoint(0).tolist())
      for i in range(num_waypoints):
        waypoint = np.zeros(waypoint_dof)
        if waypoint_dof == (max(values_offset, velocities_offset, deltatime_offset) + 2):
          waypoint[-1] = self.input_traj.GetWaypoint(0).tolist()[-1]
        if i == 0:
          waypoint[deltatime_offset] = 0
        else:
          waypoint[deltatime_offset] = self.timestep
        waypoint[values_offset:values_offset+self.DOF] = np.array(qstart) + s[i] * (np.array(qend)-np.array(qstart))
        waypoint[velocities_offset:velocities_offset+self.DOF] = v[i] * (np.array(qend)-np.array(qstart))
        output_traj.Insert(i, waypoint, True)
    else:
      output_traj = self.input_traj
    return output_traj, success

# END