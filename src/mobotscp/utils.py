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
    [floor_Xmin, floor_Xmax] = np.floor( np.array(floor_xrange)/floor_gridsize )
    [floor_Ymin, floor_Ymax] = np.floor( np.array(floor_yrange)/floor_gridsize )
    X, Y = np.mgrid[floor_Xmin:floor_Xmax+1, floor_Ymin:floor_Ymax+1]
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
        self.points.append( ru.visual.draw_point(env=env, point=targets_xyz, size=3, \
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
      self.poses.append( ru.visual.draw_point(env=env, point=base_xyz, size=30, \
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

# END