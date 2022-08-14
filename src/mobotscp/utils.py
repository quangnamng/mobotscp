#!/usr/bin/env python
import copy
import numpy as np

##############################################################################################################
# Useful utility functions 
##############################################################################################################


def z_rotation(xyz, yaw_rad):
  rot_matrix = [ [np.cos(yaw_rad), -np.sin(yaw_rad), 0.],
                 [np.sin(yaw_rad), np.cos(yaw_rad),  0.],
                 [0.,              0.,               1.] ]
  rotated_xyz = np.matmul(rot_matrix, np.transpose(xyz))
  return np.transpose(rotated_xyz)


def to_array(transform):
  """
  Convert a homogeneous transformation into an `numpy.array`.

  Parameters
  ----------
  transform:  array_like
    The input homogeneous transformation

  Returns
  -------
  transform: numpy.array
    The resulting array
  """
  array = np.ravel( [transform[:3,3], transform[:3,2]] )
  return array


def add_orien_to_targets(transform, yaw):
  new_tran = copy.copy(transform)
  orient = np.array(transform[:3,2])
  new_orient = z_rotation(orient, yaw)
  new_tran[:3,2] = new_orient
  return new_tran


class RectangularFloor(object):
  def __init__(self, floor_gridsize=0.1, floor_xrange=[-1., 0.], floor_yrange=[-1., 1.], floor_z=0.):
    [floor_Xmin, floor_Xmax] = np.array(floor_xrange)//floor_gridsize
    [floor_Ymin, floor_Ymax] = np.array(floor_yrange)//floor_gridsize
    X, Y = np.mgrid[floor_Xmin:floor_Xmax, floor_Ymin:floor_Ymax]
    self.floor_allpoints = np.c_[X.flat, Y.flat] * floor_gridsize
    self.floor_z = floor_z

# END