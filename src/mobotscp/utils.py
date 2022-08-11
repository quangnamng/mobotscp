#!/usr/bin/env python
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

# END