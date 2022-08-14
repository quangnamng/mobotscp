#!/usr/bin/env python
from __future__ import print_function
from mobotscp import utils
import copy
import numpy as np

##############################################################################################################
### Connect targets to the foor points
# Input:
#   * alltargets: (array_like) array of orpy.Ray elements representing 5D targets
#   * floor grid: X, Y = numpy.mgrid[xmin:xmax, ymin:ymax]
#   * floor.allpoints = numpy.c_[X.flat, Y.flat] * gridsize
#   * reach_param (fkreach.FocusedKinematicReachability.calculate_reach_limits)
# Output:
#   * floor_validinds_i = numpy.flatnonzero(conditions): indices of floor's points that can reach target i
#   * floor_validids_per_tar.append(floor_validids_i): sets of floor's valid indices for all targets
#   * targets_reachable: lists of targets having at least 1 floor's valid index
#   * targets_unreachable: lists of targets having 0 floor's valid index
##############################################################################################################


class ConnectTargets2Floor(object):
  def __init__(self, targets, floor, reach_param):
    # targets
    self.targets = copy.copy(targets)
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
    print("--Connection finished successfully:")
    print("  * Number of targets reachable = {}/{}".format(len(targets_reachable), len(self.targets)))
    print("  * Number of targets unreachable = {}/{}".format(len(targets_unreachable), len(self.targets)))
    return floor_validids_per_tar, targets_reachable, targets_unreachable

# END