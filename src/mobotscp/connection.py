#!/usr/bin/env python
import numpy as np
from mobotscp import utils

##############################################################################################################
# Input:
#   * alltargets: (array_like) array of orpy.Ray elements representing 5D targets
#   * floor grid: X, Y = numpy.mgrid[xmin:xmax, ymin:ymax]
#   * floor.allpoints = numpy.c_[X.flat, Y.flat] * gridsize
#   * reach_param (fkreach.FocusedKinematicReachability.calculate_reach_limits)
# Output:
#   * floor_validinds_i = numpy.flatnonzero(conditions)
#   * floor_validpoints = floor_allpoints[numpy.unique(floor_validinds)]
#   * floor_validpoints.append(floor_validpoints_i)

#   * validpoints_targets: (array_like) contains target indices reachable at each validpoint, initial = -1
#   * floor_reachpoints = floor_allpoints[floor_reachinds]
#   * reachpoints_targets = validpoints_targets[floor_reachinds]
##############################################################################################################


class RectangularFloor(object):
  def __init__(self, gridsize=0.1, floor_xmin=-1., floor_xmax=0., floor_ymin=-1., floor_ymax=1.):
    floor_Xmin = floor_xmin//gridsize
    floor_Xmax = floor_xmax//gridsize
    floor_Ymin = floor_ymin//gridsize
    floor_Ymax = floor_ymax//gridsize
    self.gridsize = gridsize
    X, Y = np.mgrid[floor_Xmin:floor_Xmax, floor_Ymin:floor_Ymax]
    self.allfloorpoints = np.c_[X.flat, Y.flat] * gridsize


class ConnectTargets2Floor(object):
  def __init__(self, targets, floor, reach_param):
    # targets
    self.targets = targets
    self.targets_theta = np.arccos(self.targets[:,-1])
    self.targets_see_dir = [ self.targets[:,-3]/np.sqrt(self.targets[:,-3]**2+self.targets[:,-2]**2), \
                           self.targets[:,-2]/np.sqrt(self.targets[:,-3]**2+self.targets[:,-2]**2) ]
    self.targets_phi = np.arctan2( self.targets_see_dir[1], self.targets_see_dir[0] )
    # floor
    self.gridsize = floor.gridsize
    self.allfloorpoints = floor.allfloorpoints
    self.Xmin_wrt_arm = reach_param.Xmin_wrt_arm
    self.Zmin_wrt_arm = reach_param.Zmin_wrt_arm
    self.Zmax_wrt_arm = reach_param.Zmax_wrt_arm
    self.spheres_center_wrt_arm = np.array(reach_param.spheres_center_wrt_arm)
    self.arm_ori_wrt_base = np.array(reach_param.arm_ori_wrt_base)
    self.Rmin = reach_param.Rmin
    self.Rmax = reach_param.Rmax

  def connect(self):
    # Sets of floor's valid indices
    sets_floor_validinds = []
    for i in range(len(self.targets)):
      spheres_center_wrt_floor = self.targets[i,:2] - utils.z_rotation(self.spheres_center_wrt_arm, \
                                                                        self.targets_phi[i])[:2]
      z_target_wrt_z_center = self.targets[i,2] - self.spheres_center_wrt_arm[2] - self.arm_ori_wrt_base[2]
      rmin2 = self.Rmin**2 - z_target_wrt_z_center**2
      rmax2 = self.Rmax**2 - z_target_wrt_z_center**2
      floor_validinds_cond1 = np.flatnonzero( \
        np.sum((self.allfloorpoints-spheres_center_wrt_floor)**2, 1) <= rmax2 )
      floor_validinds_cond2 = np.flatnonzero( \
        np.sum((self.allfloorpoints[floor_validinds_cond1]-spheres_center_wrt_floor)**2, 1) >= rmin2 )
      floor_validinds_cond12 = floor_validinds_cond1[floor_validinds_cond2]
      r_tar_to_point = self.targets[i,:2]-self.allfloorpoints[floor_validinds_cond12]
      floor_validinds_cond3 = np.flatnonzero( ( r_tar_to_point[:,0]*self.targets_see_dir[0][i] + \
        r_tar_to_point[:,1]*self.targets_see_dir[1][i] ) >= self.Xmin_wrt_arm )
      floor_validinds_i = floor_validinds_cond12[floor_validinds_cond3]
      sets_floor_validinds.append(floor_validinds_i)
    list_floor_validinds = np.unique(np.concatenate(sets_floor_validinds))
    # List of floor's valid points
    list_floor_validpoints = self.allfloorpoints[list_floor_validinds]
    print("List of floor's valid points = {}".format(list_floor_validpoints))
    return sets_floor_validinds


# END