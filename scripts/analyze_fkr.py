#!/usr/bin/env python
import openravepy as orpy
import IPython
import numpy as np
import raveutils as ru
from mobotscp import fkreach

##############################################################################################################
# Input: 
#   * robot model
#   * a set of orientations [sin(theta), 0, cos(theta)] with theta = polar angle
# Output: 
#   * raw Focus Kinematic Reachability (FKR): reachable points in discrete space relative to robot
#   * parameterize the reachable region by reachability limits: Rmin, Rmax, Zmin, Zmax, Xmin
##############################################################################################################


if __name__ == "__main__":
  # Load the OpenRAVE environment
  env = orpy.Environment()
  if not env.Load('worlds/empty_world.env.xml'):
    print('Failed to load the world. Did you run: catkin_make install?')
    exit(1)

  # Setup robot and manipulator
  robot = env.GetRobot('robot')
  manipulator = robot.SetActiveManipulator('drill')

  # Load IKFast and links stats
  iktype = orpy.IkParameterizationType.Transform6D
  if not ru.kinematics.load_ikfast(robot, iktype):
    print('Failed to load IKFast {0}'.format(iktype.name))
    exit()
  success = ru.kinematics.load_link_stats(robot, xyzdelta=0.04)

  # FKR parameters
  # > sample orientations
  theta = np.deg2rad([114.2, 116.6, 121.0, 132.0, 140.2, 148.0])
  phi = 0.
  orien_list = []
  for i in range(len(theta)):
    orien_list.append( [ np.sin(theta[i])*np.cos(phi), np.sin(theta[i])*np.sin(phi), np.cos(theta[i]) ] )
  # > set FKR generator's parameters 
  gen_fkr_param = fkreach.GenerateFKR(sampling_mode="visible-front", xyz_delta=0.04, angle_inc=np.pi/12, \
                                      angle_offset=-np.pi/4, max_radius=None, orientation_list=orien_list, \
                                      l1_from_ground=0.5175)
  # > comment the line below to generate new FKR
  gen_fkr_param = None

  # Focused Kinematic Reachability
  # > generate new (if gen_fkr_param is not None) or load available FKR data (if gen_fkr_param=None)
  fkr_param = fkreach.FKRParameters(data_id="mobile_manipulator_drill_114_148deg", gen_fkr_param=gen_fkr_param)
  fkr = fkreach.FocusedKinematicReachability(env, robot, fkr_param)
  # > analyze FKR for reachability limits
  reach_param = fkr.calculate_reach_limits(Xmin_wrt_arm=0.5, Zmin_wrt_arm=0.08, Zmax_wrt_arm=0.88, \
                                           arm_ori_wrt_base=[0.2115, 0., 0.320], safe_margin=0.05, \
                                           l0_name='denso_link0', l1_name='denso_link1', l2_name='denso_link2')
  # > uncomment the line below to visualize FKR iso surface & reachability limits
  fkr.visualize(l0_name='denso_link0', l1_name='denso_link1', showlimits=True, reach_param=reach_param)


  IPython.embed()
# END