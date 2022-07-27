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
#   * a reachable region with geometric shape based on raw FKR data
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
  orien_list = []
  theta = np.deg2rad([148.0, 140.2, 132.0, 121.0, 116.6, 114.2])
  phi = 0.
  for i in range(len(theta)):
    orien_list.append( [ np.sin(theta[i])*np.cos(phi), np.sin(theta[i])*np.sin(phi), np.cos(theta[i]) ] )
  # > FKR generator's parameters
  gen_fkr_param = None   # uncomment the line below to generate new FKR 
  # gen_fkr_param = fkreach.GenerateFKR([1., 0], "visible-front", 0.05, np.pi/12, -np.pi/4, orientation_list=orien_list, l1_from_ground=0.5175)

  # Get FKR data
  fkr_param = fkreach.FKRParameters(data_id="mobile_manipulator_drill_114_148deg", gen_fkr_param=gen_fkr_param)
  fkr = fkreach.FocusKinematicReachability(env, robot, fkr_param)

  # Solve FKR for limits
  reach_param = fkr.calculate_reach_limits(Xmin_wrt_arm=0.4, Zmin_wrt_arm=-0.2, Zmax_wrt_arm=0.3, \
                                           arm_ori_wrt_base=[0.2115, 0., 0.320], safe_margin=0.05, \
                                           l0_name='denso_link0', l1_name='denso_link1', l2_name='denso_link2')

  # visualize FKR & limits
  fkr.visualize(l0_name='denso_link0', l1_name='denso_link1', showlimits=True, reach_param=reach_param)


  IPython.embed()