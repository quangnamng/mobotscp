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
  # > scale down joint limits for safety
  joint_limits = [robot.GetDOFLimits()[0]*0.95, robot.GetDOFLimits()[1]*0.95]
  robot.SetDOFLimits(joint_limits[0], joint_limits[1])

  # Load IKFast and links stats
  iktype = orpy.IkParameterizationType.Transform6D
  if not ru.kinematics.load_ikfast(robot, iktype):
    print('Failed to load IKFast {0}'.format(iktype.name))
    exit()
  success = ru.kinematics.load_link_stats(robot, xyzdelta=0.04)

  # FKR parameters
  # > sample orientations
  theta_min = np.deg2rad(100)
  theta_max = np.deg2rad(150)
  theta_gap = theta_max - theta_min
  samples = 10
  phi = 0.
  fkr_orien_list = []
  for i in range(samples):
    theta_i = theta_min + i*theta_gap/(samples-1)
    fkr_orien_list.append( [ np.sin(theta_i)*np.cos(phi), np.sin(theta_i)*np.sin(phi), np.cos(theta_i) ] )
  # > set FKR generator's parameters 
  gen_fkr_param = fkreach.GenerateFKR(sampling_mode="visible-front", xyz_delta=0.05, angle_inc=np.pi/4, \
                                      angle_offset=0., max_radius=None, lbase_name='ridgeback_chassis_link', \
                                      l1_name='denso_link1', j1_lim=[-np.pi/2, np.pi/2], \
                                      orientation_list=fkr_orien_list)
  ##################################################
  # > comment the next line to generate new FKR data
  gen_fkr_param = None
  ##################################################

  # Focused Kinematic Reachability (FKR)
  # > generate new (if gen_fkr_param is not None) or load available FKR data (if gen_fkr_param=None)
  fkr_param = fkreach.FKRParameters(data_id="mobile_manipulator_drill_110-150deg", gen_fkr_param=gen_fkr_param)
  fkr = fkreach.FocusedKinematicReachability(env, robot, fkr_param)
  # > analyze FKR for reachability parameters
  reach_param = fkr.calculate_reach_limits(Xmin_wrt_arm=0.3, Zmin_wrt_arm=0.08, Zmax_wrt_arm=0.88, \
                                           safe_margin=0.0, lbase_name='ridgeback_chassis_link', \
                                           l0_name='denso_link0', l1_name='denso_link1', l2_name='denso_link2')
  # > uncomment the line below to visualize FKR iso surface & reachability limits
  fkr.visualize(l0_name='denso_link0', l1_name='denso_link1', showlimits=True, reach_param=reach_param)


  IPython.embed()
# END