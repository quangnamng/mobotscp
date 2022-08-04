#!/usr/bin/env python
import IPython
import logging
import numpy as np
import openravepy as orpy
import signal
import time
import tf.transformations as tr
# utils
import criutils as cu
import raveutils as ru
from mobotscp import fkreach, connection, utils

##############################################################################################################
# Input: 
#   * robot model: mobile_manipulator_drill
#   * task: drill on all targets on the wing surface
# Output: 
#   * cluster the task into sub-tasks by assigning all targets into groups
#   * find optimal drilling sequence to complete the task with minimum base's moves
##############################################################################################################


def exit_handler(signum, frame):
  print("[ERROR] Ctrl+C was pressed, Exiting...")
  env.Reset()
  env.Destroy()
  exit()


if __name__ == "__main__":
  # Detect Ctrl+C signal
  signal.signal(signal.SIGINT, exit_handler)

  # Configure the logger
  logger = logging.getLogger('MoboTSCP_wing_drilling_demo')
  cu.logger.initialize_logging(format_level=logging.INFO)

  # Load the OpenRAVE environment
  env = orpy.Environment()
  world_xml = 'worlds/wing_drilling_task.env.xml'
  if not env.Load(world_xml):
    logger.error('Failed to load: {0}'.format(world_xml))
    raise IOError
  logger.info('Loaded OpenRAVE environment: {}'.format(world_xml))
  orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

  # Setup robot and manipulator
  robot = env.GetRobot('robot')
  manipulator = robot.SetActiveManipulator('drill')
  # TODO: set active DOFs, home config, velocity limits

  # Load IKFast and links stats
  iktype = orpy.IkParameterizationType.Transform6D
  if not ru.kinematics.load_ikfast(robot, iktype):
    logger.error('Failed to load IKFast {0}'.format(iktype.name))
    raise IOError
  success = ru.kinematics.load_link_stats(robot, xyzdelta=0.04)

  # Collect all targets
  wing = env.GetKinBody('wing')
  targets_ray = []
  targets_array = []
  targets_theta = []
  for link in wing.GetLinks():
    lname = link.GetName()
    if lname.startswith('hole'):
      targets_ray.append( ru.conversions.to_ray(link.GetTransform()) )
      targets_array.append( utils.to_array(link.GetTransform()) )
      targets_theta.append( np.arccos(link.GetTransform()[2,2]) )
  targets_array = np.vstack(targets_array)
  logger.info("No of targets: {}".format(len(targets_ray)))

  # FKR parameters
  # > sample orientations
  theta_min = min(targets_theta)
  theta_max = max(targets_theta)
  theta_gap = theta_max - theta_min
  samples = 5
  phi = 0.
  orien_list = []
  for i in range(samples):
    theta_i = theta_min + i*theta_gap/(samples-1)
    orien_list.append( [ np.sin(theta_i)*np.cos(phi), np.sin(theta_i)*np.sin(phi), np.cos(theta_i) ] )
  # > set FKR generator's parameters 
  gen_fkr_param = fkreach.GenerateFKR(sampling_mode="visible-front", xyz_delta=0.05, angle_inc=np.pi/12, \
                                      angle_offset=-np.pi/4, max_radius=None, orientation_list=orien_list, \
                                      l1_from_ground=0.5175)
  # > comment the line below to generate new FKR, otherwise leave it uncommented
  gen_fkr_param = None

  # Focused Kinematic Reachability
  # > generate new (if gen_fkr_param is not None) or load available FKR data (if gen_fkr_param=None)
  fkr_param = fkreach.FKRParameters(data_id="mobile_manipulator_drill_wing_task", gen_fkr_param=gen_fkr_param)
  fkr = fkreach.FocusedKinematicReachability(env, robot, fkr_param)
  # > analyze FKR for reachability limits
  reach_param = fkr.calculate_reach_limits(Xmin_wrt_arm=0.5, Zmin_wrt_arm=0.08, Zmax_wrt_arm=0.88, \
                                           arm_ori_wrt_base=[0.2115, 0., 0.320], safe_margin=0.05, \
                                           l0_name='denso_link0', l1_name='denso_link1', l2_name='denso_link2')
  # > uncomment the line below to visualize FKR iso surface & reachability limits
  #fkr.visualize(l0_name='denso_link0', l1_name='denso_link1', showlimits=True, reach_param=reach_param)

  # Define and discretize the floor
  floor = connection.RectangularFloor(0.1, -1.0, -0.3, -1.5, 1.5)

  # Connect targets to floor points
  tar2floor = connection.ConnectTargets2Floor(targets_array, floor, reach_param)
  sets_floor_validinds = tar2floor.connect()

  # geoSCP: find the least number of points on floor to cover all targets

  # # Viewer
  # env.SetDefaultViewer()
  # while env.GetViewer() is None:
  #   time.sleep(0.1)
  # Tcamera = tr.euler_matrix(*np.deg2rad([-110, 0, 210]))
  # Tcamera[:3,3] = [-2, 3, 2]
  # viewer = env.GetViewer()
  # viewer.SetCamera(Tcamera)
  # viewer.SetBkgndColor([.8, .85, .9])


  # Clear and exit
  logger.info("Finished.")
  IPython.embed()
# END