#!/usr/bin/env python
#MoboTSCP
import mobotscp as mtscp
#utils
import criutils as cu
import IPython
import logging
import numpy as np
import openravepy as orpy
import raveutils as ru
import signal
import time
import tf.transformations as tr

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
  logger = logging.getLogger('MoboTSCP_demo')
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

  # Register the targets
  wing = env.GetKinBody('wing')
  targets_ray = []   # list of targets in orpy.Ray type for OpenRAVE usage
  targets_array = [] # list of targets in numpy.array type for computation usage
  max_no_tar = 288   # number of targets to be registered
  i_tar = 0
  for link in wing.GetLinks():
    lname = link.GetName()
    if lname.startswith('hole') and i_tar<max_no_tar:
      azimuth = np.deg2rad((i_tar%24)*4-46)
      transform = mtscp.utils.add_orien_to_targets(link.GetTransform(), azimuth)
      targets_ray.append( ru.conversions.to_ray(transform) )
      targets_array.append( mtscp.utils.to_array(transform) )
      i_tar += 1
  targets_array = np.vstack(targets_array)
  targets_theta = np.arccos(targets_array[:,-1])
  logger.info("No of targets: {}".format(len(targets_ray)))

  # FKR parameters
  # > sample orientations
  theta_min = min(targets_theta)
  theta_max = max(targets_theta)
  theta_gap = theta_max - theta_min
  samples = 5
  phi = 0.
  fkr_orien_list = []
  for i in range(samples):
    theta_i = theta_min + i*theta_gap/(samples-1)
    fkr_orien_list.append( [ np.sin(theta_i)*np.cos(phi), np.sin(theta_i)*np.sin(phi), np.cos(theta_i) ] )
  # > set FKR generator's parameters 
  gen_fkr_param = mtscp.fkreach.GenerateFKR(sampling_mode="visible-front", xyz_delta=0.05, angle_inc=np.pi/12, \
                                            angle_offset=-np.pi/4, l1_from_ground=0.5175, max_radius=None, \
                                            orientation_list=fkr_orien_list)
  # > comment the line below to generate FKR data, or leave it uncommented if offline data is available
  gen_fkr_param = None

  # Focused Kinematic Reachability (FKR)
  # > generate new (if gen_fkr_param is not None) or load available FKR data (if gen_fkr_param=None)
  fkr_param = mtscp.fkreach.FKRParameters(data_id="mobile_manipulator_drill_wing_task", gen_fkr_param=gen_fkr_param)
  fkr = mtscp.fkreach.FocusedKinematicReachability(env, robot, fkr_param)
  # > analyze FKR for reachability parameters
  reach_param = fkr.calculate_reach_limits(Xmin_wrt_arm=0.5, Zmin_wrt_arm=0.08, Zmax_wrt_arm=0.88, \
                                           arm_ori_wrt_base=[0.2115, 0., 0.320], safe_margin=0.05, \
                                           l0_name='denso_link0', l1_name='denso_link1', l2_name='denso_link2')

  # Define and discretize the floor
  floor = mtscp.utils.RectangularFloor(floor_gridsize=0.1, floor_xrange=[-1., 0.3], floor_yrange=[-1., 1.])
  floor_allpoints = floor.floor_allpoints

  # MoboTSCP solver
  solver = mtscp.solver.MoboTSCP(targets_array, floor_allpoints, reach_param)
  output = solver.solve(SCP_solver='SCPy', SCP_maxiters=20, maxiters=100)
  # > extract info
  logger.info("MoboTSP solver finished successfully:")
  logger.info("* number of clusters: {}".format(output["clusters_no"]))
  logger.info("* cluster_time: {} s".format(output["cluster_time"]))

  # Viewer
  env.SetDefaultViewer()
  while env.GetViewer() is None:
    time.sleep(0.1)
  Tcamera = tr.euler_matrix(*np.deg2rad([-110, 0, 210]))
  Tcamera[:3,3] = [-2, 3, 2]
  viewer = env.GetViewer()
  viewer.SetCamera(Tcamera)
  viewer.SetBkgndColor([.8, .85, .9])

  # Visualize clusters
  clusters_no = output["clusters_no"]
  draws = []
  for i in range(clusters_no):
    tars = targets_array[output["clusters"][i]]
    for j in range(len(tars)):
      draws.append( ru.visual.draw_point(env, tars[j], 4, (np.sqrt(1.-(i/clusters_no)**2), (i+1)/clusters_no, 0.)) )


  # Clear and exit
  logger.info("Finished.")
  IPython.embed()
# END