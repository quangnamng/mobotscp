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
  print("[ERROR] Ctrl+C was pressed, exiting...")
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
  manip = robot.SetActiveManipulator('drill')
  robot.SetActiveDOFs(manip.GetArmIndices(), \
                      robot.DOFAffine.X|robot.DOFAffine.Y|robot.DOFAffine.RotationAxis,[0,0,1])
  # > home config
  base_home = [-2.0, 0., 0.]   # (x, y, yaw)
  arm_home = np.deg2rad([0, -20, 130, 0, 70, 0])
  qhome = np.append(arm_home, base_home)
  with env:
    robot.SetActiveDOFValues(qhome)
    Thome = robot.GetTransform()
    phome = manip.GetEndEffectorTransform()[:3,3]
  # > load IKFast and links stats
  iktype = orpy.IkParameterizationType.Transform6D
  if not ru.kinematics.load_ikfast(robot, iktype):
    logger.error('Failed to load IKFast {0}'.format(iktype.name))
    raise IOError
  success = ru.kinematics.load_link_stats(robot, xyzdelta=0.04)
  # > set velocity & acceleration limits
  velocity_limits = (robot.GetDOFVelocityLimits()*0.1).tolist()
  acceleration_limits = [5., 4.25, 4.25, 5.25, 6., 8.]
  robot.SetDOFVelocityLimits(velocity_limits)
  robot.SetDOFAccelerationLimits(acceleration_limits)
  velocity_limits.extend([0.3, 0.3, 0.3])
  acceleration_limits.extend([0.2, 0.2, 0.2])


  # Register the targets
  wing = env.GetKinBody('wing')
  targets_ray = []   # list of targets in orpy.Ray type for OpenRAVE usage
  targets_array = [] # list of targets in numpy.array type for computation usage
  max_no_tar = 288   # number of targets to be registered
  i_tar = 0
  for link in wing.GetLinks():
    lname = link.GetName()
    if lname.startswith('hole') and i_tar<max_no_tar:
      azimuth = np.deg2rad(46-(i_tar%24)*4)
      #azimuth = 0.
      transform = mtscp.utils.add_orien_to_targets(link.GetTransform(), azimuth)
      targets_ray.append( ru.conversions.to_ray(transform) )
      targets_array.append( mtscp.utils.to_array(transform) )
      i_tar += 1
  targets_array = np.vstack(targets_array)
  targets_theta = np.arccos(targets_array[:,-1])
  logger.info("No of targets: {}".format(len(targets_ray)))

  # Define and discretize the floor
  floor = mtscp.utils.RectangularFloor(floor_gridsize=0.1, floor_xrange=[-1., 0.3], \
                                       floor_yrange=[-1., 1.], floor_z = 0.)


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


  # MoboTSCP solver
  # > SCP parameters
  scp_param = mtscp.solver.SCPparameters(SCP_solver='SCPy', SCP_maxiters=20, cluster_maxiters=100)
  # > TODO: TSP parameters
  tsp_param = mtscp.solver.TSPparameters(Thome, qhome, phome)

  # > solve
  solver = mtscp.solver.MoboTSCP(robot, targets_array, floor, reach_param, scp_param, tsp_param)
  output = solver.solve()
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

  # Visualize clusters of targets
  arrows = []
  points = []
  axes = []
  tour = []
  colors = []
  colors.append((0., 1., 1.))
  colors.append((1., 0., 1.))
  colors.append((1., 1., 0.))
  colors.append((1., 0., 0.))
  colors.append((0., 1., 0.))
  colors.append((0., 0., 1.))
  colors.append((1., 1., 1.))
  colors.append((0.1, 0.1, 0.1))
  clusters_no = output["clusters_no"]
  base_poses = output["base_poses"]
  base_tour = output["base_tour"]
  for k in range(clusters_no):
    i = base_tour[k]
    # > draw arrows on targets
    for j in range(len(output["clusters"][i])):
      arrow_len = 0.05
      tar_ray = targets_ray[output["clusters"][i][j]]
      tar_ray = orpy.Ray(tar_ray.pos()-arrow_len*tar_ray.dir(), tar_ray.dir())
      arrows.append( ru.visual.draw_ray(env=env, ray=tar_ray, dist=arrow_len, linewidth=0., color=colors[i]) )
    # > draw points at the base poses
    base_xyz = np.array(list(base_poses[i][:2])+[0])
    points.append( ru.visual.draw_point(env=env, point=base_xyz, size=20, color=colors[i]) )
    # > draw axes at the base poses
    base_trans = tr.euler_matrix(0, 0, base_poses[i][2], 'sxyz')
    base_trans[:3,3] = base_xyz
    axes.append( ru.visual.draw_axes(env=env, transform=base_trans, dist=0.2, linewidth=4) )
    # > draw arrows representing base tour
    if k < len(base_poses)-1:
      base_xyz_next = np.array(list(base_poses[base_tour[k+1]][:2])+[0])
      tour_len = np.linalg.norm(base_xyz_next-base_xyz)
      tour_dir = (base_xyz_next-base_xyz)/tour_len
      tour_ray = orpy.Ray(base_xyz, tour_dir)
      tour.append( ru.visual.draw_ray(env=env, ray=tour_ray, dist=tour_len, linewidth=2, color=colors[i]) )
  # > draw the reachability limits that bound each cluster
  # drawer = mtscp.fkreach.DrawReachLimits(reach_param)
  # drawer.draw_limits(env, base_poses[0])


  # TODO: Execute the trajectories
  logger.info("Executing the trajectories...")




  # Clear and exit
  logger.info("Finished.")
  IPython.embed()
# END