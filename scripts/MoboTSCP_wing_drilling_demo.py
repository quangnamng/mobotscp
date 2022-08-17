#!/usr/bin/env python
#MoboTSCP
import mobotscp as mtscp
#utils
import criutils as cu
import IPython
import logging
import numpy as np
import openravepy as orpy
import robotsp as rtsp
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
  # > scale down joint limits for safety
  joint_limits = [robot.GetDOFLimits()[0]*0.95, robot.GetDOFLimits()[1]*0.95]
  robot.SetDOFLimits(joint_limits[0], joint_limits[1])
  # > set velocity & acceleration limits
  velocity_limits = (robot.GetDOFVelocityLimits()*0.1).tolist()
  acceleration_limits = [1., 1., 1., 1., 1., 1.]
  robot.SetDOFVelocityLimits(velocity_limits)
  robot.SetDOFAccelerationLimits(acceleration_limits)
  velocity_limits.extend([0.3, 0.3, 0.3])
  acceleration_limits.extend([0.1, 0.1, 0.1])


  # Register the targets
  wing = env.GetKinBody('wing')
  max_targets = 288
  # > comment 1 of 2 lines below to add azimuthal angles to the targets or not
  azimuths = [np.deg2rad(46-(i%24)*4) for i in range(max_targets)]
  # azimuths = [0]*max_targets
  # > register targets with the azimuthal angles added
  targets = mtscp.utils.RegisterTargets(links=wing.GetLinks(), targets_name='hole', \
                                        max_targets=max_targets, add_azimuth=azimuths)
  logger.info("Number of targets: {}".format(len(targets.targets_ray)))
  targets_theta = np.arccos(targets.targets_array[:,-1])
  theta_deg_min = np.rad2deg(min(targets_theta))
  theta_deg_max = np.rad2deg(max(targets_theta))
  logger.info("Range of targets' polar angles: {}-{} deg".format(theta_deg_min, theta_deg_max))

  # Define and discretize the floor
  floor = mtscp.utils.RectangularFloor(floor_gridsize=0.1, floor_xrange=[-1., 0.3], \
                                       floor_yrange=[-1., 1.], floor_z = 0.)


  # Focused Kinematic Reachability (FKR)
  # > generate new (if gen_fkr_param is not None) or load available FKR data (if gen_fkr_param=None)
  fkr_param = mtscp.fkreach.FKRParameters(data_id="mobile_manipulator_drill_110-150deg", gen_fkr_param=None)
  fkr = mtscp.fkreach.FocusedKinematicReachability(env, robot, fkr_param)
  # > analyze FKR for reachability parameters
  reach_param = fkr.calculate_reach_limits(Xmin_wrt_arm=0.40, Zmin_wrt_arm=0.08, Zmax_wrt_arm=0.88, \
                                           safe_margin=0.00, lbase_name='ridgeback_chassis_link', \
                                           l0_name='denso_link0', l1_name='denso_link1', l2_name='denso_link2')


  # SCP parameters
  scp_param = mtscp.solver.SCPparameters(SCP_solver='SCPy', SCP_maxiters=30, cluster_maxiters=100)

  # TSP parameters
  tsp_param = mtscp.solver.TSPparameters(Thome, qhome, phome, stack_offset=1.0)
  # > task space parameters
  tsp_param.tsp_solver = rtsp.tsp.two_opt
  tsp_param.tspace_metric = rtsp.metric.euclidean_fn
  # > configuration space parameters
  tsp_param.cspace_metric = rtsp.metric.max_joint_diff_fn
  tsp_param.cspace_metric_args = (1./robot.GetActiveDOFMaxVel()[:6],)
  # > kinematics parameters
  tsp_param.iktype = orpy.IkParameterizationType.Transform6D
  tsp_param.standoff = 0.02
  tsp_param.step_size = np.pi/4
  tsp_param.velocity_limits = velocity_limits
  tsp_param.acceleration_limits = acceleration_limits
  # > planning parameters
  tsp_param.planner = 'BiRRT' #options: 'BiRRT' or 'BasicRRT'
  tsp_param.try_swap = True
  tsp_param.max_iters = 100
  tsp_param.max_ppiters = 50
  # > time-parameterize the trajectories to satisfy velocity & acceleration limits
  tsp_param.retimer = 'trapezoidalretimer' #options: 'trapezoidalretimer', 'parabolicsmoother', None
  tsp_param.timestep = 0.02

  # MoboTSCP solver
  # > solve
  solver = mtscp.solver.MoboTSCP(robot, targets, floor, reach_param, scp_param, tsp_param)
  output = solver.solve()
  # > extract info
  logger.info("MoboTSP solver finished successfully:")
  logger.info("* number of clusters: clusters_no = {}".format(output["clusters_no"]))
  logger.info("* total solver time: mobotscp_time = {} s".format(output["mobotscp_time"]))


  # Viewer
  env.SetDefaultViewer()
  while env.GetViewer() is None:
    time.sleep(0.1)
  Tcamera = tr.euler_matrix(*np.deg2rad([-110, 0, 0]))
  Tcamera[:3,3] = [-1., -3, 1.5]
  viewer = env.GetViewer()
  viewer.SetCamera(Tcamera)
  viewer.SetBkgndColor([.8, .85, .9])

  # Visualize clusters and base tour
  clusters = output["clusters"]
  base_tour = output["base_tour"]
  base_poses = output["base_poses"]
  visual_solution = mtscp.utils.VisualizeSolution(targets, clusters, base_tour)
  visual_solution.visualize_clusters(env, draw_arrows=False)
  visual_solution.visualize_base_tour(env, base_poses, base_home, floor.floor_z)


  # Execute the trajectories
  robot.SetActiveDOFValues(output["cgraph"].node[output["config_tour"][0]]['value'])
  raw_input("Press Enter to start simulation...\n")
  logger.info("Executing the trajectories...")
  draws = []
  draw_ttour = np.array(output["task_tour"])[1:-1]-1
  max_traj_idx = len(output["trajs"])-1
  sim_starttime = time.time()
  for i, traj in enumerate(output["trajs"]):
    robot.GetController().SetPath(traj)
    robot.WaitForController(0)
    if (max_traj_idx-i):
      # visit_xyz = output["visit_xyz"][draw_ttour[i]+1]
      # draw.append(ru.visual.draw_point(env, visit_xyz, 7, np.array([51, 51, 51])/255.))
      arrow_len = 0.05
      tar_ray = targets.targets_ray[ (output["target_taskids"])[draw_ttour[i]] ]
      tar_ray = orpy.Ray(tar_ray.pos()-arrow_len*tar_ray.dir(), tar_ray.dir())
      draws.append( ru.visual.draw_ray(env=env, ray=tar_ray, dist=arrow_len, linewidth=1, \
                                       color=np.array([255, 0, 0])/255.) )
  sim_time = time.time() - sim_starttime
  logger.info("Executed all trajectories in {} s".format(sim_time))


  # Clear and exit
  logger.info("Finished.")
  IPython.embed()
# END