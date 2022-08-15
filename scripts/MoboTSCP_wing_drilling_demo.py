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
  azimuths = [np.deg2rad(46-(i%24)*4) for i in range(max_targets)]
  targets = mtscp.utils.RegisterTargets(links=wing.GetLinks(), targets_name='hole', \
                                        max_targets=max_targets, add_azimuth=azimuths)
  logger.info("Number of targets: {}".format(len(targets.targets_ray)))

  # Define and discretize the floor
  floor = mtscp.utils.RectangularFloor(floor_gridsize=0.1, floor_xrange=[-1., 0.3], \
                                       floor_yrange=[-1., 1.], floor_z = 0.)


  # FKR parameters
  # > sample orientations
  theta_min = min(targets.targets_theta)
  theta_max = max(targets.targets_theta)
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


  # SCP parameters
  scp_param = mtscp.solver.SCPparameters(SCP_solver='SCPy', SCP_maxiters=20, cluster_maxiters=100)

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
  tsp_param.standoff = 0.002
  tsp_param.step_size = np.pi/4.
  tsp_param.affine_velocity_limits = velocity_limits
  tsp_param.affine_acceleration_limits = acceleration_limits
  # > planning parameters
  tsp_param.try_swap = True
  tsp_param.planner = 'BiRRT'
  tsp_param.max_iters = 100
  tsp_param.max_ppiters = 30

  # MoboTSCP solver
  # > solve
  solver = mtscp.solver.MoboTSCP(robot, targets, floor, reach_param, scp_param, tsp_param)
  output = solver.solve()
  # > extract info
  logger.info("MoboTSP solver finished successfully:")
  logger.info("* number of clusters: clusters_no = {}".format(output["clusters_no"]))
  logger.info("* time used: mobotscp_time = {} s".format(output["mobotscp_time"]))


  # Viewer
  env.SetDefaultViewer()
  while env.GetViewer() is None:
    time.sleep(0.1)
  Tcamera = tr.euler_matrix(*np.deg2rad([-110, 0, 210]))
  Tcamera[:3,3] = [-2, 3, 2]
  viewer = env.GetViewer()
  viewer.SetCamera(Tcamera)
  viewer.SetBkgndColor([.8, .85, .9])

  # Visualize clusters and base tour
  clusters = output["clusters"]
  base_tour = output["base_tour"]
  base_poses = output["base_poses"]
  visual_solution = mtscp.utils.VisualizeSolution(targets.targets_ray, clusters, base_tour)
  visual_solution.visualize_clusters(env)
  visual_solution.visualize_base_tour(env, base_poses, base_home, floor.floor_z)


  # Execute the trajectories
  robot.SetActiveDOFValues(output["cgraph"].node[output["config_tour"][0]]['value'])
  raw_input("Press Enter to start simulation...")
  logger.info("Executing the trajectories...")
  draw = []
  draw_htour = np.array(output["target_tour"])[1:-1]-1
  max_traj_idx = len(output["trajs"])-1
  sim_starttime = time.time()
  for i, traj in enumerate(output["trajs"]):
    robot.GetController().SetPath(traj)
    robot.WaitForController(0)
    if(max_traj_idx-i):
      tgt = output["targets_xyz"][draw_htour[i]+1]
      draw.append(ru.visual.draw_point(env, tgt, 10, np.array([1,1,1])))
  sim_time = time.time() - sim_starttime
  logger.info("Executed all trajectories in {} s".format(sim_time))


  # Clear and exit
  logger.info("Finished.")
  IPython.embed()
# END