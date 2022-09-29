#!/usr/bin/env python
# MoboTSCP
import mobotscp as mtscp
# Utils
import IPython
import numpy as np
import openravepy as orpy
import robotsp as rtsp
import raveutils as ru
import signal
import time
import tf.transformations as tr
# ROS
import rospy
from denso_ridgeback_control.conversions import ros_trajs_from_openrave_affined_traj
from denso_ridgeback_control.whole_body_controllers import WB_Trajectory_Controller
# Retiming with jerk limit
from esocstopp import esocstopp_retimer
from denso_ridgeback_control.conversions import ros_trajs_from_openrave_mobile_traj


##############################################################################################################
# Input: 
#   * robot model: mobile_manipulator_drill
#   * task: drill on all targets on the wing surface
# Output: 
#   * cluster the task-space by assigning all targets into clusters
#   * find optimal drilling sequence to complete the task with minimum base's moves
##############################################################################################################


def exit_handler(signum, frame):
  print("[ERROR] Ctrl+C was pressed, exiting...")
  env.Reset()
  env.Destroy()
  exit()

namespace = 'cpr_r100_0018'


if __name__ == "__main__":
  # Detect Ctrl+C signal
  signal.signal(signal.SIGINT, exit_handler)

  # Initialize ROS node
  rospy.init_node('drilling_task')

  # Connect to the hardware interface
  WBC_trajectory = WB_Trajectory_Controller(namespace, timeout=10.0)
  WBC_rate = WBC_trajectory.rate
  timestep = 1./WBC_rate
  print("Whole-body controller rate: {0} Hz".format(WBC_rate))
  print("Choose timestep = {0} s".format(timestep))


  ### OpenRAVE Environment Setup
  # Setup world
  env = orpy.Environment()
  world_xml = 'worlds/drilling_task.env.xml'
  if not env.Load(world_xml):
    rospy.loginfo('Failed to load: {0}'.format(world_xml))
    raise IOError
  rospy.loginfo('Loaded OpenRAVE environment: {}'.format(world_xml))
  orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

  # Setup robot and manipulator
  robot = env.GetRobot('robot')
  manip = robot.SetActiveManipulator('drill')
  robot.SetActiveDOFs(manip.GetArmIndices(), \
                      robot.DOFAffine.X|robot.DOFAffine.Y|robot.DOFAffine.RotationAxis,[0,0,1])
  # > home config
  base_home = [0, -2, np.pi/2]   # (x, y, yaw)
  arm_home = np.deg2rad([0, -20, 130, 0, 70, -45])
  qhome = np.append(arm_home, base_home)
  with env:
    robot.SetActiveDOFValues(qhome)
    Thome = robot.GetTransform()
    phome = manip.GetEndEffectorTransform()[:3,3]
  # > load IKFast and links stats
  iktype = orpy.IkParameterizationType.Transform6D
  if not ru.kinematics.load_ikfast(robot, iktype):
    rospy.loginfo('Failed to load IKFast {0}'.format(iktype.name))
    raise IOError
  success = ru.kinematics.load_link_stats(robot, xyzdelta=0.04)
  # > narrow joint limits for safety
  joint_limits = [robot.GetDOFLimits()[0]*0.95, robot.GetDOFLimits()[1]*0.95]
  robot.SetDOFLimits(joint_limits[0], joint_limits[1])
  # > set velocity & acceleration limits
  velocity_limits = (robot.GetDOFVelocityLimits()*0.2).tolist()
  acceleration_limits = (robot.GetDOFAccelerationLimits()*0.2).tolist()
  robot.SetDOFVelocityLimits(velocity_limits)
  robot.SetDOFAccelerationLimits(acceleration_limits)
  velocity_limits.extend([0.3, 0.3, 0.3])
  acceleration_limits.extend([0.05, 0.05, 0.05])


  ### Task Definition
  # Register the targets
  wing = env.GetKinBody('wing')
  max_num_targets = 336   # the first 288 targets on front side, and the next 48 targets on back side
  # > use the 2nd-3rd lines below to add azimuthal angles to the targets, otherwise comment them
  azimuths = [0]*max_num_targets
  azimuths = [np.deg2rad(36.8-(i%24)*3.2) for i in range(288)] + \
              [np.deg2rad((i%24)-11.5) for i in range(288,max_num_targets)]
  # > register the first 'max_num_targets' targets
  targets = mtscp.utils.RegisterTargets(links=wing.GetLinks(), targets_name='hole', \
                                        max_targets=max_num_targets, add_azimuth=azimuths)
  rospy.loginfo("Number of targets: {}".format(len(targets.targets_ray)))
  targets_theta = np.arccos(targets.targets_array[:,-1])
  theta_deg_min = np.rad2deg(min(targets_theta))
  theta_deg_max = np.rad2deg(max(targets_theta))
  rospy.loginfo("Range of targets' polar angles: {}-{} deg".format(theta_deg_min, theta_deg_max))

  # Define and discretize the floor
  floor = mtscp.utils.RectangularFloor(floor_gridsize=0.1, floor_xrange=[-1., 1.], \
                                       floor_yrange=[-1.5, 0.5], floor_z = 0.)


  ### MoboTSCP Solver
  # Focused Kinematic Reachability (FKR)
  # > load the generated FKR data
  fkr_param = mtscp.fkreach.FKRParameters(data_id="mobile_manipulator_drill_110-150deg", gen_fkr_param=None)
  fkr = mtscp.fkreach.FocusedKinematicReachability(env, robot, fkr_param)
  # > analyze FKR data for reachability parameters
  reach_param = fkr.calculate_reach_limits(Xmin_wrt_arm=0.40, Zmin_wrt_arm=0.08, Zmax_wrt_arm=0.88, \
                                           safe_margin=0.00, lbase_name='ridgeback_chassis_link', \
                                           l0_name='denso_link0', l1_name='denso_link1', l2_name='denso_link2')

  # SCP parameters: available options are 'SCPy', 'LPr', 'greedy'
  scp_param = mtscp.solver.SCPparameters(SCP_solver='SCPy', point_maxiters=10, orient_maxiters=100)

  # TSP parameters
  tsp_param = mtscp.solver.TSPparameters(Thome, qhome, phome)
  # > task space parameters
  tsp_param.tsp_solver = rtsp.tsp.two_opt
  tsp_param.tspace_metric = rtsp.metric.euclidean_fn
  tsp_param.stack_offset = 1.0
  # > configuration space parameters
  tsp_param.cspace_metric = rtsp.metric.max_joint_diff_fn
  tsp_param.cspace_metric_args = (1./robot.GetActiveDOFMaxVel()[:6],)
  # > kinematics parameters
  tsp_param.iktype = orpy.IkParameterizationType.Transform6D
  tsp_param.standoff = 0.04
  tsp_param.step_size = np.pi/6
  tsp_param.velocity_limits = velocity_limits
  tsp_param.acceleration_limits = acceleration_limits
  # > planning parameters
  tsp_param.planner = 'BiRRT' #options: 'BiRRT', 'BasicRRT'
  tsp_param.try_swap = True
  tsp_param.max_iters = 100
  tsp_param.max_ppiters = 50
  # > time-parameterize the trajectories to satisfy velocity_limits & acceleration_limits
  tsp_param.retimer = 'parabolicsmoother'

  # Solve
  solver = mtscp.solver.MoboTSCP(robot, targets, floor, reach_param, scp_param, tsp_param)
  output = solver.solve()


  ### Visualization
  # Setup viewer
  env.SetDefaultViewer()
  while env.GetViewer() is None:
    time.sleep(0.1)
  Tcamera = tr.euler_matrix(*np.deg2rad([-145, 0, 160]))
  Tcamera[:3,3] = [0.5, 2.5, 4]
  viewer = env.GetViewer()
  viewer.SetCamera(Tcamera)
  viewer.SetBkgndColor([1., 1., 1.])

  # Visualize results: clusters of targets and base tour
  clusters = output["clusters"]
  base_tour = output["base_tour"]
  base_poses = output["base_poses"]
  visual_solution = mtscp.utils.VisualizeSolution(targets, clusters, base_tour)
  visual_solution.visualize_clusters(env, draw_arrows=False)
  visual_solution.visualize_base_tour(env, base_poses, base_home, floor.floor_z)


  ##### Execution
  # OpenRAVE to ROS trajectories with jerk limit
  trajs = output["trajs"]
  jerk_limits = [15.,15.,15.,15.,15.,15., 0.5, 0.5, 0.5]
  arm_trajs = []
  base_trajs = []
  for traj in trajs:
    spec = traj.GetConfigurationSpecification()
    values_group = spec.GetGroupFromName('joint_values {0}'.format(robot.GetName()))
    dof = values_group.dof
    start_waypoint = traj.GetWaypoint(0).tolist() 
    start_pose = start_waypoint[values_group.offset+dof:values_group.offset+dof+3]
    end_waypoint = traj.GetWaypoint(traj.GetNumWaypoints()-1).tolist() 
    end_pose = end_waypoint[values_group.offset+dof:values_group.offset+dof+3]
    base_move_length = np.array(end_pose) - np.array(start_pose)
    # > retime traj if base does not move
    if np.isclose(np.linalg.norm(base_move_length), 0):
      retimer = esocstopp_retimer.OpenraveRetimer(robot, traj, timestep, velocity_limits, \
                                                  acceleration_limits, jerk_limits, 'scurve')
      [traj, success] = retimer.retime()
      # > OpenRAVE to ROS
      [arm_traj, base_traj] = ros_trajs_from_openrave_mobile_traj(robot.GetName(), traj)
      arm_trajs.append(arm_traj)
      base_trajs.append(base_traj)
    else:
      # > OpenRAVE to ROS
      [arm_traj, base_traj] = ros_trajs_from_openrave_affined_traj(robot.GetName(), traj, timestep)
      arm_trajs.append(arm_traj)
      base_trajs.append(base_traj)

  # Execute the output trajectories
  robot.SetActiveDOFValues(output["cgraph"].node[output["config_tour"][0]]['value'])
  raw_input("Press Enter to start the robot...\n")
  rospy.loginfo("Executing the output trajectories...")
  draws = []
  draw_ttour = np.array(output["task_tour"])[1:-1]-1
  max_traj_idx = len(output["trajs"])-1
  real_starttime = time.time()
  for i, traj in enumerate(trajs):
    arm_traj = arm_trajs[i]
    base_traj = base_trajs[i]
    # ROS: move base and arm
    WBC_trajectory.set_trajectory(arm_traj, base_traj)
    WBC_trajectory.start(delay=0.04)
    # OpenRAVE: move
    robot.GetController().SetPath(traj)
    # wait for all controllers to finish
    WBC_trajectory.wait(timeout=10.0)
    robot.WaitForController(0)
    # draw
    if (max_traj_idx-i):
      arrow_len = tsp_param.standoff
      tar_ray = targets.targets_ray[ (output["target_taskids"])[draw_ttour[i]] ]
      tar_ray = orpy.Ray(tar_ray.pos()-arrow_len*tar_ray.dir(), tar_ray.dir())
      draws.append( ru.visual.draw_ray(env=env, ray=tar_ray, dist=arrow_len, linewidth=1, \
                                       color=np.array([255, 0, 0])/255.) )
  output["real_exe_time"] = time.time() - real_starttime
  rospy.loginfo("Executed all trajectories in: real_exe_time = {} s".format(output["real_exe_time"]))


  # Clear and exit
  rospy.loginfo("Finished.")
  IPython.embed()
# END