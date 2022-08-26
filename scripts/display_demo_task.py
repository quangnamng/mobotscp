#!/usr/bin/env python
#MoboTSCP
import mobotscp as mtscp
#utils
import IPython
import numpy as np
import openravepy as orpy
import raveutils as ru
import time
import tf.transformations as tr


if __name__ == "__main__":
  ### OpenRAVE Environment Setup
  # Setup world
  env = orpy.Environment()
  world_xml = 'worlds/wing_drilling_task.env.xml'
  if not env.Load(world_xml):
    print('Failed to load: {0}'.format(world_xml))
    raise IOError
  print('Loaded OpenRAVE environment: {}'.format(world_xml))
  orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

  # Setup robot and manipulator
  robot = env.GetRobot('robot')
  manip = robot.SetActiveManipulator('drill')
  robot.SetActiveDOFs(manip.GetArmIndices(), \
                      robot.DOFAffine.X|robot.DOFAffine.Y|robot.DOFAffine.RotationAxis,[0,0,1])
  # > home config
  base_home = [-1.0, 0., 0.]   # (x, y, yaw)
  arm_home = np.deg2rad([0, -20, 130, 0, 70, 0])
  qhome = np.append(arm_home, base_home)
  with env:
    robot.SetActiveDOFValues(qhome)
    Thome = robot.GetTransform()
    phome = manip.GetEndEffectorTransform()[:3,3]


  ### Task Definition
  # Register the targets
  wing = env.GetKinBody('wing')
  max_num_targets = 336   # the wing has 288 targets on front side and 48 targets on back side
  # > use the 2nd line below to add azimuthal angles to the targets, otherwise comment it
  azimuths = [0]*max_num_targets
  azimuths = [np.deg2rad(46-(i%24)*4) for i in range(288)] + \
              [np.deg2rad((i%8)*8-28) for i in range(288,max_num_targets)]
  # > register the first 'max_num_targets' targets
  targets = mtscp.utils.RegisterTargets(links=wing.GetLinks(), targets_name='hole', \
                                        max_targets=max_num_targets, add_azimuth=azimuths)
  print("Number of targets: {}".format(len(targets.targets_ray)))
  targets_theta = np.arccos(targets.targets_array[:,-1])
  theta_deg_min = np.rad2deg(min(targets_theta))
  theta_deg_max = np.rad2deg(max(targets_theta))
  print("Range of targets' polar angles: {}-{} deg".format(theta_deg_min, theta_deg_max))


  ### Visualization
  # Setup viewer
  env.SetDefaultViewer()
  while env.GetViewer() is None:
    time.sleep(0.1)
  Tcamera = tr.euler_matrix(*np.deg2rad([-145, 0, 160]))
  Tcamera[:3,3] = [0.5, 2.5, 4]
  viewer = env.GetViewer()
  viewer.SetCamera(Tcamera)
  viewer.SetBkgndColor([1, 1, 1])

  # Visualize targets
  arrows = []
  for i in range(len(targets.targets_ray)):
    arrow_len = 0.05
    ray = targets.targets_ray[i]
    ray = orpy.Ray(ray.pos()-arrow_len*ray.dir(), ray.dir())
    arrows.append( ru.visual.draw_ray(env, ray, arrow_len, linewidth=0., color=np.array([255,0,0])/255.) )


  # Clear and exit
  IPython.embed()
# END