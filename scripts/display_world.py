#!/usr/bin/env python
import openravepy
import IPython
import time
import numpy as np
import tf.transformations as tr

##############################################################################################################
# Display the simulation setup according to the .xml file set in line 15
##############################################################################################################


if __name__ == "__main__":
  env = openravepy.Environment()
  env.Load('worlds/wing_drilling_task.env.xml')
  #env.SetViewer('qtcoin')
  env.SetDefaultViewer()
  while env.GetViewer() is None:
    time.sleep(0.1)
  Tcamera = tr.euler_matrix(*np.deg2rad([-147, 0, 180]))
  Tcamera[:3,3] = [-0.25, 1.8, 3.2]
  viewer = env.GetViewer()
  viewer.SetCamera(Tcamera)
  viewer.SetBkgndColor([.8, .85, .9])

  IPython.embed()