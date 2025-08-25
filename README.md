# mobotscp
MoboTSCP - ROS package for "[Task-Space Clustering for Mobile Manipulator Task Sequencing](https://doi.org/10.1109/ICRA48891.2023.10161293)"

MoboTSCP is a task sequencing planner for mobile manipulators. 
For fixed-base robots, please try [RoboTSP](https://github.com/crigroup/robotsp.git) instead.

Citation:
```
@inproceedings{nguyen2023task,
  title={Task-Space Clustering for Mobile Manipulator Task Sequencing},
  author={Nguyen, Quang-Nam and Adrian, Nicholas and Pham, Quang-Cuong},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3693--3699},
  year={2023},
  organization={IEEE}
}
```


## Getting started
### Prerequisites
Install ROS and OpenRAVE that are compatible to each other: the below have been tested 
(for newer versions some debugging efforts are expected)
* ROS Kinetic/Melodic in Ubuntu 16.04/18.04: follow ROS documentation
* Gazebo and ROS control packages
```
sudo apt-get install ros-$ROS_DISTRO-gazebo-ros-pkgs ros-$ROS_DISTRO-gazebo-ros-control
sudo apt-get install ros-$ROS_DISTRO-ros-control ros-$ROS_DISTRO-ros-controllers
```
* OpenRAVE 0.9.0: follow CRI Group's [instructions](https://github.com/crigroup/openrave-installation.git)
```
# clone the repository
cd && git clone https://github.com/crigroup/openrave-installation.git
cd openrave-installation

# in Ubuntu 18.04 ONLY, use the next line to check out an old commit for OpenRAVE 0.9.0
git checkout b2766bd789e2432c4485dff189e75cf328f243ec

# install using scripts
./install-dependencies.sh -j4
./install-osg.sh -j4
./install-fcl.sh -j4
./install-openrave.sh -j4
cd && sudo rm -rf openrave-installation
```

Some dependencies must be installed manually:
* [SetCoverPy](https://github.com/guangtunbenzhu/SetCoverPy): (optional) a python SCP solver
```
pip install SetCoverPy
```
* [mayavi2](https://docs.enthought.com/mayavi/mayavi/overview.html): data visualization
```
sudo apt install mayavi2
```

Other dependencies are specified in `.rosinstall` file and will be installed using wstool 
during the installation step below.


### Installation
After installing all prerequisites above, clone and install MoboTSCP using the provided script: 
```
cd ~/<your_catkin_ws>/src
git clone https://github.com/quangnamng/mobotscp.git
cd mobotscp
./install.sh
cd && source .bashrc
```
Note: if your ROS workspace directory is not named `catkin_ws`, try `./install.sh -w <your_catkin_ws>`.


### Testing the installation
Display the setup of our demo task in OpenRAVE environment:
```
rosrun mobotscp display_task.py
```


## Demo
Task: mobile manipulator (Denso VS087 arm mounted on Clearpath Ridgeback base) 
drilling 336 targets on the surface of a mock wing. 

Solution: computed clusters are visualized by different colors, and the poses for the base to 
visit each cluster are shown in corresponding colors.

Make sure `robot.*` and `kinematics.*` folders from `data/reachability/` 
have been copied into `~/.openrave/` directory. 
This has been done if you installed mobotscp by script, otherwise please copy those folders manually. 

For simulation in OpenRAVE environment, simply run:
```
rosrun mobotscp OpenRAVE_drilling_task.py
```


## ROS + Gazebo/hardware
We follow the plug-and-play concept: 
* mobotscp plays as a solver to the Mobile Manipulator Task Sequencing. 
* to implement mobotscp on a robot, such as in Gazebo simulation or hardware experiment, 
robot models and Gazebo/hardware supporting files should be stored in a separate package.  

Gazebo simulation: requires [denso_ridgeback](https://github.com/nqnam1/denso_ridgeback.git)
```
# Terminal 1:
roslaunch mobotscp drilling_task_gazebo.launch

# Terminal 2:
roslaunch mobotscp drilling_task_controllers.launch

# Terminal 3:
rosrun mobotscp ROS_drilling_task.launch
```

Hardware experiment: refer to [denso_ridgeback](https://github.com/nqnam1/denso_ridgeback.git).


## Kinematic Reachability database
The reachability data are saved in .pp files which store all points reachable by the robot's 
end-effector at some orientations. The raw reachability data will then be analyzed to define a 
"reachable region" relative to the robot with analytical geometry called "reachability limits." 

File name: e.g. `fkr.mobile_manipulator_drill_110-150deg<auto_ID_number>.pp`
* `mobile_manipulator_drill`: robot's name
* `110-150deg`: orientation of the end-effector (in this case, the drill tip), i.e. polar angle 
ranges from 110 to 150 degrees, whereas azimuthal angle is fixed at 0 deg by default.

Reachability data are stored in `data/reachability/robot.<robot_id>/` for each robot. 
If you installed mobotscp using provided script, `robot.*` and `kinematics.*` folders from 
`data/reachability/` have already been copied into `~/.openrave/`. 
Newly generated data will also be located in `~/.openrave/`. 
The script `scripts/generate_reachdata.py` demonstrates how to generate reachability database, 
analyze database for reachability limits, and visualization:
```
rosrun mobotscp generate_reachdata.py
```
Some screenshots of the visualization of FKR data & limits can be found in `data/figs`.


## Troubleshoot
* `No module named <module_name>`: there are modules/libraries missing when running script files, 
firstly check whether all dependencies have been installed:
```
cd ~/<catkin_workspace>/src
rosdep update --include-eol-distros && \
sudo rosdep install --rosdistro $ROS_DISTRO --ignore-src --from-paths . -y
```

* `No module named <module_name>`: if all dependences have been installed but some python 
modules/libraries are still missing, install them manually by `pip install <module_name>`. 
If the latest version cannot be installed, search for the version that supports Python 2.7.12 
(in Ubuntu 16.04) or 2.7.17 (in Ubuntu 18.04). You may try `pip install --no-deps` when 
`pip install` fails especially in Ubuntu 16.04 since its `pip`'s version is no longer supported.

* `Imported VTK version (8.1) does not match the one`: unable to load VTK viewer for mayavi2, 
you may not view the FKR visualization but it will not affect the codes that generate/analyze FKR.

* `IOError: Unable to create file` when saving/loading .h5py/.pp files: please make sure you 
have permission to create/modify files into the desired directory (`~/.openrave` for example) 
and its subdirectories by `sudo chown -R <user_name> <directory_path>`.


## Maintainer
* [Quang-Nam Nguyen](mailto:namnguyen@nyu.edu)
