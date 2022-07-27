# mobotscp
ROS package for MoboTSCP - Solution to the Mobile robot's Task Sequencing and Clustering Problem 
(TSCP) based on Travelling Salesman Problem (TSP) and Set Cover Problem (SCP)


## Getting started
### Prerequisites
* ROS Kinetic/Melodic
* Gazebo and ROS control packages
```
sudo apt-get install ros-$ROS_DISTRO-gazebo-ros-pkgs ros-$ROS_DISTRO-gazebo-ros-control
sudo apt-get install ros-$ROS_DISTRO-ros-control ros-$ROS_DISTRO-ros-controllers
```
* OpenRAVE 0.9.0 (mobipliant may not work with other versions of OpenRAVE)
```
# clone the repository
cd && git clone https://github.com/crigroup/openrave-installation.git
cd openrave-installation

# In Ubuntu 18.04, use the next line to check out an old commit that installs OpenRAVE 0.9.0
# Because latest commit will install OpenRAVE 0.9.0 for Ubuntu 16.04 but 0.53.1 for Ubuntu 18.04
git checkout b2766bd789e2432c4485dff189e75cf328f243ec

# install using scripts
./install-dependencies.sh -j4
./install-osg.sh -j4
./install-fcl.sh -j4
./install-openrave.sh -j4
cd && sudo rm -rf openrave-installation
```
* mayavi2: for visualization in Focus Kinematic Reachability solver
```
sudo apt install mayavi2
```

### Installation
Clone and build mobotscp using scripts:
```
cd ~/<catkin_workspace>/src
git clone https://github.com/nqnam1/mobotscp.git
cd mobotscp
./install.sh -w <catkin_workspace_name>
source ~/.bashrc
```
If `-w <catkin_workspace_name>` is not specified, the default is `catkin_ws`.


## Focus Kinematic Reachability (FKR)
The FKR file stores all points reachable by the end-effector with end-effector's orientation as 
described below. File name explained: TODO

The raw FKR data will then be analyzed to define a "reachable region" (relative to the robot) 
with analytical geometry. This reachable region is an important part of our compliant controller 
which minimizes the base movement.

FKR data is stored in `data/reachability/robot.<robot_id>/` for each robot. 
To use FKR data, please copy all `robot.*` and `kinematics.*` folders into `~/.openrave/`. 
If you generate new data, it will also be located inside `~/.openrave/`. 

The script `scripts/solve_fkr.py` demonstrates how to generate FKR, solve for reachability 
limits, visualize FKR and limits. Please make sure you have the `~/.openrave/` directory with 
the copied `robot.*` and `kinematics.*` folders inside before running: 
```
rosrun mobotscp solve_fkr.py
```
Some screenshots of the visualization of available FKR data & limits can be found in 
`data/figs/fkr` directory.


## Simulation demo
TODO


## Troubleshoot
* `c++: internal compiler error: Killed (program cc1plus)` when building the package: due to CPU 
and/or RAM overload during the `make` process. If building the package manually, try 
`catkin build -j <no_of_CPUs>` or `catkin make install -j <no_of_CPUs>`. If using the provided 
scripts, try running `./install.sh -n <no_of_CPUs>` with a smaller number of CPU processors. 

* `No module named <module_name>`: if there are libraries missing when running script files, 
firstly check whether all dependencies have been installed:
```
cd ~/<catkin_workspace>/src
rosdep update --include-eol-distros && \
sudo rosdep install --rosdistro $ROS_DISTRO --ignore-src --from-paths . -y
```

* `No module named <module_name>`: if all dependences have been installed but some python2 
libraries are still missing, you can install them manually by 
`pip install --no-deps <module_name>==<version>`. Sometimes the latest version cannot be 
installed, so you may need to search for the version that supports python 2.7.12 (in Ubuntu 
16.04) or 2.7.17 (in Ubuntu 18.04). The flag `--no-deps` is important especially in Ubuntu 16.04 
since its `pip`'s version is no longer supported and cannot be upgraded.

* Unable to load `VTK viewer for mayavi2`: you will not be able to visualize FKR & limits but, 
as long as you have installed mayavi2, `rosrun mobotscp solve_fkr.py` should still work to 
generate new FKR data or solve FKR data for reachability limits.

* `IOError: Unable to create file` when saving/loading .h5py/.pp files: please make sure you 
have permission to create/modify files into the desired directory (`~/.openrave` for example) 
and its subdirectories by `sudo chown -R <your_user_name> <directory_path>`.


## Maintainer
* [Quang-Nam Nguyen](mailto:quangnam.nguyen@ntu.edu.sg)
