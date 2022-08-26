# MoboTSCP: Mobile Robot's Task Sequencing and Clustering Problem
MoboTSCP is a ROS package for solving the Mobile Robot's Task Sequencing and Clustering Problem.


## Getting started
### Prerequisites
* ROS Kinetic/Melodic
* Gazebo and ROS control packages
```
sudo apt-get install ros-$ROS_DISTRO-gazebo-ros-pkgs ros-$ROS_DISTRO-gazebo-ros-control
sudo apt-get install ros-$ROS_DISTRO-ros-control ros-$ROS_DISTRO-ros-controllers
```
* OpenRAVE 0.9.0 (MoboTSCP may not work with other versions)
```
# clone the repository
cd && git clone https://github.com/crigroup/openrave-installation.git
cd openrave-installation

# In Ubuntu 18.04, use the next line to check out an old commit for OpenRAVE 0.9.0
git checkout b2766bd789e2432c4485dff189e75cf328f243ec

# install using scripts
./install-dependencies.sh -j4
./install-osg.sh -j4
./install-fcl.sh -j4
./install-openrave.sh -j4
cd && sudo rm -rf openrave-installation
```

Some dependencies must be installed manually:
* [SetCoverPy](https://github.com/guangtunbenzhu/SetCoverPy): a python SCP solver
```
pip install SetCoverPy
```
* [mayavi2](https://docs.enthought.com/mayavi/mayavi/overview.html): for data visualization
```
sudo apt install mayavi2
```
Other dependencies are specified in `.rosinstall` file and will be installed using wstool 
during the installation step below.

### Installation
After installing all prerequisites above, clone and install MoboTSCP using the provided script: 
```
cd ~/<your_catkin_ws>/src
git clone https://github.com/nqnam1/mobotscp.git
cd mobotscp
./install.sh
cd && source .bashrc
```
Note: if your ROS workspace directory is not named `catkin_ws`, try `./install.sh -w <your_catkin_ws>`.

### Testing the installation
Display the setup of our demo task in OpenRAVE environment:
```
rosrun mobotscp display_demo_task.py
```


## Demo
Firstly, make sure `robot.*` and `kinematics.*` folders from `mobotscp/data/reachability/` 
are copied into `~/.openrave/` directory. This has been done if you installed MoboTSCP by script, 
otherwise please copy them manually. 

The task is for our mobile manipulator (Denso VS087 arm mounted on Clearpath Ridgeback base) 
drilling 336 targets on a mock wing. 

To solve the task and play the simulation, run: 
```
rosrun mobotscp MoboTSCP_wing_drilling_demo.py
```
Computed clusters are visualized by different colors, and the poses for the base to visit each 
cluster are also shown in corresponding colors, with arrows representing the base sequence.


## Focused Kinematic Reachability (FKR)
The FKR data are saved in .pp files which store all points reachable by the robot's end-effector 
at some orientations as described below. The raw FKR data will then be analyzed to define a "
reachable region" relative to the robot with analytical geometry called "reachability limits." 

File name: e.g. `fkr.mobile_manipulator_drill_110-150deg<auto_ID_number>.pp`
* `mobile_manipulator_drill`: robot's name
* `110-150deg`: orientation of the end-effector (in this case, the drill tip), i.e. polar angle 
ranges from 110 to 150 degrees, whereas azimuthal angle is fixed at 0 deg by default.

FKR data are stored in `data/reachability/robot.<robot_id>/` for each robot. If you installed 
MoboTSCP using provided script, `robot.*` and `kinematics.*` folders from `data/reachability/` 
have already been copied into `~/.openrave/`. Newly generated data will also be located in 
`~/.openrave/`. The script `scripts/generate_fkr.py` demonstrates how to generate FKR, analyze 
FKR data for reachability limits, and visualization:
```
rosrun mobotscp generate_FKR.py
```
Some screenshots of the visualization of FKR data & limits can be found in `data/figs`.


## Troubleshoot
* `c++: internal compiler error: Killed (program cc1plus)` when building the package: due to 
CPU or RAM overload during the `make` process, especially if you use virtual machines or WSL. 
If using the provided scripts, try `./install.sh -n <num_CPUs>` with a small number of CPUs. 
If building manually, try `catkin build -j <num_CPUs>` or `catkin make install -j <num_CPUs>`. 

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
* [Quang-Nam Nguyen](mailto:quangnam.nguyen@ntu.edu.sg)
