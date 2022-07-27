cmake_minimum_required(VERSION 2.8.3)
project(mobotscp)

find_package(catkin REQUIRED COMPONENTS
  openrave_catkin
  raveutils
  robotsp
)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY data DESTINATION ${OpenRAVE_DEVEL_DIR})
install(DIRECTORY data DESTINATION ${OpenRAVE_INSTALL_DIR})