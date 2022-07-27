ROSDISTRO=$ROS_DISTRO
CATKINWS=catkin_ws
NPROC=4

# define options
while getopts "w:n:" arg; do
	case $arg in
		w) CATKINWS=$OPTARG;;
		n) NPROC=$OPTARG;;
	esac
done

echo "ROS distro: $ROSDISTRO"
echo "Workspace name: $CATKINWS"

cd /home/`id -un`/$CATKINWS && \
catkin init && \
	catkin config --extend /opt/ros/$ROSDISTRO && \
	catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && \
	catkin config --merge-devel

cd /home/`id -un`/$CATKINWS/src
wstool init .  && \
	wstool merge mobotscp/.rosinstall && \
	wstool update

sudo apt update && \
	rosdep update --include-eol-distros && \
	sudo rosdep install --rosdistro $ROSDISTRO --ignore-src --from-paths . -y

cd /home/`id -un`/$CATKINWS
catkin config --install && \
	catkin build -j$NPROC

if [[ ! -d "/home/`id -un`/.openrave" ]]
then
	mkdir /home/`id -un`/.openrave && \
		echo "Created new directory /home/`id -un`/.openrave"
fi

sudo cp -r /home/`id -un`/$CATKINWS/src/mobotscp/data/reachability/* /home/`id -un`/.openrave/ && \
	echo "Copied reachability data files to /home/`id -un`/.openrave"

echo "source /home/`id -un`/$CATKINWS/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc