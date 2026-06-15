#!/bin/bash
cd /basic_dev
source devel/setup.bash

# rosrun basic_dev basic_dev

#debug
source /opt/ros/noetic/setup.bash
source /basic_dev/devel/setup.bash
roslaunch --screen foxglove_bridge foxglove_bridge.launch &

roslaunch my_drone my_drone.launch