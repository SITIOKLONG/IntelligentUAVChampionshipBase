#!/bin/bash
set -e

cd /drone_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash

# roslaunch --screen foxglove_bridge foxglove_bridge.launch &

roslaunch my_drone my_drone.launch
