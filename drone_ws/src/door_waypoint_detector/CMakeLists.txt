cmake_minimum_required(VERSION 3.0.2)
project(door_waypoint_detector)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/door_waypoint_detector.py
  scripts/door_fusion_tracker.py
  scripts/pointcloud_preprocessor.py
  scripts/yolo_door_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
