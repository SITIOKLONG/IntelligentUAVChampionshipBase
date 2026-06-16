cmake_minimum_required(VERSION 3.0.2)
project(airsim_ros)

find_package(catkin REQUIRED COMPONENTS
  geographic_msgs
  geometry_msgs
  message_generation
  roscpp
  rospy
  std_msgs
)

add_message_files(
  FILES
  Altimeter.msg
  CarControls.msg
  CarState.msg
  Environment.msg
  GimbalAngleEulerCmd.msg
  GimbalAngleQuatCmd.msg
  GPSYaw.msg
  PoseCmd.msg
  RotorPWM.msg
  VelCmd.msg
  VelCmdGroup.msg
)

add_service_files(
  FILES
  Land.srv
  LandGroup.srv
  Reset.srv
  SetGPSPosition.srv
  SetLocalPosition.srv
  Takeoff.srv
  TakeoffGroup.srv
)

generate_messages(
  DEPENDENCIES
  geographic_msgs
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  geographic_msgs
  geometry_msgs
  message_runtime
  roscpp
  rospy
  std_msgs
)
