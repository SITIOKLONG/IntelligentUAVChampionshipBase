cmake_minimum_required(VERSION 3.0.2)
project(drone_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  airsim_ros
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf2
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS airsim_ros geometry_msgs nav_msgs roscpp sensor_msgs std_msgs tf2
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(waypoint_pwm_controller
  src/waypoint_pwm_controller.cpp
  src/PDcontroller.cpp
)
target_link_libraries(waypoint_pwm_controller ${catkin_LIBRARIES})

add_executable(waypoint_vel_controller
  src/waypoint_vel_controller.cpp
)
target_link_libraries(waypoint_vel_controller ${catkin_LIBRARIES})
