cmake_minimum_required(VERSION 3.0.2)
project(stereo_depth)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_filters
  roscpp
  sensor_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(stereo_depth_node src/stereo_depth_node.cpp)
target_link_libraries(stereo_depth_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS stereo_depth_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
