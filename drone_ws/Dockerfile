FROM osrf/ros:noetic-desktop-full-focal

ADD src /drone_ws/src/
ADD setup.bash /

RUN chmod +x /setup.bash

USER root

RUN apt update && apt install -y \
    libopencv-dev \
    net-tools \
    python3-catkin-tools \
    python3-pip \
    ros-noetic-cv-bridge \
    ros-noetic-geographic-msgs \
    ros-noetic-image-transport \
    # ros-noetic-foxglove-bridge \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-sensor-msgs && \
    rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=noetic

WORKDIR /drone_ws/
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    catkin_make --only-pkg-with-deps airsim_ros && \
    . devel/setup.sh && \
    catkin_make --only-pkg-with-deps imu_gps_odometry my_drone debug stereo_depth

ENTRYPOINT [ "/setup.bash" ]
