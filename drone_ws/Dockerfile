# syntax=docker/dockerfile:1.6

# ============ Stage 1: Builder ============
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1-latest.list && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopencv-dev net-tools \
    python3-catkin-pkg python3-catkin-tools python3-empy \
    python3-pip python3-numpy python3-rospkg python3-yaml \
    ros-noetic-ros-base ros-noetic-cv-bridge ros-noetic-geographic-msgs \
    ros-noetic-image-transport ros-noetic-tf2-geometry-msgs ros-noetic-tf2-sensor-msgs && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --extra-index-url https://download.pytorch.org/whl/cu118 \
    numpy==1.24.4 typing_extensions==4.13.2 open3d==0.13.0 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 ultralytics==8.2.103

ADD third_party/tensorrt/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz /opt/
RUN ln -s /opt/TensorRT-8.6.1.6 /opt/tensorrt && \
    pip install /opt/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl && \
    find /opt/TensorRT-8.6.1.6 -name '*.a' -delete && \
    rm -rf /opt/TensorRT-8.6.1.6/data /opt/TensorRT-8.6.1.6/samples \
           /opt/TensorRT-8.6.1.6/doc /opt/TensorRT-8.6.1.6/include

ENV LD_LIBRARY_PATH=/opt/tensorrt/targets/x86_64-linux-gnu/lib:/opt/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/opt/tensorrt/targets/x86_64-linux-gnu/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /drone_ws/

# Keep package manifests in their own cache layer. During active development,
# run with src bind-mounted and rebuild incrementally inside the container.
COPY src/airsim_ros/package.xml src/airsim_ros/CMakeLists.txt /drone_ws/src/airsim_ros/
COPY src/debug/package.xml src/debug/CMakeLists.txt /drone_ws/src/debug/
COPY src/door_waypoint_detector/package.xml src/door_waypoint_detector/CMakeLists.txt /drone_ws/src/door_waypoint_detector/
COPY src/drone_control/package.xml src/drone_control/CMakeLists.txt /drone_ws/src/drone_control/
COPY src/imu_gps_odometry/package.xml src/imu_gps_odometry/CMakeLists.txt /drone_ws/src/imu_gps_odometry/
COPY src/my_drone/package.xml src/my_drone/CMakeLists.txt /drone_ws/src/my_drone/
COPY src/navigation_vision/package.xml /drone_ws/src/navigation_vision/
COPY src/stereo_depth/package.xml src/stereo_depth/CMakeLists.txt /drone_ws/src/stereo_depth/

COPY src /drone_ws/src/
ADD setup.bash /
RUN ln -sfn /opt/ros/${ROS_DISTRO}/share/catkin/cmake/toplevel.cmake /drone_ws/src/CMakeLists.txt && \
    chmod +x /setup.bash

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    catkin_make --only-pkg-with-deps airsim_ros && \
    . devel/setup.sh && \
    catkin_make --only-pkg-with-deps imu_gps_odometry my_drone drone_control debug stereo_depth door_waypoint_detector && \
    rm -rf build

# ============ Stage 2: Runtime ============
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg2 lsb-release net-tools && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1-latest.list && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-numpy python3-rospkg python3-yaml \
    ros-noetic-ros-base ros-noetic-cv-bridge ros-noetic-geographic-msgs \
    ros-noetic-image-transport ros-noetic-tf2-geometry-msgs ros-noetic-tf2-sensor-msgs \
    libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --extra-index-url https://download.pytorch.org/whl/cu118 \
    numpy==1.24.4 typing_extensions==4.13.2 open3d==0.13.0 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 ultralytics==8.2.103

COPY --from=builder /opt/TensorRT-8.6.1.6 /opt/TensorRT-8.6.1.6
RUN ln -s /opt/TensorRT-8.6.1.6 /opt/tensorrt && \
    pip install /opt/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl

COPY --from=builder /drone_ws /drone_ws
COPY --from=builder /setup.bash /

ENV LD_LIBRARY_PATH=/opt/tensorrt/targets/x86_64-linux-gnu/lib:/opt/tensorrt/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/opt/tensorrt/targets/x86_64-linux-gnu/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN chmod +x /setup.bash

WORKDIR /drone_ws/
ENTRYPOINT [ "/setup.bash" ]
