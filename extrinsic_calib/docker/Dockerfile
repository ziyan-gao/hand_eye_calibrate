# Camera extrinsic calibration dev container: Ubuntu 24.04 + ROS 2 Jazzy,
# xarm_ros2 (jazzy) + realsense-ros + calibration backend + RViz2 panel.
ARG ROS_IMAGE=ros:jazzy-ros-base-noble
FROM ${ROS_IMAGE}

ARG XARM_REPO=https://github.com/xArm-Developer/xarm_ros2.git
ARG XARM_BRANCH=jazzy
# Pin a reviewed commit for reproducible builds (spec 3.1). Empty = branch head (record it!)
ARG XARM_COMMIT=3dc2b5e8294758d96b54b15fa5920d581b7cbb3d

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# One librealsense source: the ROS apt packages (ros-jazzy-librealsense2).
# OpenCV: Ubuntu python3-opencv, the one cv_bridge is built against.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake \
      python3-colcon-common-extensions python3-rosdep python3-pytest \
      python3-numpy python3-opencv python3-yaml \
      qtbase5-dev \
      ros-jazzy-rviz2 ros-jazzy-rviz-common ros-jazzy-rviz-default-plugins \
      ros-jazzy-pluginlib ros-jazzy-rclcpp-action ros-jazzy-cv-bridge \
      ros-jazzy-tf2-ros ros-jazzy-image-transport \
      ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher ros-jazzy-xacro \
      ros-jazzy-realsense2-camera ros-jazzy-librealsense2 \
      mesa-utils libgl1 libegl1 usbutils iputils-ping \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true; rosdep update --rosdistro jazzy

# ---- driver underlay: only xarm_api and its dependency closure (no MoveIt/Gazebo)
WORKDIR /opt/driver_ws
RUN git clone -b ${XARM_BRANCH} ${XARM_REPO} src/xarm_ros2 \
    && cd src/xarm_ros2 \
    && if [ -n "${XARM_COMMIT}" ]; then git checkout ${XARM_COMMIT}; fi \
    && git submodule sync && git submodule update --init --recursive
RUN . /opt/ros/jazzy/setup.sh \
    && PKGS=$(colcon list --packages-up-to xarm_api xarm_description uf_ros_lib --paths-only | tr '\n' ' ') \
    && apt-get update \
    && rosdep install -y --rosdistro jazzy --ignore-src --from-paths ${PKGS} \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build --packages-up-to xarm_api uf_ros_lib --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log

# ---- calibration overlay: NOT built into the image. src/ and the colcon build/install/log
# dirs are bind-mounted from the host (compose.yaml); run.sh builds incrementally at start,
# so code changes never require an image rebuild. Only the dependencies live here.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-std-srvs ros-jazzy-geometry-msgs ros-jazzy-sensor-msgs \
      ros-jazzy-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspaces/calib_ws

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/record_versions.sh /usr/local/bin/record_versions.sh
RUN chmod +x /entrypoint.sh /usr/local/bin/record_versions.sh \
    && mkdir -p /data/calibration \
    && printf '%s\n' \
       'source /opt/ros/jazzy/setup.bash' \
       'source /opt/driver_ws/install/setup.bash' \
       '[ -f /workspaces/calib_ws/install/setup.bash ] && source /workspaces/calib_ws/install/setup.bash' \
       >> /root/.bashrc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
