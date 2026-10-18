#!/bin/bash
# Write the dependency lock record (spec 3.1) to stdout.
source /opt/ros/jazzy/setup.bash
echo "# generated $(date -Iseconds)"
echo "base_image: ${ROS_IMAGE_DIGEST:-unknown (pass --build-arg / record docker image digest)}"
echo "xarm_ros2: $(git -C /opt/driver_ws/src/xarm_ros2 rev-parse HEAD)"
git -C /opt/driver_ws/src/xarm_ros2 submodule status --recursive | sed 's/^/  submodule: /'
echo "apt:"
dpkg-query -W -f='  ${Package}: ${Version}\n' 'ros-jazzy-realsense2-*' 'ros-jazzy-librealsense2' \
  'ros-jazzy-rviz2' 'ros-jazzy-rviz-common' 'ros-jazzy-cv-bridge' 'python3-opencv' \
  'python3-numpy' 'qtbase5-dev' 2>/dev/null
python3 -c "import cv2, numpy; print(f'python_opencv: {cv2.__version__}\nnumpy: {numpy.__version__}')"
