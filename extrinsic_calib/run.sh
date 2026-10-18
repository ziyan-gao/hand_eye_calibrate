#!/bin/bash
# One command: start the container (image is built only if missing), incrementally build the
# calibration workspace (seconds; source is mounted from the host), then launch.
# Extra args go to the launch file, e.g.  ./run.sh robot_ip:=192.168.1.234
# Force an image rebuild (only needed after editing docker/Dockerfile):  ./run.sh --rebuild
set -e
cd "$(dirname "$0")"
COMPOSE="docker compose -f docker/compose.yaml"
IMAGE=camera_extrinsic_calibration:jazzy

if [ "$1" = "--rebuild" ]; then
  shift
  $COMPOSE build
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  $COMPOSE build
fi
mkdir -p .colcon/build .colcon/install .colcon/log data
xhost +local:root >/dev/null
$COMPOSE up -d

exec docker exec -it camera_extrinsic_calibration bash -c "
  source /opt/ros/jazzy/setup.bash && source /opt/driver_ws/install/setup.bash &&
  cd /workspaces/calib_ws &&
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --event-handlers console_cohesion- status- > log/last_build.txt 2>&1 \
    || { tail -40 log/last_build.txt; echo 'colcon build failed'; exit 1; } &&
  source install/setup.bash &&
  ros2 launch camera_extrinsic_calibration_bringup calibration.launch.py $*"
