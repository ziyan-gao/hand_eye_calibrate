#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /opt/driver_ws/install/setup.bash
if [ -f /workspaces/calib_ws/install/setup.bash ]; then
  source /workspaces/calib_ws/install/setup.bash
fi
exec "$@"
