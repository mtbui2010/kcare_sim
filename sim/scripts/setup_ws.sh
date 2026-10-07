#!/bin/bash
# Run INSIDE the kaair_sim container, once.
# Imports the third-party repos (xarm_ros2, OrbbecSDK_ROS2, realsense-ros) with
# the repo's own script, then builds only what `xacro robot.urdf.xacro` needs:
# the description packages. The hw spec yaml (kaair_bringup/config/robots) is
# read from the source tree, so kaair_bringup and its driver deps are not
# built — the sim never talks to hardware.
set -eo pipefail
source /opt/ros/humble/setup.bash
cd /workspace

if [ ! -d ros2/xarm_ws/src/xarm_ros2 ]; then
    bash scripts/import_third_party.sh -t humble
fi

(cd ros2/xarm_ws   && colcon build --symlink-install --packages-select xarm_description)
(cd ros2/camera_ws && colcon build --symlink-install --packages-up-to orbbec_description realsense2_description)
source ros2/xarm_ws/install/setup.bash
source ros2/camera_ws/install/setup.bash
(cd ros2/ros2_ws   && colcon build --symlink-install \
    --packages-select former_description kaair_description)
echo "✅ workspace ready — open a new shell (or: source ~/.bashrc)"
