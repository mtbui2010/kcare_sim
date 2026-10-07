#!/bin/bash
# Expand the kaair xacro into plain URDFs, one per mobile-base variant, exactly
# as the robot's own bringup does (mode=robot, include_ros2_control=false —
# the same mappings vla_moveit_bringup.launch.py passes to MoveIt).
#
#   slamtec : kaair_description/urdf/robot.urdf.xacro        (specs_02/03/dure)
#   clober  : kaair_description/urdf/clober_robot.urdf.xacro (specs_01)
#
# Output: /sim/urdf/raw/kaair_<variant>.urdf  (package:// URIs left as-is)
set -eo pipefail
source ~/.bashrc >/dev/null 2>&1 || true
DESC=$(ros2 pkg prefix kaair_description)/share/kaair_description/urdf
SPECS=/workspace/ros2/ros2_ws/src/kaair_bringup/config/robots   # source tree: kaair_bringup is not built
OUT=/sim/urdf/raw
mkdir -p "$OUT"

gen() {  # variant xacro spec
    xacro "$DESC/$2" mode:=robot use_fake_hardware:=true include_ros2_control:=false \
        hw_spec_file:="$SPECS/$3" > "$OUT/kaair_$1.urdf"
    echo "  $OUT/kaair_$1.urdf  ($(grep -c '<link' "$OUT/kaair_$1.urdf") links, $(grep -c '<joint' "$OUT/kaair_$1.urdf") joints)"
}
gen slamtec robot.urdf.xacro        kaair_specs_02.yaml
gen clober  clober_robot.urdf.xacro kaair_specs_01.yaml
