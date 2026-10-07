# 1. Running the robot's own software in a simulator

!!! abstract "In this chapter"
    - A robot's software talks to its hardware through a fixed set of messages; reproduce those and the software cannot tell the difference.
    - kcare-sim reproduces 6 actions, 5 services and the camera / state topics kcare uses.
    - What it does **not** reproduce is listed — that is where sim results can mislead.

## The idea

kcare's skills (`move`, `find`, `movej`, …) never touch motors. They send ROS 2 goals to
action servers (`/navigate_to_pose`, `/kaair_worker/arm_moveJ`, …), call services
(`/mobile/rotate`, `/create_object_marker`) and read topics (`/joint_states`, the cameras). A
simulator that serves the same names with the same message types is, to kcare, the robot.

| kcare uses | real robot | kcare-sim |
|---|---|---|
| `/navigate_to_pose` (nav2) | Slamtec base + nav stack | A* on the house occupancy grid, kinematic base |
| `/kaair_worker/arm_moveJ`, `moveL`, `moveT` | xArm7 controller | MuJoCo position servos; MoveL/MoveT via IK |
| `lift_move`, `head_move`, `gripper_cmd` | lift, pan-tilt, gripper drivers | MuJoCo servos |
| head + wrist images, depth, camera_info | Femto Bolt, D405 drivers | MuJoCo renders + camera / depth models |
| `/create_object_marker` | pixel + depth → 3-D point | same, with the simulated depth |

!!! code "Where in the code"
    The interface list is the docstring of [`house_bridge.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_bridge.py); kcare is
    pointed at it by [`make_kcare_sim_location.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/make_kcare_sim_location.py), which
    rewrites every connection to 127.0.0.1.

!!! measured "Measured"
    The same kcare code base, unmodified, drove the simulated robot to 8 of 8 places in the house
    and found 7 of 8 objects (tour test, 7 Oct 2026).

## What this means when you use it

- A bug found in the sim is a bug in the real code path (two were: a shared action-result slot and
  numpy values in HTTP replies).
- Navigation is idealised: no wheel slip, no localisation drift. Do not read navigation success
  here as a real-world number.
- kcare does not check the navigation result; the bridge log does (`no path to …`).

**Next:** [2. A scene from a floor plan and photos →](02-scene-from-photos.md)
