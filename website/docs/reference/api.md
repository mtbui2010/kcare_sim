# API: ROS 2 interfaces and kcare skills

## ROS 2 (served by `house_bridge.py`)

`ROS_DOMAIN_ID=77`. Names and message types are the real robot's.

### Topics published

| topic | type | content |
|---|---|---|
| `/joint_states` | sensor_msgs/JointState | lift, head pan, head tilt, joint1..7, virtual_gripper_joint |
| `/femto/color/image_raw` | sensor_msgs/Image (rgb8) | head camera, 1280×720 |
| `/femto/depth/aligned` | sensor_msgs/Image (16UC1, mm) | head depth registered to colour |
| `/femto/color/camera_info`, `/femto/depth/camera_info` | sensor_msgs/CameraInfo | intrinsics + distortion of the real camera |
| `/hand/camera/color/image_raw/compressed` | sensor_msgs/CompressedImage (JPEG) | wrist camera, 848×480 |
| `/hand/camera/depth/image_rect_raw`, `/hand/camera/depth/camera_info` | Image (16UC1, mm), CameraInfo | wrist depth |
| `/robot_pose/mobile_pose`, `mobile_base_tool_pose`, `arm_base_tool_pose`, `map_base_tool_pose` | geometry_msgs/PoseStamped | base and tool poses |
| `/sim/overview/image_raw/compressed` | CompressedImage | follow camera (viewer only) |
| `/sim/topdown/image_raw/compressed` | CompressedImage | fixed camera over the house (viewer only) |
| `/sim/drawer_states` | JointState | opening of every drawer [m] — ground truth for evaluation only |

### Topics subscribed

| topic | type | content |
|---|---|---|
| `/sim/arm_servo` | JointState | joint1..7 + virtual_gripper_joint targets, applied straight to the servos (policy control at 30 Hz) |

### Actions

| action | type | behaviour in the sim |
|---|---|---|
| `/navigate_to_pose` | nav2_msgs/NavigateToPose | A* on the occupancy grid; aborts with `no path` when unreachable |
| `/kaair_worker/arm_moveJ` | kaair_msgs/MoveJoint | joint-space move, `velocity_scale` × 0.9 rad/s |
| `/kaair_worker/arm_moveL`, `arm_moveT` | kaair_msgs/MoveLinear, MoveTool | IK to the target, then a joint move |
| `/kaair_worker/lift_move`, `head_move` | kaair_msgs/LiftMove, HeadMove | servo moves (lift 0.08 m/s, head 1 rad/s) |
| `/body/tool_controller/gripper_cmd` | control_msgs/GripperCommand | gripper width |
| `/drawer/open` (VLA node) | max_interfaces/OpenDrawer: goal `mode, prompt` → result `success, message` | ready pose, then the policy for 15 s |

### Services

| service | type |
|---|---|
| `/mobile/shift_pose`, `/mobile/rotate`, `/mobile/absolute_rotate` | kaair_msgs MobileShift / MobileRotate (refused when blocked) |
| `/create_object_marker`, `/create_tool_marker` | kaair_msgs/CreateObjectMarker: pixels (+ depths) → 3-D points in `base_footprint` or `map` |
| `/arm/init_set` | std_srvs/Trigger |

## kcare skills used here (HTTP, `POST :8001/skill/<name>`, body `{"inputs": "..."}`)

| skill | inputs | effect |
|---|---|---|
| `move` | an ENV place, e.g. `counter@kitchen`, or `home` | navigate, then turn to the place's heading |
| `find` | `object@place` | open-vocabulary detection in the head image + 3-D position |
| `moveh`, `lift`, `movej` | kcare presets | head, lift, arm |
| `vla_drawer_open` | prompt, e.g. `open the top drawer` | calls `/drawer/open` |
| `mobile_pose` | – | current base pose |
