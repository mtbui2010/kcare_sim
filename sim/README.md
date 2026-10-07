# kaair_sim — kaair URDF → physics simulation (step 1 of the VLA plan)

Turns `kaair_description/urdf/{robot,clober_robot}.urdf.xacro` (the kinematic
description the real robot's MoveIt uses) into a physics model that is ready
for simulation, and verifies it.

```
~/trung_workdir/kaair_sim/
├── keti_assist_ai_robot_ros2/   copy of the robot's ROS 2 repo  → /workspace in the container
└── sim/                         this directory                  → /sim
    ├── docker/Dockerfile.sim    kaair-moveit:humble + mujoco, trimesh, coacd
    ├── docker-compose.yml       container kaair_sim_humble (ROS_DOMAIN_ID=77, no /dev)
    ├── config/sim_urdf.yaml     masses, collision strategy, joint fixes
    ├── config/mjcf.yaml         actuators, gains, cameras, gripper, contact excludes
    ├── scripts/                 pipeline (below) + debug/
    ├── urdf/raw/                xacro output, untouched
    ├── urdf/kaair_<v>_sim.urdf  physics-ready URDF (self-contained, meshes in assets/)
    ├── assets/<v>/              meshes baked to metres + convex collision parts
    ├── mjcf/                    MuJoCo robot + tabletop scene
    └── out/<v>/                 check report.json + rendered images
```

## Running it

```bash
# host (A6000)
cd ~/trung_workdir/kaair_sim/sim
make image        # kaair-moveit:humble (repo's docker/Dockerfile) + kaair-sim:humble
make up           # start kaair_sim_humble
make shell

# container (cwd /sim)
make ws           # once: vcs import xarm_ros2/Orbbec/realsense, build description pkgs
make all VARIANT=slamtec   # urdf -> sim-urdf -> mjcf -> check
make all VARIANT=clober
```

`slamtec` = `robot.urdf.xacro` (specs_02/03/dure); `clober` =
`clober_robot.urdf.xacro` (specs_01). Only the base and the lift/head mount
heights differ.

## What the pipeline changes

| Gap in the kinematic URDF | Fix |
|---|---|
| 44 links without `<inertial>` (every kaair link: lift, head, tool, fingers, base) | configured mass, COM and inertia from the convex hull of the visual mesh |
| collision = raw visual STL (non-convex, mm-scale, `package://`) | fingers + gripper body + lift column: CoACD convex decomposition; other links: one convex hull; the MoveIt-only `tooltip_collision_link` padding box is dropped |
| `<mimic>` fingers | MuJoCo equality `finger = 0.5 × virtual_gripper_joint` (stiffened) |
| finger velocity limit 0.005 m/s (inconsistent with ×0.5 mimic of 0.05) | 0.025 m/s |
| position servos / gravity | position actuators on the 11 joints the real robot drives with its forward controllers (joint1..7, lift, head pan/tilt, gripper); gravcomp on the robot like the xArm controller |
| gripper speed | virtual joint damping 1000 N·s/m → closes at ~0.05 m/s like the Koras. Without it the 5 g virtual joint slammed the fingers into objects at m/s and they tunnelled through |
| overlapping hulls by design | contact excludes: telescoping lift, pan-tilt brackets, finger↔finger (limit 0 is the mechanical stop) |
| cameras are only frames | `head_rgb` on `femto_color_optical_frame` (fovy 51°), `wrist_rgb` on `camera_color_optical_frame` (D405, fovy 58°) |

Kinematics (joint origins/axes/limits) are never modified.

## Results (both variants)

| Check | slamtec | clober |
|---|---|---|
| FK vs original xacro, all links × 200 random configs | 1.0e-6 m / 2.3e-6 rad ✅ | same ✅ |
| self-contact at home | 0 ✅ | 0 ✅ |
| hold home 3 s under gravity | drift 4e-13 rad ✅ | ✅ |
| arm step 0.3 rad, error after 1 s | ≤1.6e-4 except joint2 1.2e-2 * | same |
| grasp 4 cm cube, lift 10 cm | squeezes to 35 mm, cube rises 98.3 mm ✅ | ✅ |

\* in the folded home pose the gripper rests against `link_base`; stepping
joint2 +0.3 rad presses it harder. Real geometry, not a model error.

## Open items (need real-robot data)

1. **Masses are estimates** (`source: estimate` in `config/sim_urdf.yaml`):
   weigh the gripper body + fingers first (they matter for contact); base/lift
   only need to be non-zero since the base is fixed.
2. **Actuator gains / damping** are a starting point. System-ID: send the same
   forward-controller commands to the robot and the sim, and match the joint
   tracking.
3. **Camera intrinsics**: replace datasheet `fovy` with `fy` from the real
   `/femto/color/camera_info` and `/hand/.../camera_info`; the D405 mount
   extrinsic is the nominal one (`use_nominal_extrinsics`).
4. **Finger pad friction** (1.5) is a guess; the Koras pad material decides it.
5. Visuals are flat colours. For VLA, add textures / real2sim-scanned objects
   and domain randomization (next step).
6. Which base does the target robot use? Both variants are built; pick one for
   data generation.

## Next step (plan)

Isaac Lab (already installed in the `isaaclab` conda env on this machine) can
import `urdf/kaair_<v>_sim.urdf` directly for GPU-parallel rendering; MuJoCo
(`mjcf/`) is the fast-iteration twin. Then: tabletop task scenes, a scripted
expert, LeRobot-format recording with the same keys as the real robot, and a
ROS 2 bridge so the robot's own teleop/bringup can drive the sim.
