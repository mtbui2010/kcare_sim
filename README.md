# kcare-sim

A simulated KETI apartment where the **kcare home-care robot runs its real software**, and where
its **vision-language-action (VLA) policy for opening drawers** is trained and tested.

**Documentation and video:** https://mtbui2010.github.io/kcare_sim/

- MuJoCo house built from the owner's floor plan and photos (10 rooms, 16 YCB objects)
- the robot's own ROS 2 interfaces (nav2, arm / lift / head / gripper actions, cameras), so the
  unmodified `kcare_robot` software drives the simulated robot
- camera models fitted to the robot's Femto Bolt and RealSense D405 images (lens, exposure,
  white balance, noise, JPEG, motion blur, depth artefacts)
- scripted expert → LeRobot datasets → SmolVLA fine-tuning → closed-loop evaluation on fixed scenes
- the trained policy served by [vla-serve](https://github.com/mtbui2010/vla-serve) and called from
  kcare's `vla_drawer_open` skill

| policy | scenes | drawers opened |
|---|---|---|
| trained in the office, 20k steps | office | 44/50 |
| same | house | 4/50 |
| + 2.5k steps on office + house data | house | 26/50 |

```
sim/
  scripts/      house, ROS 2 bridge, camera models, data generation, evaluation, kcare launcher
  config/       task, randomisation, camera and layout files
  mjcf/ urdf/ assets/   the robot model and textures (YCB scans: sim/scripts/fetch_ycb.sh)
docs/draws/     floor plan (drawio)
website/        documentation site (MkDocs) and video sources
```

The robot maker's ROS 2 workspace (URDF sources, `kaair_msgs`, base image) and the `kcare_robot`
application are not part of this repository.

Licence: Apache-2.0 for the code in this repository. YCB models: CC BY 4.0.
