# Code map

All code is in `sim/scripts/`; configuration in `sim/config/`.

## The house

Builds the apartment and everything in it.

| name | does |
|---|---|
| [`house_scene.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_scene.py) | rooms, walls, furniture, spots, YCB objects → `mjcf/house.xml` + `config/house_layout.yaml`; `drawer_scene_xml()` for training scenes; `--render` comparisons |
| [`cabinet.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/cabinet.py) | parametric drawer cabinets (`STYLES`: `A_real`, `A_home`, `W_chest`, …), slide joints, handles, feet |
| [`office_scene.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/office_scene.py) | the office room the first policy was trained in |

## The live robot

| name | does |
|---|---|
| [`house_bridge.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_bridge.py) | MuJoCo loop, A* planner, IK, the robot's ROS 2 actions/services/topics, camera rendering |
| [`camera_model.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/camera_model.py) | `CameraModel` (distortion, auto-exposure, colour, noise, JPEG, motion blur, rolling shutter), `DepthModel` (Femto ToF, D405 stereo) |
| [`kcare_sim_app.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/kcare_sim_app.py) | starts kcare's FastAPI app on the sim profile, with two runtime patches |
| [`make_kcare_sim_location.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/make_kcare_sim_location.py) | kcare site profile `sim_house`: connections to 127.0.0.1, ENV from the spots |
| [`vla_drawer_node.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/vla_drawer_node.py) | ROS action `/drawer/open` → vla-serve chunk executor → `/sim/arm_servo` |
| [`run_house.sh`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/run_house.sh) | start / stop / status of the three processes |

## Data, training, evaluation

| name | does |
|---|---|
| [`drawer_ready.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/drawer_ready.py) | kcare's ready pose for the drawer; places the cabinet relative to the gripper; `build_scene()` |
| [`expert_open_drawer.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/expert_open_drawer.py) | scripted expert: approach, grasp the handle, pull |
| [`randomize.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/randomize.py) | one episode's randomisation from a YAML config |
| [`gen_dataset.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/gen_dataset.py) | parallel workers → LeRobot shards → one dataset |
| [`eval_policy.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/eval_policy.py), [`watch_eval.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/watch_eval.py), [`eval_clips.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/eval_clips.py) | closed-loop evaluation, per-checkpoint watcher, labelled clips |
| [`run_house_train.sh`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/run_house_train.sh) | merge office + house data, baseline, fine-tune, watch |

## Calibration and checks

| name | does |
|---|---|
| [`estimate_camera.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/estimate_camera.py), [`calibrate_color.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/calibrate_color.py), [`calibrate_home.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/calibrate_home.py) | intrinsics + noise, office colour fit, home colour fit |
| [`depth_compare.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/depth_compare.py) | real vs simulated depth at a recorded pose |
| [`render_real_compare.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/render_real_compare.py) | real vs sim camera images before generating data |
| [`house_tour_test.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_tour_test.py) | kcare move + find at every place, error against the true positions |
| [`house_record_grid.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_record_grid.py), [`house_record.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_record.py), [`house_snapshot.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_snapshot.py) | videos and snapshots |

## The robot model

| name | does |
|---|---|
| [`flatten_urdf.sh`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/flatten_urdf.sh) → [`build_sim_urdf.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/build_sim_urdf.py) → [`urdf_to_mjcf.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/urdf_to_mjcf.py) | robot xacro → physics-ready URDF (masses, convex collision, mimic fingers) → MJCF with actuators and cameras |
| [`check_sim.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/check_sim.py) | FK against the URDF, holding still, stepping, grasping, rendering |

## Other files

| path | what |
|---|---|
| `sim/config/*.yaml` | see [Configuration](../reference/configuration.md) |
| `sim/docker/`, `sim/docker-compose.yml`, `sim/Makefile` | container and model pipeline |
| `sim/third_party/vlaserve/` | vla-serve's Python client (copy) |
| `docs/draws/floor.drawio` | the apartment floor plan |
| `website/` | this site; `website/tools/` figures, recorded data, video sources |
