# Command line

Run inside the container, cwd `/sim`.

## run_house.sh

| command | does |
|---|---|
| `start` | sim bridge, VLA node, kcare API (:8001) — waits until each is ready |
| `stop` | stops all three |
| `status` | which are running |
| `sim` | only the bridge |
| `restart-sim`, `restart-kcare`, `restart-vla` | restart one process |

Environment: `BRIDGE_ARGS` (passed to `house_bridge.py`), `HOUSE_CAB_SITE` (`nw` / `fridge`).

## Scripts and their flags

| script | flags |
|---|---|
| `house_scene.py` | `--render` |
| `house_bridge.py` | `--xml` (default `mjcf/house.xml`), `--rate` camera Hz (10), `--no-camera-model`, `--no-depth-model` |
| `make_kcare_sim_location.py` | `--standoff` m in front of a surface (0.55) |
| `house_tour_test.py` | `--api` (http://localhost:8001) |
| `house_record_grid.py` | `--plan "skill:inputs;…"` (required), `--speed` (1), `--fps` (15), `--out`, `--api` |
| `gen_dataset.py` | `--n` episodes to keep (50), `--workers` (8), `--gpu-ids` (0), `--task`, `--rand-config`, `--camera-model`, `--seed` (0), `--repo-id`, `--preview` (1) |
| `eval_policy.py` | `--ckpt` (required), `--n` (50), `--videos` (4), `--timeout` s (15), `--success-m` (0.20), `--seed0` (1e9), `--device`, `--task`, `--rand-config`, `--camera-model`, `--tag` |
| `watch_eval.py` | `--run` (required), `--n` (50), `--settle` s (120), `--poll` s (60), `--tag`, `--eval-args` |
| `eval_clips.py` | `--ckpt`, `--ok` (3), `--fail` (3), `--max` (24), `--seed0`, `--task`, `--rand-config`, `--camera-model` |
| `vla_drawer_node.py` | `--url` (http://127.0.0.1:11450), `--model`, `--seconds` (15), `--fps` (30), `--success-m` (0.20) |
| `calibrate_home.py`, `depth_compare.py <real dir>` | – |
| `check_ranges.py` | `--rand-config` |
| `check_sim.py`, `urdf_to_mjcf.py`, `build_sim_urdf.py` | `slamtec` / `clober` (+ `--n-fk`, `--audit`) |

## Makefile (robot model pipeline, `sim/Makefile`)

| target | does |
|---|---|
| `image`, `up`, `shell` | build and enter the container |
| `ws` | import the robot's third-party repos, build its description packages (once) |
| `urdf` → `sim-urdf` → `mjcf` → `check` (`all`) | xacro → physics URDF → MJCF → acceptance checks, `VARIANT=slamtec|clober` |
