# Generate open-drawer data

A scripted expert opens a drawer from the robot's ready pose; every successful episode is saved in
LeRobot format with the head and wrist images, the joint state and the commanded joint targets.

```bash
python3 scripts/gen_dataset.py --n 500 --workers 8 --gpu-ids 0,3 \
    --task /sim/config/task_open_drawer_house.yaml --rand-config /sim/config/randomization_house.yaml \
    --camera-model --seed 5000 --repo-id kaair_sim_open_drawer_house500
```

```text
[   136s] w4 OK   opened 24.9 cm  (118.2 s)   kept 1/500, tried 1
[   138s] w5 OK   opened 24.9 cm  (121.4 s)   kept 2/500, tried 2
...
done in 7684s: 500 episodes kept of 502 tried (100% success) -> /nas/datasets/kaair_sim_open_drawer_house500
```

(The `done` line is from the previous run of the same command; 502 tried because two workers
started one episode each past the target.)

## What each episode contains

| key | shape | meaning |
|---|---|---|
| `observation.images.head` | 480×848×3 | Femto image, 1280×720 resized like the robot's recorder |
| `observation.images.wrist` | 480×848×3 | D405 image |
| `observation.state` | 8 | joint1..7 [rad] + gripper opening [m] |
| `action` | 8 | the same 8 values as absolute targets, 30 Hz |
| `task` | text | e.g. `open the top drawer`; `open drawer` means the middle one, as on the robot |

## What is randomised per episode

From `config/randomization_house.yaml` ([`randomize.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/randomize.py)):

| factor | range |
|---|---|
| cabinet place in the house | two sites: the floor plan's `Drawer` corner, or beside the fridge (where the robot's own photos show it) |
| cabinet yaw / sideways / depth offset | ±5°, ±5 cm, ±5 cm |
| target drawer + phrasing | top / middle / bottom, 4 phrasings each |
| head pan / tilt around the ready pose | ±10°, −11…+9° |
| lift height | kcare `lift::0.56…0.70` |
| lighting | lamp and ambient intensity |
| textured vs plain-coloured cabinet | 80 % textured |
| camera | exposure, white balance, JPEG quality, shutter time (motion blur) |

Before a long run, render the extremes: [`check_ranges.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/check_ranges.py) and
[`preview_prompts.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/preview_prompts.py). Look at the per-worker preview videos in
`out/dataset_preview/` before training on anything.
