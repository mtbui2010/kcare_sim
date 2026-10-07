---
hide:
  - navigation
---

# kcare-sim

**A simulated apartment where the KETI kcare robot runs its real software, and where its
vision-language-action (VLA) policy for opening drawers is trained and tested.**

kcare is a home-care robot: a mobile base, a lift, a pan-tilt head with an Orbbec Femto Bolt
camera, a 7-joint xArm7 arm and a gripper with an Intel RealSense D405. kcare-sim rebuilds the
KETI demo apartment in MuJoCo from a floor plan and photos, puts a physics model of the robot in
it, and exposes the robot's own ROS 2 interfaces — so the unmodified `kcare_robot` software
(navigation, object finding, the dashboard) drives the simulated robot exactly as it drives the
real one. The same scenes generate training data for a SmolVLA policy and evaluate it.

![The VLA policy opening the middle drawer in the simulated kitchen, seen by the wrist camera
(frames 23 s to 42 s of the demo run)](assets/img/filmstrip_vla.png)

## Watch: the whole workflow, narrated

<video controls preload="metadata" playsinline style="width:100%;border-radius:8px"
       poster="assets/video/tutorial-poster.png">
  <source src="assets/video/tutorial.mp4" type="video/mp4">
  Your browser cannot play this video — <a href="assets/video/tutorial.mp4">download it</a>.
</video>

About five minutes: the house, the robot software running in it, the camera models, data,
training, evaluation, and the policy opening a drawer through kcare.

## In one minute

```bash
# inside the sim container (cwd /sim)
bash scripts/run_house.sh start          # MuJoCo house + ROS 2 bridge + VLA node + kcare_robot API :8001
curl -s -X POST localhost:8001/skill/move -H 'Content-Type: application/json' \
     -d '{"inputs": "drawer@kitchen"}'   # kcare drives to the cabinet
curl -s -X POST localhost:8001/skill/vla_drawer_open -H 'Content-Type: application/json' \
     -d '{"inputs": "open drawer"}'      # the SmolVLA policy opens it
```

Real reply of the last call in the run shown above: `opened cab_drawer1 by 28.6 cm (sim ground
truth) in 18.7 s` (14 action chunks, 0 stalls).

## What it takes off your hands

| When you do this on the real robot… | kcare-sim |
|---|---|
| One robot, shared by a team; every test needs a person next to it | as many simulated robots as GPUs, no supervision |
| A policy trained in the office fails at home (8 % success in the house) | house scenes built from the real floor plan and photos to train on (52 % after 2.5k steps) |
| Simulators look nothing like the real cameras | camera models fitted to the robot's own images: lens, exposure, white balance, noise, JPEG, depth artefacts |
| A sim needs its own robot software | the robot's `kcare_robot` code runs unmodified against the same ROS 2 topics, actions and services |
| Demonstrations are slow to record | a scripted expert records 500 episodes in about two hours, 100 % success |

## Verified so far

| what | how it was checked | result |
|---|---|---|
| kcare navigation + `find` in the house | [`house_tour_test.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_tour_test.py): move to 8 places, find an object at each, compare with the true position | 8/8 moves, 7/8 objects found, position error 4.5–15 cm for 5 of them, 65 and 96 cm for 2 |
| Policy trained in the office, tested in the house | 50 fixed random scenes | 4/50 (8 %) |
| Same policy after 2.5k more steps on office + house data | the same 50 scenes | 26/50 (52 %, 95 % interval 38–65 %) |
| VLA through kcare, end to end | `move::drawer@kitchen`, then `vla_drawer_open` (recorded in the video) | drawer opened 28.6 cm in 18.7 s |
| Camera colour | mean colour of wood, white and floor patches, real D405 image vs sim | wood and floor within 8 levels per channel (0–255) after calibration; white 10–25 levels darker |

**Not yet:** the house-trained policy has not been run on the real robot at home; the Femto
depth model uses datasheet values (no real Femto depth frame was recorded).

## Learn

The [Learn](learn/index.md) section explains the ideas behind kcare-sim — running real robot
software in simulation, building a scene from photos, making renders look like real cameras,
training and evaluating a VLA — with this repository's code and measurements.
