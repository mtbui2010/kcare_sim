# 5. Demonstrations from a scripted expert

!!! abstract "In this chapter"
    - In simulation the exact cabinet pose is known, so a short script can open any drawer: no teleoperation needed.
    - The script starts from the robot's real ready pose (kcare's `moveh`, `lift`, `movej::approach_drawer`).
    - Everything that should not matter is randomised; each episode records a prompt naming the drawer.

## The idea

[`expert_open_drawer.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/expert_open_drawer.py) knows where the handle is and moves the
gripper through seven phases: hold, pregrasp (turn the wrist so the fingers close vertically on
the bar, stop 8 cm in front), approach, grasp, pull 25 cm straight back, release, retreat. Every
30 Hz step it solves IK for the next waypoint and commands the 8 values the real robot records
(joint1..7 + gripper width), so the recorded actions are directly learnable.

The start is the robot's own: the head, lift and arm angles of kcare's open-drawer routine
([`drawer_ready.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/drawer_ready.py)), with the cabinet placed relative to the gripper
the way the real robot finds it (calibrated from the robot's images: 0.445 m from the wrist camera
to the drawer front).

!!! code "Where in the code"
    [`gen_dataset.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/gen_dataset.py) (workers, LeRobot writer),
    [`randomize.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/randomize.py), `config/randomization_house.yaml`.

!!! measured "Measured"
    500 of 502 started episodes kept (100 % success), about 100 s per episode per worker, 8 workers on
    two GPUs: about 2 hours for 500 episodes, 168k frames.

## What this means when you use it

- The expert is perfect, the policy is not: the data has no recoveries from mistakes. Policies
  therefore fail by stalling rather than retrying.
- Prompts: `open drawer` means the middle drawer, as on the robot; top / bottom have their own
  phrasings — the policy has to read the instruction.
- Watch the preview videos of the first episodes before a long run.

**Next:** [6. VLA policies and action chunks →](06-vla.md)
