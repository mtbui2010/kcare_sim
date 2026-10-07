# 6. VLA policies and action chunks

!!! abstract "In this chapter"
    - SmolVLA reads two images, 8 joint values and the instruction, and outputs 50 future joint targets at once.
    - A chunk executor plays them at 30 Hz and asks for the next chunk before the current one runs out.
    - Co-training on office + house data keeps the office skill and adds the house.

## The idea

[SmolVLA](https://arxiv.org/abs/2506.01844) is a small vision-language-action model (a compact
vision-language backbone plus an action expert, about 450 M parameters). For kcare:

| input | output |
|---|---|
| head image 848×480, wrist image 848×480, joint1..7 + gripper, text prompt | 50 × 8 absolute joint targets |

At 30 Hz a chunk covers 1.7 s. The robot does not wait for the network: vla-serve's
`ChunkExecutor` sends the next request while the current chunk is still playing and switches when
the new one arrives (stale actions are dropped).

!!! code "Where in the code"
    [`vla_drawer_node.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/vla_drawer_node.py) (observation, `send()` to
    `/sim/arm_servo`), [`run_house_train.sh`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/run_house_train.sh) (fine-tuning
    command); the executor is vla-serve's client (`sim/third_party/vlaserve`).

!!! measured "Measured"
    The demo run: 450 control steps in 15 s, 14 chunks, 0 stalls; the drawer opened 28.6 cm.
    Training: about 1.4 s per step at batch 64 on one RTX A6000.

## What this means when you use it

- The camera model has to be the same in data, evaluation and the live sim; the task file selects
  the profile.
- Fine-tuning from the office checkpoint on office + house data (rather than house only) keeps the
  office scenes in the mix.
- Joint targets are absolute: the policy is tied to this robot's kinematics and ready pose.

**Next:** [7. Evaluating without fooling yourself →](07-evaluation.md)
