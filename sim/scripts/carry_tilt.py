#!/usr/bin/env python3
"""Tilt of a horizontally grasped mug (a cup of water) while the arm folds and the robot drives, two ways:

  old  movel back 0.15 m -> movej give -> lift 0.80 -> drive back      (what the demo / kcare do now)
  new  movel back to TCP x 0.45 (orientation kept) -> movel drz -90 (turn the gripper sideways, still level)
       -> lift 0.56 -> drive back

The mug is picked from the high shelf of scene front_demo the same way both times (front_demo.side_grasp).
Tilt = angle between the mug's up axis now and at the start (sim truth quaternion), sampled at ~10 Hz.

    python3 scripts/carry_tilt.py old|new   (KCARE_API, ROS_DOMAIN_ID as for front_demo)
"""
import json, math, sys, threading, time

import numpy as np
import rclpy

sys.path.insert(0, '/sim/scripts')
from skill_bench import Bench, tilt_deg                        # noqa: E402
import front_demo as F                                         # noqa: E402

mode = sys.argv[1]
rclpy.init(); b = Bench(); threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start(); time.sleep(1.5)
d = F.Demo(b, '')
F.http('POST', '/agent/sim_command/send', {'cmd': 'reset', 'what': 'all'})
q0 = b.sim(cmd='truth')['quat']['obj_mug']
phase, worst, stop = ['setup'], {}, threading.Event()


def sampler():
    while not stop.is_set():
        try:
            q = b.sim(cmd='truth')['quat']['obj_mug']
            t = tilt_deg(q0, q)
            worst[phase[0]] = max(worst.get(phase[0], 0.0), t)
        except Exception:
            pass
        time.sleep(0.1)


threading.Thread(target=sampler, daemon=True).start()
place = 'high shelf@living room'
d_edge, h, view = d.stand_and_look(place)
x, y, z = d.find_settled('mug', view)
print('found', round(x, 3), round(y, 3), round(z, 3), flush=True)
if mode == 'old':
    fwd = d.approach(x, y, z, d_edge, h, 'side', 'mug')
else:                                  # the real robot's two-step pre_pick: tcp (0.815, 0, lift + 0.44), gripper forward
    fwd = max(-0.3, min(x - 0.97, d_edge - F.BASE_R - F.EDGE_GAP))   # object 0.15 m beyond pre_pick's tcp
    if fwd < 0:                         # back up first: pre_pick's tcp must stay outside the furniture
        d.must('forward', inputs=round(fwd, 3))
    d.must('lift', inputs=round(min(max(z - 0.44, 0.56), 1.10), 3))
    d.must('movej', inputs='pre_pick')  # in free space, before driving in
    if fwd > 0:
        d.must('forward', inputs=round(fwd, 3))
    print('pre_pick', d.sk('arm_pose')['pose'], flush=True)
    d.must('movel', x=round(x - fwd - 0.15, 3), y=round(y, 3), z=round(z, 3))
    print('pre-grasp', d.sk('arm_pose')['pose'], flush=True)
phase[0] = 'grasp'
ok = d.side_grasp('mug')
print('grasped', ok, flush=True)
if mode == 'old':
    phase[0] = 'fold'
    d.sk('movel', dx=-0.15); d.must('movej', inputs='give'); d.must('lift', inputs=0.8)
else:
    p0 = d.sk('arm_pose')['pose']
    goal = (0.45, 0.0)
    n = 6
    for k in range(1, n + 1):          # back + turn sideways together, gripper kept level (carry_pose.py path)
        phase[0] = f'carry step {k}'
        cur = d.sk('arm_pose')['pose']
        tx = p0[0] + (goal[0] - p0[0]) * k / n; ty = p0[1] + (goal[1] - p0[1]) * k / n
        d.must('movel', dx=round(tx - cur[0], 3), dy=round(ty - cur[1], 3), drz=-90 / n)
        print(phase[0], d.sk('arm_pose')['pose'], flush=True)
    phase[0] = 'lift 0.56'
    d.must('lift', inputs=0.56)
phase[0] = 'drive'
d.must('forward', inputs=round(-fwd - 0.3, 3))
d.must('turn', inputs=90)
d.must('forward', inputs=0.5)
time.sleep(1.0)
stop.set(); time.sleep(0.2)
c = d.truth('mug')
print(json.dumps({'mode': mode, 'max_tilt_deg': {k: round(v, 1) for k, v in worst.items()},
                  'mug_still_held': bool(c and c[2] > 0.3 and not d.on_surface('mug', place)[0])}), flush=True)
