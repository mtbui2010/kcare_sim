#!/usr/bin/env python3
"""Video of kcare's align_shoes v2 in the house sim (scene shoes_demo): push the pair together, pinch both, carry them
to 'entrance', put them down. Graded by sim truth: both on the floor near the destination, side by side (centres
0.07-0.14 m apart), parallel (< 15 deg), toes pointing away from the robot, nothing held.

    python3 scripts/shoes_video.py [--video /sim/out/shoes] [--no-load]
"""
import argparse, json, math, os, sys, threading, time, urllib.request
import numpy as np
import rclpy
SIM = os.environ.get('SIM_ROOT', '/sim'); sys.path.insert(0, f'{SIM}/scripts')
from skill_bench import Bench                                   # noqa: E402
API = 'http://127.0.0.1:8001'
START = [('obj_left_shoe', -0.13, 0.80, 90.0), ('obj_right_shoe', 0.12, 0.74, 78.0)]   # as config/scenes/shoes_demo.yaml


def http(m, p, b=None, t=1200):
    r = urllib.request.Request(API + p, method=m, data=None if b is None else json.dumps(b).encode(),
                               headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=t) as f:
        return json.loads(f.read() or b'null')


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--video', default=''); ap.add_argument('--no-load', action='store_true')
    ap.add_argument('--inputs', default='black shoes>>entrance'); a = ap.parse_args()
    rclpy.init(); B = Bench(); threading.Thread(target=rclpy.spin, args=(B,), daemon=True).start(); time.sleep(3)
    if not a.no_load:
        print('load', http('POST', '/agent/sim_command/send', {'cmd': 'load', 'name': 'shoes_demo'})['msg'], flush=True)
        time.sleep(35)
    http('POST', '/skill/grip', {'inputs': 'open'}); time.sleep(1)
    http('POST', '/agent/sim_command/send', {'cmd': 'reset', 'what': 'objects'}); time.sleep(2)
    # first the robot to the shoe area (coming back from the entrance it drove right over the pair), then the pair
    # back where the scene has it (a scene reload keeps objects the robot moved where they are)
    http('POST', '/skill/move', {'inputs': 'shoe area@living room'})
    for body, x, y, yaw in START:
        http('POST', '/agent/sim_command/send', {'cmd': 'move_object', 'body': body, 'x': x, 'y': y, 'yaw_deg': yaw})
    time.sleep(2)
    t0_ = B.sim(cmd='truth')
    print('start', {k: [round(v, 3) for v in t0_['objects'][k]['centre'][:2]] for k in ('obj_left_shoe', 'obj_right_shoe')}, flush=True)
    http('POST', '/skill/move', {'inputs': 'shoe area@living room'})
    vid = None
    if a.video:
        from task_video import TaskVideo
        vid = TaskVideo(B, f'{a.video}/align_shoes.mp4', title='align_shoes v2: push, pinch both, carry, put down')
        vid.add('user', f'align_shoes::{a.inputs}')
        vid.set_step('look -> push -> pinch both -> lift -> move to entrance -> put down')
    t0 = time.time()
    try:
        r = http('POST', '/skill/align_shoes', {'inputs': a.inputs})
    except Exception as e:
        r = {'isdone': False, 'msg': str(e)}
    sec = time.time() - t0
    time.sleep(2)
    tr = B.sim(cmd='truth'); base = tr['base']; yb = math.radians(base[2])
    pos = {k: tr['objects'][k]['centre'] for k in ('obj_left_shoe', 'obj_right_shoe')}
    yaws = {k: yaw_of(tr['quat'][k]) for k in pos}
    on_floor = all(tr['objects'][k]['zmin'] < 0.02 for k in pos)
    sep = math.dist(pos['obj_left_shoe'][:2], pos['obj_right_shoe'][:2])
    dyaw = abs((math.degrees(yaws['obj_left_shoe'] - yaws['obj_right_shoe']) + 180) % 360 - 180)
    # toe = +x of the shoe frame; away from the robot = the robot heading
    toe_err = max(abs((math.degrees(yaws[k] - yb) + 180) % 360 - 180) for k in pos)
    near = all(math.dist(p[:2], base[:2]) < 0.9 for p in pos.values())
    held = [k for k in pos if k in tr['held']]
    ok = bool(r.get('isdone') and on_floor and 0.07 <= sep <= 0.14 and dyaw < 15 and toe_err < 25 and near and not held)
    res = dict(isdone=r.get('isdone'), msg=str(r.get('msg', ''))[-200:], sec=round(sec), on_floor=on_floor, sep=round(sep, 3),
               dyaw=round(dyaw, 1), toe_err=round(toe_err, 1), near_robot=near, held=held, ok=ok, pinch=r.get('pinch'), pushed=r.get('pushed'))
    print(json.dumps(res), flush=True)
    if vid:
        vid.add('robot' if r.get('isdone') else 'fail', f"align_shoes isdone={r.get('isdone')} after {sec:.0f} s {res['msg'][:80]}")
        vid.add('ok' if ok else 'fail', f"truth: on floor {on_floor}, {sep * 100:.1f} cm apart, headings differ {dyaw:.0f} deg, "
                                        f"toes {toe_err:.0f} deg off 'away from the robot', held {held or 'none'}")
        print('video', vid.close(), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
