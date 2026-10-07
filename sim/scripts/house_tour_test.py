#!/usr/bin/env python3
"""Drive kcare_robot (HTTP API) around the simulated house: move + find at each stop.

    source /workspace/ros2/ros2_ws/install/setup.bash
    python3 scripts/house_tour_test.py [--api http://localhost:8002]
      -> /sim/out/house/tour/<n>_<object>.jpg   overview | head | wrist after `find`
         /sim/out/house/tour/summary.md         what was asked, what came back, how long it took

Exactly what the dashboard / CLI would send: POST /skill/move {inputs: '<spot>'}, then
POST /skill/find {inputs: '<object>@<spot>'}. The result's pose_3d (base_footprint) is
checked against where the object really is in the sim (config/house_layout.yaml).
"""
import argparse, json, math, os, subprocess, time, urllib.request

import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
STOPS = [('desk@study room', 'orange'), ('shelf@kitchen', 'cracker box'),
         ('counter@kitchen', 'mustard bottle'), ('dining table@kitchen', 'bowl'), ('washer@laundry room', 'bleach bottle'),
         ('chest@bedroom', 'red cup'), ('stool@bedroom', 'tennis ball'), ('shelf@restroom', 'spray bottle')]


def call(api, skill, body, timeout=300):
    req = urllib.request.Request(f'{api}/skill/{skill}', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='http://localhost:8001')
    a = ap.parse_args()
    out = f'{SIM}/out/house/tour'
    os.makedirs(out, exist_ok=True)
    lay = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))
    rows = ['| # | move to | t [s] | find | t [s] | score | error vs truth [cm] |', '|---|---|---|---|---|---|---|']
    for k, (spot, obj) in enumerate(STOPS):
        mv, t_mv = call(a.api, 'move', {'inputs': spot})
        fd, t_fd = call(a.api, 'find', {'inputs': f'{obj}@{spot}'})
        pose, _ = call(a.api, 'mobile_pose', {})
        ins = (fd.get('ins') or {}).get(obj) or next(iter((fd.get('ins') or {}).values()), None)
        err, score = '-', '-'
        if ins and pose.get('isdone'):
            x, y, rz = pose['pose']; c, s = math.cos(math.radians(rz)), math.sin(math.radians(rz))
            px, py, pz = ins['pose_3d'][:3]
            wx, wy = x + c * px - s * py, y + s * px + c * py            # base_footprint -> map
            tx, ty, tz = lay['objects'][obj]['pos']
            err = f'{math.dist((wx, wy), (tx, ty)) * 100:.1f}'
            score = f"{ins.get('score', 0):.2f}"
        ok_mv, ok_fd = mv.get('isdone'), fd.get('isdone')
        rows.append(f"| {k} | {spot} | {t_mv:.0f} {'OK' if ok_mv else 'FAIL'} | {obj}: {'found' if ok_fd else 'NOT found'} "
                    f"| {t_fd:.0f} | {score} | {err} |")
        print(rows[-1], flush=True)
        subprocess.run(['python3', f'{SIM}/scripts/house_snapshot.py'], capture_output=True)
        os.replace(f'{SIM}/out/house/snap.jpg', f'{out}/{k}_{obj.replace(" ", "_")}.jpg')
    mv, t_mv = call(a.api, 'move', {'inputs': 'home'})
    rows.append(f"| - | home | {t_mv:.0f} {'OK' if mv.get('isdone') else 'FAIL'} | | | | |")
    open(f'{out}/summary.md', 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows[-1:]), '\n->', out)


if __name__ == '__main__':
    main()
