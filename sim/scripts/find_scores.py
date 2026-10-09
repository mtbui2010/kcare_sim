#!/usr/bin/env python3
"""Detector scores for objects that ARE on the surface in front of the robot vs objects that are NOT there.

    SIM_ROOT=/sim/inst2 ROS_DOMAIN_ID=78 KCARE_API=http://127.0.0.1:8003 python3 scripts/find_scores.py [--out out/find_scores.jsonl]

For each place: move there, then find::<name> for every object on that surface (present) and for a fixed list of
names that are not on it (absent). Each find is graded with the truth: the returned point is the right object
(< 6 cm from its centre), another object, or nothing. Prints score distributions to choose a min_conf.
"""
import argparse, json, math, os, sys, threading, time

import cv2
import numpy as np
import rclpy
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
from skill_bench import Bench, http, surface_of, to_world, last_call                     # noqa: E402

PLACES = ['desk@study room', 'stool@bedroom', 'dining table@kitchen', 'counter@kitchen', 'chest@bedroom',
          'shelf@kitchen', 'sofa@living room', 'test table@living room']
ABSENT = ['umbrella', 'tennis ball', 'remote control', 'scissors', 'banana', 'red cup', 'cracker box', 'sponge']


def name_of(body):
    return body[4:].replace('_', ' ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{SIM}/out/find_scores.jsonl')
    ap.add_argument('--scenes', nargs='*', default=[''], help="'' = the base house")
    ap.add_argument('--model', default='', help='detect_object model for this run (restored after)')
    ap.add_argument('--crops', default='', help='save the head frame crop of every detection here')
    ap.add_argument('--verify', action='store_true', help="find::verify=True (VLM checks each detection's crop)")
    a = ap.parse_args()
    rclpy.init(); b = Bench(); threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start()
    b.cli.wait_for_service(timeout_sec=10)
    fc0 = http('GET', '/skill-configs/FIND_CONFIGS')
    if a.model:
        fc = json.loads(json.dumps(fc0)); fc['detect_object']['model'] = a.model
        http('PUT', '/skill-configs/FIND_CONFIGS', fc)
    out = open(a.out, 'a')
    try:
        for scene in a.scenes:
            run_scene(b, scene, a.model, out, a.crops, a.verify)
    finally:
        if a.model:
            http('PUT', '/skill-configs/FIND_CONFIGS', fc0)
    rclpy.shutdown()


def run_scene(b, scene, model, out, crops='', verify=False):
    r = b.sim(cmd='apply', scene={}) if not scene else b.sim(cmd='load', name=scene)
    http('PUT', '/skill-configs/ENV', r['env']); http('PUT', '/skill-configs/MAP', r['map'])
    b.fresh(time.time()); b.sim(cmd='reset', what='all')
    spots = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))['spots']
    for place in PLACES:
        if place not in r['env']:
            continue
        http('POST', '/skill/move', {'inputs': place})
        T = b.sim(cmd='truth')
        here = {k: v for k, v in T['objects'].items() if surface_of(spots, v['centre']) == place}
        names = [(name_of(k), True) for k in here] + [(n, False) for n in ABSENT if n not in [name_of(k) for k in here]]
        bx, by, byaw = T['base']
        for n, present in names:
            t0 = time.time()
            try:
                res = http('POST', '/skill/find', {'inputs': n, **({'verify': True} if verify else {})})
            except Exception as e:
                res = {'isdone': False, 'msg': str(e)}
            rec = {'scene': scene or 'base', 'model': model or 'default', 'verify': verify, 'place': place, 'name': n,
                   'present': present, 'isdone': bool(res.get('isdone')),
                   'sec': round(time.time() - t0, 1)}
            ins = res.get('ins') or {}
            e = ins.get(n) or (next(iter(ins.values())) if ins else None)
            if res.get('isdone') and isinstance(e, dict) and e.get('pose_3d'):
                p = e['pose_3d']
                w = to_world(T['base'], p)
                near = min(T['objects'].items(), key=lambda kv: math.hypot(kv[1]['centre'][0] - w[0], kv[1]['centre'][1] - w[1]))
                d = math.hypot(near[1]['centre'][0] - w[0], near[1]['centre'][1] - w[1])
                if crops and e.get('box') and 'head' in b.img:
                    # the frame the detector saw: just before find's get3d call (the head moves on afterwards)
                    at = last_call('get3d..create_object_marker', t0, time.time())
                    older = [mm for tt, mm in b.ring if at and tt < at - 0.05]
                    m = older[-1] if older else b.img['head'][0]
                    im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
                    x0, y0, x1, y1 = [int(v) for v in e['box'][:4]]
                    if x1 <= x0 or y1 <= y0:                       # x, y, w, h
                        x1, y1 = x0 + x1, y0 + y1
                    pad = 12
                    crop = im[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                    if crop.size:
                        os.makedirs(crops, exist_ok=True)
                        f = f"{crops}/{len(os.listdir(crops)):04d}_{n.replace(' ', '_')}.jpg"
                        cv2.imwrite(f, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)); rec['crop'] = f
                rec.update(score=e.get('score'), box=e.get('box'), dist=round(math.hypot(p[0], p[1]), 2), nearest=name_of(near[0]),
                           nearest_d=round(d, 3),
                           verdict='right' if near[0] == 'obj_' + n.replace(' ', '_') and d < 0.06
                           else (f'wrong:{name_of(near[0])}' if d < 0.06 else 'nothing'))
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()


if __name__ == '__main__':
    main()
