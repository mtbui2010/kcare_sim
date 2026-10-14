#!/usr/bin/env python3
"""Video of kcare's pnp skill (pick an object, put it on / into another one at the same spot) in the house sim.

    python3 scripts/pnp_video.py --video /sim/out/pnp [--only apple]      (scene config/scenes/pnp_demo.yaml)

Graded by sim truth: the picked object ends within the destination object's radius, above its base.
"""
import argparse, json, math, os, sys, threading, time, urllib.request

import rclpy

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
from skill_bench import Bench                                  # noqa: E402

API = os.environ.get('KCARE_API', 'http://127.0.0.1:8001')
CASES = [   # (object, destination, location, title)
    ('rubiks cube', 'bowl', 'desk@study room', 'PNP  rubiks cube -> bowl (desk 0.72 m)'),
    ('banana', 'red plate', 'counter@kitchen', 'PNP  banana -> red plate (counter 0.90 m)'),
]


def http(method, path, body=None, timeout=600):
    r = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read() or b'null')


def body(name):
    return 'obj_' + ''.join(c if c.isalnum() else '_' for c in name.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--video', default='')
    ap.add_argument('--no-load', action='store_true')
    ap.add_argument('--at', action='store_true', help='start at the location (arm folded) instead of the robot start')
    a = ap.parse_args()
    rclpy.init(); b = Bench(); threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start()
    time.sleep(2)
    if not a.no_load:
        print('load scene', http('POST', '/agent/sim_command/send', {'cmd': 'load', 'name': 'pnp_demo'}), flush=True)
        for _ in range(60):                    # the render loop needs ~30 s after a reload
            try:
                t = b.sim(cmd='truth')
                if t.get('render_version') == t.get('version'):
                    break
            except Exception:
                pass
            time.sleep(1)
        time.sleep(3)
    res = {}
    for obj, dest, loc, title in CASES:
        if a.only and obj not in a.only:
            continue
        print(f'== {title}', flush=True)
        http('POST', '/skill/grip', {'inputs': 'open'})          # first: an open command releases the grasp assist
        time.sleep(1)
        http('POST', '/agent/sim_command/send', {'cmd': 'reset', 'what': 'objects' if a.at else 'all'})
        time.sleep(3)
        if a.at:
            http('POST', '/skill/move', {'inputs': loc}); http('POST', '/skill/movej', {'inputs': 'fold'}); time.sleep(1)
        vid = None
        if a.video:
            from task_video import TaskVideo
            vid = TaskVideo(b, f'{a.video}/pnp_{obj.split()[-1]}_to_{dest.split()[-1]}.mp4', title=title)
            vid.add('user', f'pnp::{obj}>>{dest}@{loc}')
            vid.set_step('pnp: move -> find both -> pick -> place -> fold')
        t0 = time.time()
        try:
            r = http('POST', '/skill/pnp', {'inputs': f'{obj}>>{dest}@{loc}'})
        except Exception as e:
            r = {'isdone': False, 'msg': str(e)}
        sec = time.time() - t0
        time.sleep(1.5)
        tr = b.sim(cmd='truth')
        oo, od = tr['objects'].get(body(obj)), tr['objects'].get(body(dest))
        po, pd = (oo or {}).get('centre'), (od or {}).get('centre')
        dxy = math.hypot(po[0] - pd[0], po[1] - pd[1]) if po and pd else None
        ok_truth = dxy is not None and dxy < 0.08 and po[2] > pd[2] - 0.01 and body(obj) not in tr.get('held', [])
        msg = (f"skill isdone {r.get('isdone')}  {str(r.get('msg') or r.get('place_check') or '')[:80]} | "
               f"truth: {obj} {dxy * 100:.1f} cm from the {dest} centre, z {po[2]:.3f} vs {pd[2]:.3f}" if dxy is not None
               else f"skill {r}")
        print('  ', msg, f'{sec:.0f} s', json.dumps(r.get('timings')), flush=True)
        res[obj] = dict(isdone=r.get('isdone'), truth_ok=ok_truth, sec=round(sec, 1), dxy=dxy)
        if vid:
            vid.add('robot' if r.get('isdone') else 'fail', f"pnp returned isdone={r.get('isdone')} after {sec:.0f} s")
            if r.get('timings'):
                vid.add('info', 'steps [s]: ' + ', '.join(f'{k} {v:.1f}' for k, v in r['timings'].items()))
            vid.add('ok' if ok_truth else 'fail', f"sim truth: {obj} {'IN/ON' if ok_truth else 'NOT on'} the {dest} "
                                                  f"({dxy * 100:.0f} cm from its centre)" if dxy is not None else 'no truth')
            print('  video', vid.close(), flush=True)
    print(json.dumps(res), flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
