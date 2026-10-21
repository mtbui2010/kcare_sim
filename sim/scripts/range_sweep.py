#!/usr/bin/env python3
"""How far along / across the counter kcare's pick and place work, and what mforward does (2026-10-19).

    python3 scripts/range_sweep.py pick|place [--objs "soup can,banana"] [--out out/range/<name>.json]

The robot stands at counter@kitchen's ENV stop (left mode: base heading +x, the counter on its -y side, front edge
0.55 m, back 1.19 m). pick: the target is teleported to (bx, by) in the base frame, then pick::<obj>@counter@kitchen;
graded by sim truth (held). place: the target is picked from an easy spot, then place::counter@kitchen with
placepose x / y absolute (the base frame); graded by where it ends (released on the counter, distance to the request).
"""
import argparse, json, math, os, sys, threading, time, urllib.request
import rclpy
SIM = os.environ.get('SIM_ROOT', '/sim'); sys.path.insert(0, f'{SIM}/scripts')
from skill_bench import Bench                                   # noqa: E402
API = 'http://127.0.0.1:8001'
LOC = 'counter@kitchen'
PARK = (1.18, -2.25)                                            # dining table: where the other object waits
BX = [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
BY = [-0.62, -0.75, -0.9]


def http(m, p, b=None, t=900):
    r = urllib.request.Request(API + p, method=m, data=None if b is None else json.dumps(b).encode(),
                               headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=t) as f:
        return json.loads(f.read() or b'null')


def body(n):
    return 'obj_' + ''.join(c if c.isalnum() else '_' for c in n.lower())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('kind'); ap.add_argument('--objs', default='soup can,banana')
    ap.add_argument('--out', default=''); a = ap.parse_args()
    rclpy.init(); B = Bench(); threading.Thread(target=rclpy.spin, args=(B,), daemon=True).start(); time.sleep(2)
    objs = [o.strip() for o in a.objs.split(',')]
    print('load', http('POST', '/agent/sim_command/send', {'cmd': 'load', 'name': 'range_test'})['msg'], flush=True)
    time.sleep(30)
    rows = []

    def truth():
        for k in range(4):              # the bridge sometimes misses one /sim/command (2026-10-19: one 120 s timeout)
            try:
                return B.sim(cmd='truth')
            except TimeoutError:
                time.sleep(5)
        return B.sim(cmd='truth')

    def home():
        http('POST', '/skill/grip', {'inputs': 'open'}); time.sleep(1)
        http('POST', '/agent/sim_command/send', {'cmd': 'reset', 'what': 'objects'}); time.sleep(1)
        http('POST', '/skill/move', {'inputs': LOC}); http('POST', '/skill/movej', {'inputs': 'fold'})
        http('POST', '/skill/lift', {'inputs': 0.8})
        return truth()['base']

    def place_obj(obj, bx, by, base):
        x0, y0, yaw = base[0], base[1], math.radians(base[2])
        wx, wy = x0 + bx * math.cos(yaw) - by * math.sin(yaw), y0 + bx * math.sin(yaw) + by * math.cos(yaw)
        for o in objs:
            if o != obj:
                http('POST', '/agent/sim_command/send', {'cmd': 'move_object', 'body': body(o), 'x': PARK[0], 'y': PARK[1]})
        r = {}
        tx, ty = wx, wy
        for _ in range(3):              # the mesh centre is not the body origin (soup can: 8 cm): correct and repeat
            r = http('POST', '/agent/sim_command/send', {'cmd': 'move_object', 'body': body(obj), 'x': tx, 'y': ty})
            time.sleep(1.5)
            o = truth()['objects'][body(obj)]
            ex_, ey_ = wx - o['centre'][0], wy - o['centre'][1]
            if math.hypot(ex_, ey_) < 0.01 and abs(o['zmin'] - 0.9) < 0.03:
                break
            tx, ty = tx + ex_, ty + ey_
        r['on_counter'] = abs(o['zmin'] - 0.9) < 0.03
        r['at_err'] = round(math.hypot(ex_, ey_), 3)
        return r

    def to_base(p, base):
        yaw = math.radians(base[2]); dx, dy = p[0] - base[0], p[1] - base[1]
        return [round(math.cos(yaw) * dx + math.sin(yaw) * dy, 3), round(-math.sin(yaw) * dx + math.cos(yaw) * dy, 3)]

    for obj in objs:
        for by in BY:
            for bx in BX:
                base = home()
                t0 = time.time()
                if a.kind == 'pick':
                    mo = place_obj(obj, bx, by, base)
                    try:
                        r = http('POST', '/skill/pick', {'inputs': f'{obj}@{LOC}'})
                    except Exception as e:
                        r = {'isdone': False, 'msg': str(e)}
                    tr = truth()
                    held = body(obj) in tr['held']
                    row = dict(kind='pick', obj=obj, bx=bx, by=by, isdone=r.get('isdone'), held=held,
                               mforward=r.get('mforward'), lying=r.get('islying'),
                               found=r.get('pose_3d') and [round(v, 3) for v in r['pose_3d'][:3]],
                               msg=str(r.get('msg', ''))[-600:], sec=round(time.time() - t0, 1), on_counter=mo.get('on_counter'), at_err=mo.get('at_err'))
                else:
                    place_obj(obj, 0.3, -0.72, base)
                    try:
                        rp = http('POST', '/skill/pick', {'inputs': f'{obj}@{LOC}'})
                    except Exception as e:
                        rp = {'isdone': False, 'msg': str(e)}
                    if body(obj) not in truth()['held']:
                        rows.append(dict(kind='place', obj=obj, bx=bx, by=by, skipped='pick failed', msg=str(rp.get('msg', ''))[:120]))
                        print(json.dumps(rows[-1]), flush=True); continue
                    t0 = time.time()
                    try:
                        r = http('POST', '/skill/place', {'inputs': LOC, 'placepose': {'x': bx, 'y': abs(by)}})
                    except Exception as e:
                        r = {'isdone': False, 'msg': str(e)}
                    time.sleep(1.5)
                    tr = truth(); o = tr['objects'][body(obj)]
                    end = to_base(o['centre'], tr['base'] if False else base)
                    on_counter = abs(o['zmin'] - 0.9) < 0.05
                    row = dict(kind='place', obj=obj, bx=bx, by=by, isdone=r.get('isdone'), mforward=r.get('mforward'),
                               spot=r.get('pose_3d') and [round(v, 3) for v in r['pose_3d'][:3]], end=end,
                               on_counter=on_counter, err=round(math.hypot(end[0] - bx, end[1] - by), 3),
                               held_after=body(obj) in tr['held'], msg=str(r.get('msg', ''))[-600:], sec=round(time.time() - t0, 1))
                rows.append(row); print(json.dumps(row), flush=True)
                if a.out:
                    json.dump(rows, open(a.out, 'w'), indent=1)
    os._exit(0)


if __name__ == '__main__':
    main()
