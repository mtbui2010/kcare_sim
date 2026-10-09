#!/usr/bin/env python3
"""Skill bench: run kcare's move / find / pick / place many times over many scenes and grade each step
against the simulator's ground truth.

    source /workspace/ros2/ros2_ws/install/setup.bash; export ROS_DOMAIN_ID=77
    python3 scripts/skill_bench.py --scenes bench_100 bench_101 ... [--tasks 5] [--repeat 1] --out out/bench/run1

One trial = reset everything (robot at the start, objects at the scene's poses, world beliefs cleared), then
    move::<surface A>      graded: nav error to the ENV loc, heading error
    find::<object>         graded: detected point (head: base frame, wrist: tool frame) vs the object's true centre,
                           and WHICH object it is nearest to (right / wrong object / nothing there)
    pick::<object>         graded: both finger tips on the object and lifted (held), dropped / knocked off,
                           other objects disturbed; kcare's own isdone is recorded next to it
    place::<surface B>     (when pick reported done) graded: object resting on B, upright, released
Commands go through kcare's HTTP API exactly as from the dashboard; ground truth comes straight from the
bridge's /sim/command service (never through kcare: every kcare call runs begin_run()).
Writes <out>/trials.jsonl (one line per trial) and <out>/img/*.jpg (head + wrist at find and at failures).
"""
import argparse, json, math, os, random, threading, time, urllib.request

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from sim_msgs.srv import SimCommand

SIM = os.environ.get('SIM_ROOT', '/sim')
API = os.environ.get('KCARE_API', 'http://127.0.0.1:8001')
STANDOFF = 0.55


class Bench(Node):
    def __init__(self):
        super().__init__('skill_bench')
        self.cli = self.create_client(SimCommand, '/sim/command')
        self.img, self.stamp, self.ring = {}, {}, []
        self.create_subscription(Image, '/femto/color/image_raw', self._head, 1)
        self.create_subscription(CompressedImage, '/hand/camera/color/image_raw/compressed', self._wrist, 1)

    def _head(self, m):
        now = time.time()
        self.img['head'] = (m, now)
        self.ring.append((now, m)); del self.ring[:-60]          # ~15 s of head frames

    def _wrist(self, m):
        self.img['wrist'] = (m, time.time())

    def sim(self, **cmd):
        req = SimCommand.Request(); req.request = json.dumps(cmd)
        fut = self.cli.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > 120:
                raise TimeoutError(f'/sim/command {cmd.get("cmd")}')
            time.sleep(0.01)
        return json.loads(fut.result().response)

    def save_imgs(self, path, box=None, label='', head_at=None):
        """Latest head (or the one received just before `head_at`, + kcare's detection box) and wrist frames."""
        for k, v in list(self.img.items()):
            m = v[0]
            if k == 'head' and head_at is not None:
                older = [mm for tt, mm in self.ring if tt < head_at]
                m = older[-1] if older else m
            if k == 'head':
                im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR).copy()
                if box:
                    x0, y0, x1, y1 = [int(v_) for v_ in box]
                    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 255), 3)
                    cv2.putText(im, label, (x0, max(30, y0 - 10)), 0, 1.0, (0, 255, 255), 2)
                im = cv2.resize(im, (640, 360))
            else:
                im = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(f'{path}_{k}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def fresh(self, after, timeout=180):
        """Wait until the bridge renders the reloaded model and both cameras published after that."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            t = self.sim(cmd='truth')
            if t.get('render_version') == t.get('version'):
                t1 = time.time()
                while time.time() - t0 < timeout:
                    if all(k in self.img and self.img[k][1] > t1 for k in ('head', 'wrist')):
                        return time.time() - after
                    time.sleep(0.1)
            time.sleep(0.5)
        return None


def http(method, path, body=None, timeout=400):
    r = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read() or b'null')


def skill(name, inputs, trace=None):
    """Run a kcare skill over HTTP. trace=(bench, body): also sample the ground truth every 0.5 s while it runs."""
    t0, samples, stop = time.time(), [], threading.Event()

    def sample():
        b, body = trace
        while not stop.is_set():
            try:
                t = b.sim(cmd='truth')
                o = t['objects'].get(body, {})
                samples.append({'t': round(time.time() - t0, 2), 'base': t['base'], 'tcp': t['tcp'], 'lift': t['lift'],
                                'grip': t['gripper'], 'held': body in t['held'], 'obj': o.get('centre')})
            except Exception:
                pass
            stop.wait(0.5)
    th = threading.Thread(target=sample, daemon=True) if trace else None
    if th:
        th.start()
    try:
        r = http('POST', f'/skill/{name}', {'inputs': inputs})
    except Exception as e:
        r = {'isdone': False, 'msg': f'HTTP: {e}'}
    if th:
        stop.set(); th.join()
        r['_trace'] = samples
    return r, t0, time.time()


def drop_event(samples):
    """First held -> not held transition: when, and where the TCP / object were."""
    for a_, b_ in zip(samples, samples[1:]):
        if a_['held'] and not b_['held']:
            return {'t': b_['t'], 'tcp': a_['tcp'], 'base': a_['base'], 'lift': a_['lift'], 'grip_before': a_['grip'],
                    'grip_after': b_['grip'], 'obj_after': b_['obj']}
    return None


def last_call(agent, t0, t1, log=f'{SIM}/out/house/kcare.log'):
    """Time of kcare's last `<agent>...@sent` ROS call within [t0, t1] (from the tail of kcare.log)."""
    import datetime as dt
    best = None
    try:
        with open(log, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(log) - 400000))
            for line in fh.read().decode(errors='replace').splitlines():
                if f'@{agent}' in line and line.endswith('@sent saved') and line[:20].isdigit():
                    t = dt.datetime.strptime(line[:20], '%Y%m%d%H%M%S%f').timestamp()
                    if t0 <= t <= t1:
                        best = t
    except OSError:
        pass
    return best


def to_world(base, p):
    x, y, a = base[0], base[1], math.radians(base[2])
    return [x + math.cos(a) * p[0] - math.sin(a) * p[1], y + math.sin(a) * p[0] + math.cos(a) * p[1], p[2]]


def in_rect(sp, p, margin=0.0):
    fy = math.radians(sp['front_yaw_deg'])
    d = np.array(p[:2]) - np.array(sp['centre'])
    n = np.array([math.cos(fy), math.sin(fy)]); al = np.array([-math.sin(fy), math.cos(fy)])
    return abs(float(d @ n)) <= sp['depth'] / 2 + margin and abs(float(d @ al)) <= sp['width'] / 2 + margin


def surface_of(spots, p):
    """The spot whose rectangle contains p and whose height is just below p's z (else None = floor / elsewhere)."""
    best = None
    for k, sp in spots.items():
        if in_rect(sp, p, 0.02) and -0.05 < p[2] - sp['height'] < 0.35:
            if best is None or sp['height'] > spots[best]['height']:
                best = k
    return best


def tilt_deg(q0, q1):
    """Angle between the object's up axis at two poses (w, x, y, z quats)."""
    def up(q):
        w, x, y, z = q
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return math.degrees(math.acos(float(np.clip(up(q0) @ up(q1), -1, 1))))


def short(r):
    m = r.get('msg')
    return (m if isinstance(m, str) else json.dumps(m))[:600] if m else ''


def run_trial(b, scene, obj, A, B, spots, env, idx, out, log):
    rec = {'standoff': STANDOFF, 'scene': scene, 'idx': idx, 'object': obj['name'], 'ycb': obj['ycb'], 'from': A, 'to': B,
           'from_front': obj.get('bench', {}).get('from_front'), 't': time.time()}
    body = 'obj_' + ''.join(c if c.isalnum() else '_' for c in obj['name'].lower())
    b.sim(cmd='reset', what='all')
    http('PUT', '/agent/world', {'holding': None, 'found': None, 'opened': []})
    time.sleep(1.0)
    T0 = b.sim(cmd='truth')
    if body not in T0['objects']:
        rec['skipped'] = 'object not in the scene'; return rec
    start = T0['objects']
    rec['true_centre'] = start[body]['centre']
    tag = f'{out}/img/{scene}_{idx:02d}'

    # ── move ──
    r, t0, t1 = skill('move', A)
    T1 = b.sim(cmd='truth')
    loc = env[A]['loc']
    bx, by, byaw = T1['base']
    rec['move'] = {'isdone': bool(r.get('isdone')), 'sec': round(t1 - t0, 1), 'msg': short(r), 't0': t0,
                   'base': T1['base'], 'nav_err': round(math.hypot(bx - loc['x'], by - loc['y']), 3),
                   'heading_err': round(((byaw - (loc['rz'] + 90) + 180) % 360) - 180, 1)}
    c = start[body]['centre']
    a = math.radians(byaw)
    rel = [math.cos(a) * (c[0] - bx) + math.sin(a) * (c[1] - by), -math.sin(a) * (c[0] - bx) + math.cos(a) * (c[1] - by)]
    rec['obj_in_base'] = [round(rel[0], 3), round(rel[1], 3), round(c[2], 3)]       # where the object is for the arm

    # ── find ──
    r, t0, t1 = skill('find', obj['name'])
    f = {'isdone': bool(r.get('isdone')), 'sec': round(t1 - t0, 1), 'msg': short(r), 't0': t0}
    ins = (r.get('ins') or {}).get(obj['name'])
    if ins and ins.get('pose_3d'):
        p = ins['pose_3d']
        if 'grasppose' in ins:                                       # wrist entry: pose in the tool frame
            R = np.array(T1['tcp_R']).reshape(3, 3)
            w = np.array(T1['tcp']) + R @ np.array([-p[1], p[0], p[2]])
            f['cam'] = 'arm'
        else:
            w = np.array(to_world(T1['base'], p)); f['cam'] = 'head'
        f['world'] = [round(float(v), 3) for v in w]
        f['err_xy'] = round(math.hypot(w[0] - c[0], w[1] - c[1]), 3)
        f['err_z'] = round(float(w[2] - c[2]), 3)
        near = min(T1['objects'].items(), key=lambda kv: math.hypot(kv[1]['centre'][0] - w[0], kv[1]['centre'][1] - w[1]))
        dn = math.hypot(near[1]['centre'][0] - w[0], near[1]['centre'][1] - w[1])
        f['nearest'] = near[0][4:]; f['nearest_d'] = round(dn, 3)
        f['verdict'] = 'right' if near[0] == body and dn < 0.06 else (f'wrong:{near[0][4:]}' if dn < 0.06 else 'nothing there')
        f['score'] = ins.get('score'); f['islying'] = ins.get('islying')
        f['box'] = ins.get('box'); f['depths'] = ins.get('depths')
    else:
        f['verdict'] = 'not found'
    rec['find'] = f
    # the head frame kcare detected on: the last one before its 3D lookup (create_object_marker), minus the detector
    b.save_imgs(f'{tag}_find', f.get('box') if f.get('cam') == 'head' else None, f"{obj['name']}: {f['verdict']}",
                head_at=(last_call('get3d..create_object_marker', t0, t1) or t1) - 0.3)

    # ── pick ──
    r, t0, t1 = skill('pick', obj['name'], trace=(b, body))
    T2 = b.sim(cmd='truth')
    o2 = T2['objects'][body]
    held = body in T2['held']
    lifted = o2['centre'][2] > start[body]['centre'][2] + 0.03
    disturbed = [n[4:] for n, v in T2['objects'].items() if n != body and math.dist(v['centre'], start[n]['centre']) > 0.03]
    fell = [n[4:] for n, v in T2['objects'].items() if v['centre'][2] < 0.15 and start[n]['centre'][2] > 0.3]
    where = 'held' if held and lifted else ('floor' if o2['centre'][2] < 0.15 else
            ('in place' if math.dist(o2['centre'], start[body]['centre']) < 0.03 else 'moved'))
    rep = bool(r.get('isdone'))
    rec['pick'] = {'isdone': rep, 'sec': round(t1 - t0, 1), 'msg': short(r), 't0': t0, 'held': held and lifted,
                   'object_now': where, 'touching': T2['touching'].get(body, ''), 'grip_force': T2['grip_force'],
                   'disturbed': disturbed, 'fell': fell, 'gripper': T2['gripper'],
                   'ever_held': any(x['held'] for x in r['_trace']), 'drop': drop_event(r['_trace']),
                   'verdict': ('ok' if rep and held and lifted else 'FALSE OK: ' + where if rep else
                               'held anyway' if held and lifted else 'fail: ' + where)}
    if rec['pick']['verdict'] != 'ok':
        b.save_imgs(f'{tag}_pick')

    # ── place ──
    if rep:
        q0 = T2['quat'][body]
        r, t0, t1 = skill('place', B, trace=(b, body))
        T3 = b.sim(cmd='truth')
        o3 = T3['objects'][body]
        on = surface_of(spots, o3['centre'])
        still = body in T3['held']
        q_start = T0['quat'][body]
        rec['place'] = {'isdone': bool(r.get('isdone')), 'sec': round(t1 - t0, 1), 'msg': short(r), 't0': t0,
                        'on': on, 'still_held': still, 'base': T3['base'], 'drop': drop_event(r['_trace']),
                        'obj_final': o3['centre'],
                        'dest_dist': round(math.hypot(o3['centre'][0] - spots[B]['centre'][0], o3['centre'][1] - spots[B]['centre'][1]), 3),
                        'tcp_min_z_held': min([x['tcp'][2] for x in r['_trace'] if x['held']] or [None]),
                        'tilt_vs_start': round(tilt_deg(q_start, T3['quat'][body]), 1),
                        'fell': [n[4:] for n, v in T3['objects'].items() if v['centre'][2] < 0.15 and T2['objects'][n]['centre'][2] > 0.3]}
        okp = on == B and not still
        rec['place']['verdict'] = ('ok' if okp and rec['place']['isdone'] else
                                   'FALSE OK: ' + (on or 'floor/other') if rec['place']['isdone'] else
                                   'fail: ' + ('still held' if still else (on or 'floor/other')))
        if rec['place']['verdict'] != 'ok':
            b.save_imgs(f'{tag}_place')
    rec['sec'] = round(time.time() - rec['t'], 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--tasks', type=int, default=5, help='tasks per scene')
    ap.add_argument('--repeat', type=int, default=1, help='runs per task')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--standoff', type=float, default=None, help='ENV standoff (base centre to front edge), default 0.55')
    ap.add_argument('--out', default=f'{SIM}/out/bench/run')
    a = ap.parse_args()
    global STANDOFF
    STANDOFF = a.standoff or 0.55
    os.makedirs(f'{a.out}/img', exist_ok=True)
    rclpy.init()
    b = Bench()
    threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start()
    b.cli.wait_for_service(timeout_sec=10)
    log = open(f'{a.out}/trials.jsonl', 'a')
    for scene in a.scenes:
        sdef = yaml.safe_load(open(f'{SIM}/config/scenes/{scene}.yaml'))
        t0 = time.time()
        r = b.sim(cmd='load', name=scene, **({} if a.standoff is None else {'standoff': a.standoff}))
        if not r.get('ok'):
            print('load failed', scene, r.get('msg')); continue
        http('PUT', '/skill-configs/ENV', r['env']); http('PUT', '/skill-configs/MAP', r['map'])
        cam = b.fresh(time.time())
        skill('detect', 'apple')                                     # first call after a restart is slow (ROS reconnect)
        print(f'[{time.strftime("%T")}] {scene}: {r["msg"]}, cameras back after {cam:.0f} s' if cam else f'{scene}: cameras did not come back', flush=True)
        lay = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))
        spots, env = lay['spots'], r['env']
        rng = random.Random(f'{scene}-{a.seed}')
        for o in sdef['objects']:                                    # photo scenes place objects by `at` on the test table
            o.setdefault('surface', 'test table@living room' if 'photo' in sdef else None)
        targets = [o for o in sdef['objects'] if o.get('bench', {}).get('target', True) and o.get('surface') in env]
        by_surf = {}
        for o in targets:
            by_surf.setdefault(o['surface'], []).append(o)
        surfs = list(by_surf); rng.shuffle(surfs)
        tasks = []
        for k in range(a.tasks):
            A = surfs[k % len(surfs)]
            o = rng.choice(by_surf[A])
            B = rng.choice([s for s in env if s != A and s in spots])
            tasks.append((o, A, B))
        idx = 0
        for o, A, B in tasks:
            for _ in range(a.repeat):
                rec = run_trial(b, scene, o, A, B, spots, env, idx, a.out, log)
                idx += 1
                log.write(json.dumps(rec) + '\n'); log.flush()
                print(f"[{time.strftime('%T')}] {scene} #{idx} {o['name']} {A} -> {B}: "
                      f"move {rec.get('move', {}).get('nav_err')} | find {rec.get('find', {}).get('verdict')} "
                      f"{rec.get('find', {}).get('err_xy')} | pick {rec.get('pick', {}).get('verdict')} | "
                      f"place {rec.get('place', {}).get('verdict', '-')} | {rec.get('sec')} s", flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
