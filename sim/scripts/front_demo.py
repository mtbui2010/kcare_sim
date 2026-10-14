#!/usr/bin/env python3
"""Front-mode pick / place demo without a VLA (2026-10-11, the procedure agreed with the owner):

  1. stand 0.5 m further back than the working spot, facing the furniture (front mode, arm folded);
  2. one head view chosen from the place's ENV height (< 0.5 m down, 0.5-1.2 straight, > 1.2 up) -> find;
  3. drive forward by (object distance - reach of the band), kept >= 5 cm from the furniture edge;
  4. the arm unfolds at the LOWEST lift, in the free space of the stand-back spot (a high lift brings the arm into
     the head camera's mount); then the lift goes to the working height; then the base drives in;
     top-down (approach_lying) for small / lying objects, horizontal (the real robot's two-step pre_pick) for
     standing containers (mug, cup ...) from 0.55 m and for everything above 1.15 m;
  5. grasp with the wrist camera (fine_move; horizontal: its steps with a 6 cm shallower approach);
  6. a horizontally held object is carried LEVEL: 6 small movel steps back to the carry pose, each turning the
     gripper 15 deg about the vertical (measured: <= 5 deg tilt vs 44 deg with movej give); then lift down.
Place is the same with a spot on the surface's front strip instead of a detection; the object is lowered to
the height it was held above its pick surface. Skills go over HTTP to the sim's kcare (no kcare code change).

    KCARE_API=http://127.0.0.1:8001 python3 scripts/front_demo.py --video out/front_demo [--only P1 L1]
"""
import argparse, json, math, os, sys, threading, time, urllib.request

import rclpy

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
from skill_bench import Bench                                  # noqa: E402

API = os.environ.get('KCARE_API', 'http://127.0.0.1:8001')
BACK = 0.50              # extra stand-back [m]
BASE_R = 0.24            # base radius
EDGE_GAP = 0.05          # base front edge <-> furniture edge, minimum
WORK = BASE_R + 0.06     # base centre <-> furniture edge at the working spot
REACH_TOP = 0.50         # object x (base frame) for the top-down approach (approach_lying TCP x)
HIGH = 1.15              # above: horizontal approach
LIFT = (0.56, 1.10)
PP_X, PP_DZ = 0.815, 0.44  # the real robot's two-step pre_pick: tcp (0.815, 0, lift + 0.44), gripper forward, level
SIDE_GAP = 0.15          # horizontal grasp: the base stops with the object this far beyond pre_pick's tcp
CARRY = (0.45, 0.0)      # carry pose for a level object: tcp pulled back to here, gripper turned 90 deg, still level
CARRY_STEPS = 6
SIDE_OBJECTS = ('mug', 'cup', 'glass', 'bottle', 'bowl', 'pot')   # standing containers: grasp sideways, carry level

DEMOS = {   # id: (kind, object, place, title); the mug leaves the shelf before the apple goes there
    'P1': ('pick', 'tomato soup can', 'desk@study room', 'PICK  desk 0.72 m'),
    'L1': ('place', 'tomato soup can', 'sofa@living room', 'PLACE  sofa seat 0.44 m'),
    'P3': ('pick', 'mug', 'high shelf@living room', 'PICK  high shelf 1.31 m'),
    'L3': ('place', 'mug', 'desk@study room', 'PLACE  desk 0.72 m'),
    'P2': ('pick', 'apple', 'sofa@living room', 'PICK  sofa seat 0.44 m'),
    'L2': ('place', 'apple', 'high shelf@living room', 'PLACE  high shelf 1.31 m'),
}


def http(method, path, body=None, timeout=400):
    r = urllib.request.Request(API + path, method=method, data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read() or b'null')


class Demo:
    def __init__(self, bench, out):
        self.b, self.out, self.vid = bench, out, None
        self.env = http('GET', '/skill-configs/ENV')
        self.surf = {s['name']: s for s in http('GET', '/skill-configs/MAP').get('surfaces', [])}
        self.held = {}           # object -> {'style': top|side, 'hold_dz': TCP z - pick surface, 'half_h'}
        self.lift_k = 0.8
        self.log_lines = []

    # ── helpers ──
    def say(self, kind, text):
        print(f'  [{kind}] {text}', flush=True)
        self.log_lines.append((kind, text))
        if self.vid:
            self.vid.add(kind, text)

    def step(self, text):
        if self.vid:
            self.vid.set_step(text)

    def sk(self, name, **p):
        r = http('POST', f'/skill/{name}', p)
        if not isinstance(r, dict):
            r = {'isdone': False, 'msg': str(r)}
        short = {k: v for k, v in r.items() if k in ('isdone', 'msg', 'final_height', 'pose')}
        print(f'    {name} {json.dumps(p, ensure_ascii=False)} -> {json.dumps(short, default=str)[:200]}', flush=True)
        return r

    def must(self, name, **p):
        r = self.sk(name, **p)
        if not r.get('isdone'):
            raise RuntimeError(f"{name} failed: {r.get('msg') or r}")
        return r

    def geometry(self, place):
        """(stand x, y, rz deg, distance from the stand pose to the furniture edge)."""
        s, e = self.surf[place], self.env[place]
        cx, cy = s['centre']; yaw = math.radians(s['yaw_deg']); depth = s['size'][0]
        ax = (math.cos(yaw), math.sin(yaw))                      # the surface's depth axis
        lx, ly = e['loc']['x'], e['loc']['y']                    # kcare's stop is on the front side
        n = 1.0 if (lx - cx) * ax[0] + (ly - cy) * ax[1] > 0 else -1.0
        nx, ny = n * ax[0], n * ax[1]                            # outward normal of the front edge
        d_edge = WORK + BACK
        sx, sy = cx + nx * (depth / 2 + d_edge), cy + ny * (depth / 2 + d_edge)
        return sx, sy, math.degrees(math.atan2(-ny, -nx)), d_edge

    def stand_and_look(self, place, carrying_level=False):
        sx, sy, rz, d_edge = self.geometry(place)
        h = self.env[place]['height']
        view = 'down' if h < 0.5 else ('straight' if h <= 1.2 else 'up')
        self.step(f'1. stand 0.5 m back from {place}')
        self.say('plan', f'go to {place.split("@")[0]}: stand {BACK:.1f} m back, front mode, arm folded')
        if not carrying_level:                       # a level-carried cup stays in its carry pose
            self.must('movej', inputs='give')
        self.lift_to(LIFT[0])                        # lowest: the folded arm's shoulder drops out of the head view
        self.must('moveh', inputs='front,' + view)
        self.must('moveb', x=round(sx, 3), y=round(sy, 3), rz=round(rz, 1))
        return d_edge, h, view

    def lift_to(self, h):
        h = round(min(max(h, LIFT[0]), LIFT[1]), 3)
        self.must('lift', inputs=h)
        self.lift_k = h
        return h

    def drive(self, d):
        if abs(d) > 0.02:
            self.must('forward', inputs=round(d, 3))

    def pose(self):
        return self.sk('arm_pose')['pose']

    def movel_check(self, level=False, tol=0.03, **target):
        """movel, then verify: the sim's arm reports 'ok' even when it stopped short / turned the gripper."""
        self.must('movel', **target)
        p = self.pose()
        bad = [k for k in ('x', 'y', 'z') if k in target and abs(p['xyz'.index(k)] - target[k]) > tol]
        if level and abs(abs(p[4]) - 89.2) > 8:
            bad.append(f'gripper not level (ry {p[4]:.0f})')
        if bad:
            raise RuntimeError(f'movel {target} ended at {p}: {bad}')
        return p

    def unfold(self, kind, x_goal, d_edge, z_tcp):
        """Arm out at the lowest lift in free space, then the lift, then the base. kind 'side': pre_pick (two steps),
        x_goal = where pre_pick's tcp + SIDE_GAP must land; 'top': approach_lying, x_goal = object x - REACH_TOP."""
        fwd = max(-0.3, min(x_goal, d_edge - BASE_R - EDGE_GAP))
        if fwd < 0:                                  # back up first: the tcp must stay outside the furniture
            self.drive(fwd)
        self.lift_to(LIFT[0])
        self.step('3. unfold at the lowest lift (free space)')
        if kind == 'side':
            self.say('plan', 'unfold: pre_pick (2 steps) at the lowest lift, gripper level')
            self.must('movej', inputs='pre_pick')
        else:
            self.say('plan', 'unfold: approach_lying at the lowest lift, gripper down')
            self.must('movej', inputs='approach_lying')
        lift = self.lift_to(z_tcp - PP_DZ if kind == 'side' else z_tcp)
        self.step(f'3. lift {lift:.2f}, then drive {fwd:+.2f} m')
        self.say('plan', f'lift {lift:.2f} m' + (f', then drive in {fwd:.2f} m' if fwd > 0.02 else
                                                  (f' (backed up {-fwd:.2f} m first)' if fwd < -0.02 else '')))
        if fwd > 0:
            self.drive(fwd)
        return fwd

    def carry_level(self, back=True):
        """back=True: from the grasp (gripper forward) to the carry pose; False: the reverse. Small movel steps,
        each moving the tcp and turning the gripper 90/CARRY_STEPS deg about the vertical, gripper kept level."""
        p0 = self.pose()
        goal = CARRY if back else (PP_X, 0.0)
        self.say('plan', ('carry level: ' if back else 'uncarry level: ') + f'{CARRY_STEPS} small steps, gripper level')
        for k in range(1, CARRY_STEPS + 1):
            cur = self.pose()
            tx = p0[0] + (goal[0] - p0[0]) * k / CARRY_STEPS; ty = p0[1] + (goal[1] - p0[1]) * k / CARRY_STEPS
            self.must('movel', dx=round(tx - cur[0], 3), dy=round(ty - cur[1], 3), drz=(-90 if back else 90) / CARRY_STEPS)
        p = self.pose()
        if abs(abs(p[4]) - 89.2) > 8:
            raise RuntimeError(f'carry: gripper not level at {p}')

    def side_grasp(self, obj, back=0.06, trials=2):
        """fine_move's steps for a horizontal grasp: find_grasp (wrist) -> centre (dx, dy) -> approach along the
        tool axis -> close -> lift 2 cm -> pull back. fine_move's approach depth (camera depth - tool length) is
        tuned for top-down grasps, where going ~6 cm past the top face is right; sideways it pushes the object away,
        so the approach stops `back` short."""
        for _ in range(trials):
            self.must('grip', inputs='open')
            r = self.sk('find_grasp', inputs=obj)
            if not r.get('isdone'):
                continue
            dx, dy, dz, _angle, _width = r['ins'][obj]['grasppose'][:5]
            d = max(0.0, abs(dz) - back)
            self.must('movet', dx=round(dx, 3), dy=round(dy, 3))
            self.must('movet', dz=round(d, 3), acc=0.25)
            self.must('grip', inputs='close')
            time.sleep(0.5)
            self.sk('movel', dz=0.02)
            self.must('movet', dz=-round(d, 3), acc=0.25)
            if self.sk('grasp_succeed').get('isdone'):
                return True
        return False

    def truth(self, obj):
        T = self.b.sim(cmd='truth')
        o = T['objects'].get('obj_' + ''.join(c if c.isalnum() else '_' for c in obj.lower()))
        return (o or {}).get('centre')

    def on_surface(self, obj, place):
        c = self.truth(obj)
        if not c:
            return None, 'object not in the sim truth'
        s = self.surf[place]
        cx, cy = s['centre']; yaw = math.radians(s['yaw_deg'])
        dx, dy = c[0] - cx, c[1] - cy
        u = dx * math.cos(yaw) + dy * math.sin(yaw); v = -dx * math.sin(yaw) + dy * math.cos(yaw)
        ok = abs(u) <= s['size'][0] / 2 + 0.02 and abs(v) <= s['size'][1] / 2 + 0.02 and -0.02 < c[2] - s['height'] < 0.3
        return ok, f"object at ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"

    def find_settled(self, obj, view):
        """Head find after the base has settled; two finds must agree within 4 cm (a frame grabbed while the base
        was still turning at the end of moveb put the mug 18 cm off once)."""
        seen = []
        time.sleep(1.5)
        for _ in range(6):
            r = self.sk('find', inputs=obj, views=[view], cameras='head', move_arm=False)
            if not r.get('isdone'):
                continue
            p = r['ins'][obj]['pose_3d'][:3]
            for q in seen:
                if max(abs(a - b) for a, b in zip(p, q)) < 0.04:
                    return [(a + b) / 2 for a, b in zip(p, q)]
            seen.append(p)
        raise RuntimeError(f'head finds of the {obj} do not agree: {seen}')

    # ── pick / place ──
    def pick(self, obj, place):
        d_edge, h, view = self.stand_and_look(place)
        self.step(f'2. head view "{view}" -> find {obj}')
        self.say('plan', f'head looks {view} (place height {h:.2f} m) to find the {obj}')
        x, y, z = self.find_settled(obj, view)
        self.say('ok', f'found {obj} at x {x:.2f}  y {y:.2f}  z {z:.2f} m (base frame)')
        style = 'side' if (h > HIGH or (any(w in obj for w in SIDE_OBJECTS) and h >= 0.55)) else 'top'
        if style == 'side':
            fwd = self.unfold('side', x - (PP_X + SIDE_GAP), d_edge, z)
            if x - fwd > PP_X + SIDE_GAP + 0.2:
                raise RuntimeError(f'{obj} is too deep on the surface to reach from the front')
            self.movel_check(level=True, x=round(x - fwd - SIDE_GAP, 3), y=round(y, 3), z=round(z, 3))
            self.step('4. wrist camera grasp (horizontal)')
            self.say('plan', 'wrist camera grasp (fine_move steps, 6 cm shallower: horizontal)')
            self.side_grasp(obj)
            self.step('5. carry level, back off')
            self.carry_level(back=True)
            self.lift_to(LIFT[0])
        else:
            fwd = self.unfold('top', x - REACH_TOP, d_edge, h + 0.12)
            if x - fwd > REACH_TOP + 0.18:
                raise RuntimeError(f'{obj} is too deep on the surface to reach from the front')
            self.movel_check(x=round(x - fwd, 3), y=round(y, 3), z=round(z + 0.22, 3))
            self.step('4. fine_move (wrist camera, top-down)')
            self.say('plan', 'fine_move: wrist camera detects + grasps from above')
            self.sk('fine_move', inputs=obj, num_trials=2)
            self.step('5. fold, back off')
            self.must('movej', inputs='give')
            self.lift_to(0.8)
        tcp_z = self.pose()[2]
        self.held[obj] = {'style': style, 'hold_dz': max(0.02, tcp_z - h), 'half_h': max(0.02, z - h)}
        self.drive(-fwd)
        gs = self.sk('grasp_succeed').get('isdone')
        c = self.truth(obj)
        held = bool(c) and c[2] > 0.3 and not self.on_surface(obj, place)[0]   # sim truth: off its surface, not on the floor
        self.say('ok' if held else 'fail', f'{obj} ' + ('is in the gripper' if held else 'NOT in the gripper')
                 + (f' (sim truth z {c[2]:.2f} m; grasp_succeed says {gs})' if c else ''))
        return held

    def place(self, obj, place):
        hold = self.held.get(obj, {'style': 'top', 'hold_dz': 0.06, 'half_h': 0.05})
        d_edge, h, view = self.stand_and_look(place, carrying_level=hold['style'] == 'side')
        x_spot = d_edge + 0.12                       # front strip of the surface, 12 cm behind the edge
        self.step(f'2. head view "{view}": free spot on the front strip')
        self.say('plan', f'head looks {view}: put it 12 cm behind the front edge, centred ({x_spot:.2f} m ahead)')
        style = 'side' if (h > HIGH or hold['style'] == 'side') else 'top'
        if style == 'side':
            z_rel = h + hold.get('half_h', 0.05) + 0.03            # the object hangs around the tcp height
            if hold['style'] == 'side':
                self.carry_level(back=False)                        # at the lowest lift, in free space
                fwd = max(-0.3, min(x_spot - (PP_X + SIDE_GAP), d_edge - BASE_R - EDGE_GAP))
                if fwd < 0:
                    self.drive(fwd)
                tcp = self.pose()
                lift = self.lift_to(self.lift_k + (z_rel + 0.03 - tcp[2]))
                self.say('plan', f'lift {lift:.2f} m, then drive {fwd:+.2f} m')
                if fwd > 0:
                    self.drive(fwd)
            else:                                    # held from above (round apple): turn level with pre_pick
                fwd = self.unfold('side', x_spot - (PP_X + SIDE_GAP), d_edge, z_rel + 0.03)
            x = x_spot - fwd
            self.step('4. release (slide in, lower, open)')
            self.movel_check(level=True, x=round(x - SIDE_GAP, 3), y=0.0, z=round(z_rel + 0.03, 3))
            self.must('movel', dx=SIDE_GAP)
            self.must('movel', dz=-0.03)
            self.must('grip', inputs='open')
            self.sk('movel', dx=-(SIDE_GAP + 0.05))
        else:
            z_rel = h + hold['hold_dz'] + 0.02      # same grasp orientation as the pick: same tcp height above the surface
            fwd = self.unfold('top', x_spot - REACH_TOP, d_edge, h + 0.12)
            x = x_spot - fwd
            self.step('4. release (lower from above, open)')
            self.movel_check(x=round(x, 3), y=0.0, z=round(z_rel + 0.12, 3))
            self.must('movel', dz=-0.12)
            self.must('grip', inputs='open')
            self.sk('movel', dz=0.05)
        self.step('5. fold, back off')
        self.must('movej', inputs='give')
        self.lift_to(0.8)
        self.drive(-fwd)
        ok, where = self.on_surface(obj, place)
        self.say('ok' if ok else 'fail', f'{obj} ' + (f'is on the {place.split("@")[0]}' if ok else 'is NOT on the target') + f' ({where})')
        return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--video', default='')
    a = ap.parse_args()
    rclpy.init(); b = Bench(); threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start()
    time.sleep(2)
    d = Demo(b, a.video)
    results = {}
    for did, (kind, obj, place, title) in DEMOS.items():
        if a.only and did not in a.only:
            continue
        print(f'== {did} {title}: {obj}', flush=True)
        if a.video:
            from task_video import TaskVideo
            d.vid = TaskVideo(b, f'{a.video}/{did}_{kind}_{place.split("@")[0].replace(" ", "_")}.mp4',
                              title=f'{did} {title} (front, no VLA)')
            d.vid.add('user', f'{kind} the {obj} ' + ('from' if kind == 'pick' else 'onto') + f' the {place.split("@")[0]}')
        t0 = time.time()
        body = 'obj_' + ''.join(c if c.isalnum() else '_' for c in obj.lower())
        tilt, stop = [0.0], threading.Event()

        def sample():                                  # the object's up axis vs vertical (sim truth), while it moves
            while not stop.is_set():
                try:
                    w, x_, y_, z_ = b.sim(cmd='truth')['quat'][body]
                    up_z = 1 - 2 * (x_ * x_ + y_ * y_)
                    tilt[0] = max(tilt[0], math.degrees(math.acos(max(-1.0, min(1.0, up_z)))))
                except Exception:
                    pass
                time.sleep(0.5)
        threading.Thread(target=sample, daemon=True).start()
        try:
            ok = d.pick(obj, place) if kind == 'pick' else d.place(obj, place)
        except Exception as e:
            ok = False
            d.say('fail', str(e)[:200])
        stop.set()
        results[did] = {'ok': ok, 'sec': round(time.time() - t0, 1), 'max_tilt_deg': round(tilt[0], 1)}
        if 'apple' not in obj:
            d.say('info', f'{obj}: max tilt from upright during this run {tilt[0]:.0f} deg (sim truth)')
        if d.vid:
            d.vid.add('ok' if ok else 'fail', f'{did}: ' + ('SUCCESS' if ok else 'FAILED') + f' in {time.time() - t0:.0f} s')
            print('  video', d.vid.close(), flush=True)
            d.vid = None
        print(f'== {did}: {results[did]}', flush=True)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
