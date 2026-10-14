#!/usr/bin/env python3
"""Wrist-camera-only re-check of the two ends of the height range (2026-10-11, owner's request):

H  high >= 1.2 m, FRONT, lift 1.10: movej::pre_pick_high (front) -> fine_move (pre-grasp 0.10 m back along the
   approach axis -> grasp -> back holding a 6x6x8 cm box -> pre_pick -> give). No head camera at all: the head
   only counts as a body to hit (tilt chosen among -50 / 0 / +10 for the least trouble). Approach axis: the
   pre_pick tool axis, or that axis pitched 20 deg down / up. Two furniture shapes: 'bay' (a board 0.45 m above
   the object, side panels, back) and 'open' (top of a cabinet: support board + back panel only).
L  low 0.0-0.2 m, left / right / front: approach_lying -> top-down fine_move, wrist camera only. Lift down to
   kcare 0.51 (5 cm below today's 0.56 floor; the owner will lower the lift) vs the current 0.56.

The link2 <-> lift column margin is waived (kcare's own give / approach_lying sit 4.7-4.9 cm from it).
Writes /sim/out/reach/proc3_<H|L>.json and the offending parts of every failed cell.

    python3 scripts/proc_sweep3.py H|L [--procs 4]
"""
import argparse, collections, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
import reach_sweep as R                                    # noqa: E402
import proc_sweep as P                                     # noqa: E402

WAIVE = {frozenset({'link2', 'lift_base_link'}): 0.0}   # pair -> its own clearance (default M_BODY)
OFF = collections.Counter()                                # (segment, my part, other) of margin / penetration hits
SEG = ['?']


def check(self, near_surface=False, allow_tips_env=False, waive_torso=False):
    """reach_sweep's check with the link2 <-> lift column margin waived; records the offending pairs."""
    pen, viol, mind = False, False, {}
    d = self.d
    self.mj.mj_collision(self.m, d)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        k1, k2 = self.cat[g1], self.cat[g2]
        a1, a2 = k1 in ('arm', 'obj'), k2 in ('arm', 'obj')
        if not (a1 or a2):
            continue
        dist = float(c.dist)
        if a1 and a2:
            if 'obj' in (k1, k2):
                continue
            pair = tuple(sorted((self.bname[g1], self.bname[g2])))
            if dist < -0.003 and pair not in self.benign:
                pen = True; OFF[(SEG[0], 'PEN', pair[0], pair[1])] += 1
            continue
        kind = k2 if a1 else k1
        mine = self.bname[g1] if a1 else self.bname[g2]
        other = self.bname[g2] if a1 else self.bname[g1]
        mind[kind] = min(mind.get(kind, 9), dist)
        if kind == 'env':
            if dist < -0.003 and not (allow_tips_env and mine in R.TIPS | {'obj'}):
                pen = True; OFF[(SEG[0], 'PEN', mine, other)] += 1
            elif dist < R.M_ENV and not (near_surface and mine in R.NEAR_SURFACE_OK):
                viol = True; OFF[(SEG[0], 'gap', mine, other)] += 1
        else:
            pair = tuple(sorted((self.bname[g1], self.bname[g2])))
            if dist < -0.003 and pair not in self.benign:
                pen = True; OFF[(SEG[0], 'PEN', mine, other)] += 1
            elif dist < WAIVE.get(frozenset({mine, other}), R.M_BODY) and not (waive_torso and kind == 'torso'):
                viol = True; OFF[(SEG[0], 'gap', mine, other)] += 1
    return pen, viol, mind


R.Col.check = check
NAMES = ['lift@give', 'movej', 'movej2', 'fine_pre', 'grasp', 'liftoff', 'back', 'to_give', 'to_give2']


def run_path(col, segs, mode, tilt):
    pen = viol = False
    for k, s in enumerate(segs):
        SEG[0] = NAMES[k] if len(segs) == 9 else ['lift@give', 'movej', 'fine_pre', 'grasp', 'liftoff', 'back', 'to_give'][k]
        p, v, *_ = P.check_path(col, [s], mode, tilt)
        pen |= p; viol |= v
        if pen:
            break
    return pen, viol


def env_open(mode, zo, d, a):
    """Top of a cabinet: the support board + a back panel down to the floor, nothing above."""
    u, v = R.mode_frame(mode, P.DIRS)
    yaw = math.atan2(u[1], u[0])
    e = max(d - 0.10, 0.238 + 0.03)
    mid = u * (e + 0.15) + v * a; sup = mid.copy(); sup[2] = zo - 0.01
    back = u * (e + 0.31) + v * a; back[2] = zo / 2
    return {'env_shelf': (sup, [0.15, 0.4, 0.01], yaw), 'env_back': (back, [0.01, 0.42, zo / 2], yaw)}


def pitch(z, deg):
    """Rotate the approach axis z by deg about the horizontal axis perpendicular to it (+ = tip down)."""
    h = np.array([z[0], z[1], 0.0]); n = np.linalg.norm(h)
    if n < 1e-6:
        return z
    side = np.cross(h / n, [0, 0, 1.0])
    t = math.radians(deg)
    return z * math.cos(t) + np.cross(side, z) * math.sin(t) + side * (side @ z) * (1 - math.cos(t))


def cell(col, proc, mode, zo, target, env, q_proc, lifts, zdirs, tilts, retract=0.0, via_pre=False):
    q_give = np.radians(R.POSES['give'][mode])
    col.set_env(env)
    res = dict(ik=False, free=None, safe=None)
    for lj in lifts:
        Tp = P.tool(q_proc, lj)
        base_z = Tp[:3, 2] / np.linalg.norm(Tp[:3, 2])
        for zname, zdir in zdirs(base_z):
            pre = target - zdir * P.FINE_BACK
            bg = R.ik({'lift_joint': lj}, target, zdir, False, [q_proc])
            if not (bg[1] < 0.01 and bg[2] < 5):
                continue
            bp = R.ik({'lift_joint': lj}, pre, zdir, False, [bg[0], q_proc])
            if not (bp[1] < 0.01 and bp[2] < 5):
                continue
            qg, qp = bg[0], bp[0]
            q_via = q_proc
            if retract:                     # a stand-off point on the approach line instead of the fixed pre_pick
                br = R.ik({'lift_joint': lj}, target - zdir * retract, zdir, False, [qp, q_proc])
                if not (br[1] < 0.01 and br[2] < 5):
                    continue
                q_via = br[0]
            res['ik'] = True
            lj0 = 0.37
            first = ([(q_give, q_proc, lj, lj, 24, False, False, False, 'start'),
                      (q_proc, q_via, lj, lj, 16, False, False, False, None)] if via_pre and retract else
                     [(q_give, q_via, lj, lj, 24, False, False, False, 'start')])
            segs = [(q_give, q_give, lj0, lj, 10, False, False, False, 'all')] + first + [
                    (q_via, qp, lj, lj, 12, False, False, False, None),
                    (qp, qg, lj, lj, 8, False, True, False, None),
                    (qg, qp, lj, lj, 8, True, True, True, None),
                    (qp, q_via, lj, lj, 12, True, False, False, None),
                    (q_via, q_give, lj, lj, 24, True, False, False, 'end')]
            if via_pre and retract:          # and back the same way: stand-off -> pre_pick_high -> give
                segs = segs[:-1] + [(q_via, q_proc, lj, lj, 16, True, False, False, None),
                                    (q_proc, q_give, lj, lj, 24, True, False, False, 'end')]
            for tilt in tilts:
                pen, viol = run_path(col, segs, mode, tilt)
                if pen:
                    continue
                sol = dict(lift_kcare=round(lj + R.MOBILE_H, 2), axis=zname, tilt=tilt, viol=bool(viol))
                res['free'] = res['free'] or sol
                if not viol:
                    res['safe'] = sol
                    return res
    return res


def job(args):
    proc, mode, zo, variant = args
    if 'COL' not in globals():
        R.setup(); globals()['COL'] = R.Col()
    col = COL
    OFF.clear()
    u, v = R.mode_frame(mode, P.DIRS)
    cells = {}
    if proc == 'H':
        q_proc = np.radians(R.POSES['pre_pick_high']['front'])
        T = P.tool(q_proc, P.LIFT_TOP)
        d0, a0 = float(T[:3, 3] @ u), float(T[:3, 3] @ v)
        ds = [round(d0 + k * 0.1, 2) for k in range(-3, 4) if d0 + k * 0.1 >= 0.3]
        as_ = [round(a0 + k * 0.1, 2) for k in range(-3, 4)]
        zdirs = lambda z: [('kept', z), ('down20', pitch(z, 20)), ('up20', pitch(z, -20))]
        for d in ds:
            for a in as_:
                env = R.env_for(mode, zo, d, a, P.DIRS) if variant.startswith('bay') else env_open(mode, zo, d, a)
                target = u * d + v * a; target[2] = zo + P.GZ['A']
                cells[f'{d:.2f},{a:.2f}'] = cell(col, proc, mode, zo, target, env, q_proc, [P.LIFT_TOP], zdirs, (10, 0, -50),
                                                 retract=0.30 if 'retract' in variant else 0.0, via_pre=variant.endswith('via'))
    else:
        q_proc = np.radians(P.APPROACH[mode])
        lmin = 0.08 if variant == 'lift-5cm' else 0.13
        lifts = [round(x, 3) for x in np.arange(lmin, 0.40, 0.03)]
        T = P.tool(q_proc, 0.3)
        d0, a0 = float(T[:3, 3] @ u), float(T[:3, 3] @ v)
        ds = [round(d0 + k * 0.1, 2) for k in range(-3, 4) if d0 + k * 0.1 >= 0.3]
        as_ = [round(a0 + k * 0.1, 2) for k in range(-3, 4)]
        zdirs = lambda z: [('down', np.array([0, 0, -1.0]))]
        for d in ds:
            for a in as_:
                target = u * d + v * a; target[2] = zo + P.GZ['B']
                cells[f'{d:.2f},{a:.2f}'] = cell(col, proc, mode, zo, target, P.table_env(mode, zo, d, a), q_proc, lifts,
                                                 zdirs, (-50,))
    return proc, mode, zo, variant, cells, [[list(k), n] for k, n in OFF.most_common(8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('proc', choices=['H', 'L'])
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--vars', default='', help="H: bay,open,bay-retract,open-retract")
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    VARS = a.vars.split(',') if a.vars else ('bay', 'open')
    if a.proc == 'H':
        jobs = [('H', 'front', z, var) for var in VARS for z in (1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8)]
    else:
        jobs = [('L', m, z, var) for var in ('lift-5cm', 'lift-now') for z in (0.0, 0.03, 0.05, 0.1, 0.15, 0.2)
                for m in ('left', 'right', 'front')]
    out, t0 = {}, time.time()
    with Pool(a.procs) as pool:
        for k, (p, m, z, var, cells, off) in enumerate(pool.imap_unordered(job, jobs)):
            out[f'{var}|{m}|{z}'] = dict(cells=cells, offenders=off)
            print(f'{k + 1}/{len(jobs)} {var} {m} {z} {time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(f'{SIM}/out/reach/proc3_{a.proc}{a.tag}.json', 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
