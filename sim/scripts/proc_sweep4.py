#!/usr/bin/env python3
"""Front-mode HORIZONTAL grasp with the real robot's two-step pre_pick (2026-10-11):

    give -> pre_pick[0] (gripper down, tcp lift + 0.40: clears the furniture) -> pre_pick[1] (gripper forward,
    tcp (0.815, 0, lift + 0.44)) -> pre-grasp (0.15 m short of the object on the horizontal approach line, at the
    object's height) -> grasp -> lift 2 cm -> back to the pre-grasp -> pre_pick[1] -> pre_pick[0] -> give

for objects at 0.44-1.31 m on a table (< 1.0 m) or a shelf bay (>= 1.0 m), every lift 0.56-1.10. Same checker as
proc_sweep3 (full collision + margins; link2 <-> lift column margin waived), no head-camera condition.
Also reports the 'carry' metric: the gripper's tilt from horizontal along the whole path (a cup of water).

    python3 scripts/proc_sweep4.py [--procs 4]   -> /sim/out/reach/proc4.json
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
import reach_sweep as R                                    # noqa: E402
import proc_sweep as P                                     # noqa: E402
import proc_sweep3 as P3                                   # noqa: E402  (installs the check with the waiver)

# the real pre_pick[0] itself puts link3 2.5-2.7 cm from the lift column (measured): 2 cm for that pair
P3.WAIVE[frozenset({'link3', 'lift_base_link'})] = 0.02

PP = [[0, -25, 0, 75, 0, 100, -180], [-4.56, -9.06, 3.99, 31.45, -1.34, -49.55, -179.26]]
HEIGHTS = [0.44, 0.55, 0.65, 0.72, 0.8, 0.9, 1.0, 1.1, 1.2, 1.31]
LIFTS = [round(x, 3) for x in np.arange(0.13, 0.671, 0.06)]       # lift_joint (kcare = + 0.43)
STANDOFF, GZ = 0.15, 0.06
NAMES = ['lift@give', 'give->pp0', 'pp0->pp1', 'pp1->pre', 'grasp', 'liftoff', 'back', 'pre->pp1', 'pp1->pp0', 'pp0->give']


def run_path(col, segs, tilt):
    pen = viol = False
    for k, s in enumerate(segs):
        P3.SEG[0] = NAMES[k]
        p, v, *_ = P.check_path(col, [s], 'front', tilt)
        pen |= p; viol |= v
        if pen:
            break
    return pen, viol


def env(zo, d, a):
    return P.table_env('front', zo, d, a) if zo < 1.0 else R.env_for('front', zo, d, a, P.DIRS)


def job(zo):
    if 'COL' not in globals():
        R.setup(); globals()['COL'] = R.Col()
    col = COL
    P3.OFF.clear()
    q_give = np.radians(R.POSES['give']['front'])
    q0, q1 = np.radians(PP[0]), np.radians(PP[1])
    u, v = R.mode_frame('front', P.DIRS)
    cells = {}
    for d in [0.55, 0.65, 0.75, 0.85, 0.95, 1.05]:
        for a in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            col.set_env(env(zo, d, a))
            target = u * d + v * a; target[2] = zo + GZ
            res = dict(ik=False, free=None, safe=None)
            for lj in LIFTS:
                T1 = P.tool(q1, lj)
                zdir = T1[:3, 2] / np.linalg.norm(T1[:3, 2])
                zdir[2] = 0.0; zdir /= np.linalg.norm(zdir)              # horizontal approach
                pre = target - zdir * STANDOFF
                bp = R.ik({'lift_joint': lj}, pre, zdir, False, [q1])
                if not (bp[1] < 0.01 and bp[2] < 5):
                    continue
                bg = R.ik({'lift_joint': lj}, target, zdir, False, [bp[0], q1])
                if not (bg[1] < 0.01 and bg[2] < 5):
                    continue
                res['ik'] = True
                qp, qg = bp[0], bg[0]
                up = target.copy(); up[2] += 0.02
                bu = R.ik({'lift_joint': lj}, up, zdir, False, [qg])
                qu = bu[0] if (bu[1] < 0.01 and bu[2] < 5) else qg
                lj0 = 0.37
                segs = [(q_give, q_give, lj0, lj, 10, False, False, False, 'all'),
                        (q_give, q0, lj, lj, 20, False, False, False, 'start'),
                        (q0, q1, lj, lj, 20, False, False, False, None),
                        (q1, qp, lj, lj, 16, False, False, False, None),
                        (qp, qg, lj, lj, 8, False, True, False, None),
                        (qg, qu, lj, lj, 4, True, True, True, None),
                        (qu, qp, lj, lj, 8, True, False, False, None),
                        (qp, q1, lj, lj, 16, True, False, False, None),
                        (q1, q0, lj, lj, 20, True, False, False, None),
                        (q0, q_give, lj, lj, 20, True, False, False, 'end')]
                for tilt in ((-50, 10) if zo < 1.0 else (10, -50)):
                    pen, viol = run_path(col, segs, tilt)
                    if pen:
                        continue
                    sol = dict(lift_kcare=round(lj + R.MOBILE_H, 2), viol=bool(viol), tilt=tilt)
                    res['free'] = res['free'] or sol
                    if not viol:
                        res['safe'] = sol
                        break
                if res['safe']:
                    break
            cells[f'{d:.2f},{a:.2f}'] = res
    return zo, cells, [[list(k), n] for k, n in P3.OFF.most_common(6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    out, t0 = {}, time.time()
    with Pool(a.procs) as pool:
        for k, (z, cells, off) in enumerate(pool.imap_unordered(job, HEIGHTS)):
            out[str(z)] = dict(cells=cells, offenders=off)
            n = len(cells); sf = sum(bool(c['safe']) for c in cells.values()); fr = sum(bool(c['free']) for c in cells.values())
            ik = sum(c['ik'] for c in cells.values())
            lifts = sorted({c['safe']['lift_kcare'] for c in cells.values() if c['safe']})
            print(f'{k + 1}/{len(HEIGHTS)} z {z}: ik {ik}/{n} free {fr} safe {sf} lifts {lifts} | {off[:3]}  {time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(f'{SIM}/out/reach/proc4.json', 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
