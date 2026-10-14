#!/usr/bin/env python3
"""Measure kcare's two real pick procedures with the full-geometry checker of reach_sweep.py (v2 margins).

A  high ("move_top" + fine_move): head locates the object (tilt up to +10 deg), lift::highest (kcare 1.10),
   movej::pre_pick (ARM_CONFIGS pre_pick left/right; front = pre_pick_high for comparison), [forward::0.1],
   fine_move: pre_pick -> pre-grasp 0.10 m back along the pre_pick tool axis -> grasp (approach axis kept,
   roll free), back the same way holding a 6x6x8 cm box, back to give. Objects on a shelf bay 0.9-1.8 m.
B  low (approach_lying + top-down fine_move): head down locates, lift as low as works (kcare 0.56-1.10),
   movej::approach_lying (left/right; front = B's approach_lying front), fine_move top-down: pre-grasp 0.10 m
   above -> grasp, back holding the box, back to give. Objects on a support 0.0-0.8 m (0.44 = sofa seat).

    python3 scripts/proc_sweep.py --proc A|B [--procs 2]
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
import reach_sweep as R                                    # noqa: E402

PRE_PICK = {'right': [8.743469444279539, 41.818468400995876, 18.121398575656148, 52.97783295754323,
                      -294.22176393307814, 68.28793762365318, 16.83512027568642],
            'left': [14.24531232545081, 46.09995199165086, -35.125928033326396, 47.90983942358294,
                     -51.51324573910115, 58.453530728345335, -21.345445939388096],
            'front': R.POSES['pre_pick_high']['front']}
APPROACH = {m: R.POSES['approach_lying'][m] for m in ('left', 'right', 'front')}
DIRS = {'left': [0.0, -1.0], 'right': [0.0, 1.0], 'front': [1.0, 0.0]}
LIFT_TOP = 0.67                                             # lift_joint at kcare 1.10
FINE_BACK = 0.10
HEIGHTS = {'A': [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
           'B': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.44, 0.55, 0.65, 0.75, 0.8]}
GZ = {'A': 0.08, 'B': 0.04}


def tool(q, lj):
    return R.ARMC.fk(dict({'lift_joint': lj}, **dict(zip(R.ARM, q))))


def grid_for(proc, mode):
    """(d, a) grid in the mode frame centred on the procedure pose's TCP (+-0.3 m, 0.1 m steps)."""
    q = np.radians(PRE_PICK[mode] if proc == 'A' else APPROACH[mode])
    T = tool(q, LIFT_TOP if proc == 'A' else 0.3)
    u, v = R.mode_frame(mode, DIRS)
    d0, a0 = float(T[:3, 3] @ u), float(T[:3, 3] @ v)
    ds = [round(d0 + k * 0.1, 2) for k in range(-3, 4) if d0 + k * 0.1 >= 0.3]
    as_ = [round(a0 + k * 0.1, 2) for k in range(-3, 4)]
    return ds, as_, (round(d0, 3), round(a0, 3)), T


def check_path(col, segs, mode, tilt):
    """segs: [(qa, qb, n, holding, near_surface, tips_ok, waive_torso_at)] -> (pen, viol, above, lens, mind)."""
    pen = viol = above = lens = False; mind = {}
    for qa, qb, lja, ljb, n, hold, near, tips_ok, waive in segs:
        col.holding(hold)
        for k, t in enumerate(np.linspace(0, 1, n)):
            q = R.wrap_interp(qa, qb, n)[k]
            lj = lja + (ljb - lja) * t
            col.set_q(q, lj, mode, tilt)
            wt = (waive == 'start' and k < 5) or (waive == 'end' and k >= n - 5) or waive == 'all'
            p, vio, md = col.check(near_surface=near or (hold and k < 4), allow_tips_env=tips_ok, waive_torso=wt)
            pen |= p; viol |= vio
            for kk, vv in md.items():
                mind[kk] = min(mind.get(kk, 9), vv)
            if k % 3 == 0:
                a_, l_ = col.head_flags(); above |= a_; lens |= l_
            if pen:
                col.holding(False); return pen, viol, above, lens, mind
    col.holding(False)
    return pen, viol, above, lens, mind


def job(args):
    proc, mode, zo = args
    if 'COL' not in globals():
        R.setup(); globals()['COL'] = R.Col()
    col = COL
    u, v = R.mode_frame(mode, DIRS)
    ds, as_, c0, T0 = grid_for(proc, mode)
    q_give = np.radians(R.POSES['give'][mode])
    q_proc = np.radians(PRE_PICK[mode] if proc == 'A' else APPROACH[mode])
    cells = {}
    for d in ds:
        for a in as_:
            col.set_env(R.env_for(mode, zo, d, a, DIRS) if proc == 'A' else table_env(mode, zo, d, a))
            target = u * d + v * a; target[2] = zo + GZ[proc]
            # 1) head locates the object (arm at give, lift low; the head does not ride the lift)
            col.set_q(q_give, 0.13, mode, -40); loc = col.view(target)
            res = dict(locate_tilt=loc[0], locate_visible=bool(loc[1] and not loc[2]), ik=False, free=None, safe=None, strict=None)
            lifts = [LIFT_TOP] if proc == 'A' else [round(x, 3) for x in np.arange(0.13, 0.671, 0.06)]
            for lj in lifts:
                Tp = tool(q_proc, lj)
                if proc == 'A':
                    zdir = Tp[:3, 2] / np.linalg.norm(Tp[:3, 2])       # pre_pick approach axis kept
                else:
                    zdir = np.array([0, 0, -1.0])
                pre = target - zdir * FINE_BACK
                bg = R.ik({'lift_joint': lj}, target, zdir, False, [q_proc])
                if not (bg[1] < 0.01 and bg[2] < 5):
                    continue
                bp = R.ik({'lift_joint': lj}, pre, zdir, False, [bg[0], q_proc])
                if not (bp[1] < 0.01 and bp[2] < 5):
                    continue
                res['ik'] = True
                qg, qp = bg[0], bp[0]
                col.set_q(qp, lj, mode, -40); vp = col.view(target)
                tilt = vp[0] if vp[0] is not None else (10 if proc == 'A' else -50)
                lj0 = 0.37                                                # mode home lift (kcare 0.80)
                segs = [(q_give, q_give, lj0, lj, 10, False, False, False, 'all'),        # lift with the arm at give
                        (q_give, q_proc, lj, lj, 24, False, False, False, 'start'),       # movej pre_pick / approach_lying
                        (q_proc, qp, lj, lj, 12, False, False, False, None),              # fine_move to the pre-grasp
                        (qp, qg, lj, lj, 8, False, True, False, None),                    # grasp
                        (qg, qp, lj, lj, 8, True, True, True, None),                      # lift off with the object
                        (qp, q_proc, lj, lj, 12, True, False, False, None),
                        (q_proc, q_give, lj, lj, 24, True, False, False, 'end')]
                pen, viol, above, lens, mind = check_path(col, segs, mode, tilt)
                if pen:
                    continue
                sol = dict(lift_kcare=round(lj + R.MOBILE_H, 2), viol=bool(viol), above=bool(above), lens=bool(lens),
                           mind={k: round(x, 3) for k, x in mind.items()}, pre_view=vp[:4])
                res['free'] = res['free'] or sol
                if not viol:
                    res['safe'] = res['safe'] or sol
                    if not above and not lens:
                        res['strict'] = sol; break
            cells[f'{d:.2f},{a:.2f}'] = res
    return proc, mode, zo, c0, cells


def table_env(mode, zo, d, a):
    if zo <= 0.001:
        return {}
    u, v = R.mode_frame(mode, DIRS)
    half_u = 0.238 if mode == 'front' else 0.215
    e = max(d - 0.10, half_u + 0.03)
    c = u * (e + 0.3); c[2] = zo - 0.02
    yaw = math.atan2(u[1], u[0])
    return {'env_table': (c, [0.3, 0.6, 0.02], yaw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proc', choices=['A', 'B'], required=True)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--out', default=f'{SIM}/out/reach')
    a = ap.parse_args()
    jobs = [(a.proc, m, z) for z in HEIGHTS[a.proc] for m in ('left', 'right', 'front')]
    t0 = time.time(); out = {}
    with Pool(a.procs) as pool:
        for k, (p, m, z, c0, cells) in enumerate(pool.imap_unordered(job, jobs)):
            out[f'{m}|{z}'] = dict(centre=c0, cells=cells)
            print(f'{k + 1}/{len(jobs)} {m} {z} {time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(f'{a.out}/proc_{a.proc}.json', 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
