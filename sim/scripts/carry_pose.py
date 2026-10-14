#!/usr/bin/env python3
"""Search a front-mode CARRY pose for a horizontally grasped object (a cup of water stays upright):
gripper horizontal, facing forward, TCP as close to the body and as low as the arm allows, holding a 6x6x8 cm
box, no collision and the body / head margins kept. Reports candidates sorted by how far the TCP sticks out.

    python3 scripts/carry_pose.py   -> prints joint angles (deg) + TCP + clearances
"""
import os, sys
import numpy as np

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
import reach_sweep as R                                    # noqa: E402
import proc_sweep as P                                     # noqa: E402
import proc_sweep3 as P3                                   # noqa: E402

PP1 = np.radians([-4.56, -9.06, 3.99, 31.45, -1.34, -49.55, -179.26])
R.setup(); col = R.Col(); col.set_env({})
import math


def ok_at(q, lj):
    col.holding(True); col.set_q(q, lj, 'front', -50); pen, viol, md = col.check(); col.holding(False)
    return pen, viol, md


def path(q_start, p0, yaw0, p1, yaw1, lj, n=30):
    """Cartesian path with the gripper kept horizontal: TCP p0 -> p1, approach yaw yaw0 -> yaw1 (deg).
    Returns (joint list, max angle jump between steps) or None when IK breaks / a collision appears."""
    q, qs, jump = q_start, [], 0.0
    for t in np.linspace(0, 1, n):
        p = p0 + (p1 - p0) * t; yw = math.radians(yaw0 + (yaw1 - yaw0) * t)
        z = np.array([math.cos(yw), math.sin(yw), 0.0])
        b = R.ik({'lift_joint': lj}, p, z, False, [q])
        if not (b[1] < 0.01 and b[2] < 5):
            return None
        jump = max(jump, float(np.max(np.abs(b[0] - q))))
        q = b[0]
        pen, viol, md = ok_at(q, lj)
        if pen or (viol and t > 0.1):
            return None
        qs.append(q)
    return qs, math.degrees(jump)


lj = 0.13
T = P.tool(PP1, lj); p_grasp = np.array([float(T[0, 3]), 0.0, float(T[2, 3])])   # where the horizontal grasp ends
print('grasp end tcp', p_grasp.round(3))
rows = []
for yaw1 in (0, 30, 60, 90, -30, -60, -90):
    for x1 in (0.25, 0.35, 0.45, 0.55):
        for y1 in (-0.30, -0.15, 0.0, 0.15, 0.30):
            for dz in (0.0, -0.15):
                p1 = np.array([x1, y1, p_grasp[2] + dz])
                r = path(PP1, p_grasp, 0.0, p1, yaw1, lj)
                if r is None:
                    continue
                qs, jump = r
                pen, viol, md = ok_at(qs[-1], lj)
                if viol:
                    continue
                rows.append((round(math.hypot(x1, y1), 3), x1, y1, round(float(p1[2]), 2), yaw1, round(jump, 1), {k: round(v, 3) for k, v in md.items()},
                             np.degrees(qs[-1]).round(1).tolist()))
rows.sort()
print(len(rows), 'reachable carry poses (lift 0.56); closest to the base centre first:')
for r in rows[:12]:
    print(f'  r {r[0]:.2f}  tcp ({r[1]:.2f}, {r[2]:.2f}, {r[3]:.2f}) yaw {r[4]:4d}  max step {r[5]} deg  clear {r[6]}  q {r[7]}')
