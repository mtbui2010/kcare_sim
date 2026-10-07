#!/usr/bin/env python3
"""Acceptance checks for the MuJoCo kaair model.

    python3 check_sim.py slamtec        # writes /sim/out/<variant>/report.json + PNGs

  1. FK      — every link pose in MuJoCo vs a plain-numpy FK of the ORIGINAL
               xacro output (urdf/raw), at random joint configs. Must be ~0:
               the sim pipeline may change inertia/collision, never kinematics.
  2. self-contact at home — robot geoms that touch each other at rest (these
               need a <contact><exclude> or the robot fights itself).
  3. hold    — home pose held for 3 s under gravity (gravcomp + position servos).
  4. step    — per-joint step response of the arm (tracking error after 1 s).
  5. grasp   — arm posed gripper-down over a 4 cm cube on a table, close the
               gripper, raise the lift 10 cm: the cube must come up with it.
  6. render  — head / wrist / overview images at home and at the grasp.
"""

import argparse
import json
import os

import mujoco
import numpy as np
import yaml
from lxml import etree

SIM = os.environ.get('SIM_ROOT', '/sim')


# ── reference FK straight from the URDF ──────────────────────────────────────
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class UrdfFK:
    def __init__(self, path):
        root = etree.parse(path).getroot()
        self.joints = {}
        children = set()
        for j in root.findall('joint'):
            o = j.find('origin')
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = rpy_to_R(*[float(v) for v in o.get('rpy', '0 0 0').split()])
                T[:3, 3] = [float(v) for v in o.get('xyz', '0 0 0').split()]
            ax = j.find('axis')
            mm = j.find('mimic')
            lim = j.find('limit')
            self.joints[j.get('name')] = dict(
                type=j.get('type'), parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                T=T, axis=np.array([float(v) for v in ax.get('xyz').split()]) if ax is not None else np.array([1., 0, 0]),
                mimic=(mm.get('joint'), float(mm.get('multiplier', 1)), float(mm.get('offset', 0))) if mm is not None else None,
                limit=(float(lim.get('lower', 0)), float(lim.get('upper', 0))) if lim is not None else None)
            children.add(j.find('child').get('link'))
        links = [l.get('name') for l in root.findall('link')]
        self.root = [l for l in links if l not in children][0]
        self.by_parent = {}
        for n, j in self.joints.items():
            self.by_parent.setdefault(j['parent'], []).append(n)

    def active(self):
        return [n for n, j in self.joints.items() if j['type'] in ('revolute', 'prismatic', 'continuous') and not j['mimic']]

    def fk(self, q):
        out = {self.root: np.eye(4)}
        stack = [self.root]
        while stack:
            p = stack.pop()
            for n in self.by_parent.get(p, []):
                j = self.joints[n]
                v = q.get(n, 0.0)
                if j['mimic']:
                    v = q.get(j['mimic'][0], 0.0) * j['mimic'][1] + j['mimic'][2]
                M = np.eye(4)
                if j['type'] in ('revolute', 'continuous'):
                    M[:3, :3] = axis_angle(j['axis'], v)
                elif j['type'] == 'prismatic':
                    M[:3, 3] = j['axis'] / np.linalg.norm(j['axis']) * v
                out[j['child']] = out[p] @ j['T'] @ M
                stack.append(j['child'])
        return out


def ik_down(ref, q0, arm, target, rng, tries=30):
    """7-DoF IK on the reference FK: tool_tcp_link at `target`, its z-axis
    pointing at the floor, fingers opening along world y. Returns (q, T) or None."""
    from scipy.optimize import least_squares
    lo = np.array([ref.joints[j]['limit'][0] for j in arm])
    hi = np.array([ref.joints[j]['limit'][1] for j in arm])
    # keep the ±2π joints within ±π so the solution stays near the home branch
    lo, hi = np.maximum(lo, -np.pi), np.minimum(hi, np.pi)

    def res(x):
        T = ref.fk(dict(q0, **dict(zip(arm, x))))['tool_tcp_link']
        return np.concatenate([(T[:3, 3] - target) * 10, T[:3, 2] - [0, 0, -1], T[:3, 1] - [0, 1, 0]])

    seed0 = np.clip([q0[j] for j in arm], lo + 1e-3, hi - 1e-3)
    for k in range(tries):
        x0 = seed0 if k == 0 else rng.uniform(lo, hi)
        sol = least_squares(res, x0, bounds=(lo, hi))
        if np.linalg.norm(sol.fun) < 1e-3:
            q = dict(q0, **dict(zip(arm, sol.x)))
            return q, ref.fk(q)['tool_tcp_link']
    return None


# ── MuJoCo helpers ───────────────────────────────────────────────────────────
def set_q(m, d, q, cfg):
    g = cfg['gripper']
    for name, v in q.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = v
    if g['driver'] in q:
        for f in g['fingers']:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f)
            d.qpos[m.jnt_qposadr[jid]] = g['ratio'] * q[g['driver']]


def get_q(m, d, name):
    return float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]])


def ctrl_from(m, q, cfg):
    return np.array([q[a['joint']] for a in cfg['actuators']])


def body_T(m, d, name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    T = np.eye(4)
    T[:3, :3] = d.xmat[bid].reshape(3, 3)
    T[:3, 3] = d.xpos[bid]
    return T


def body_min_z(m, d, body):
    """Lowest world z of a body's collision geoms (AABB corners)."""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    zmin = np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_contype[g]:
            a = m.geom_aabb[g]
            w = (d.geom_xmat[g].reshape(3, 3) @ (signs * a[3:] + a[:3]).T).T + d.geom_xpos[g]
            zmin = min(zmin, w[:, 2].min())
    return zmin


def robot_geom(m, gid):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) not in (None, 'world', 'cube')


def render(m, d, cam, path, w=640, h=360):
    import imageio
    r = mujoco.Renderer(m, h, w)
    if cam == 'overview':
        c = mujoco.MjvCamera()
        c.lookat[:] = [0.4, 0, 0.8]
        c.distance, c.azimuth, c.elevation = 2.6, 150, -20
        r.update_scene(d, c)
    else:
        r.update_scene(d, cam)
    imageio.imwrite(path, r.render())
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['slamtec', 'clober'])
    ap.add_argument('--n-fk', type=int, default=200)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(SIM, 'config', 'mjcf.yaml')))
    out_dir = os.path.join(SIM, 'out', args.variant)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    rep = {}

    ref = UrdfFK(os.path.join(SIM, 'urdf', 'raw', f'kaair_{args.variant}.urdf'))
    m = mujoco.MjModel.from_xml_path(os.path.join(SIM, 'mjcf', f'scene_tabletop_{args.variant}.xml'))
    home = dict(cfg['home'])
    # tests 1-4 are about the robot alone: park the table and cube far away
    # (step 5 brings the table back under the gripper)
    m.geom_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table')] = [5.0, 0.0, 0.35]
    cadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free')]
    m.qpos0[cadr:cadr + 3] = [5.0, 0.0, 0.725]
    d = mujoco.MjData(m)

    # 1. FK ------------------------------------------------------------------
    active = ref.active()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    common = [n for n in names if n in ref.fk({}) and n != ref.root]
    base = ref.root if ref.root in names else None
    perr = rerr = 0.0
    worst = ''
    for _ in range(args.n_fk):
        q = {n: rng.uniform(*ref.joints[n]['limit']) for n in active}
        set_q(m, d, q, cfg)
        mujoco.mj_kinematics(m, d)
        Tr = ref.fk(q)
        B = body_T(m, d, base) if base else np.eye(4)
        for n in common:
            Tm = np.linalg.inv(B) @ body_T(m, d, n)
            e = np.linalg.norm(Tm[:3, 3] - Tr[n][:3, 3])
            er = np.arccos(np.clip((np.trace(Tm[:3, :3].T @ Tr[n][:3, :3]) - 1) / 2, -1, 1))
            if e > perr:
                perr, worst = e, n
            rerr = max(rerr, er)
    rep['fk'] = dict(configs=args.n_fk, links_compared=len(common), max_pos_err_m=perr,
                     max_rot_err_rad=rerr, worst_link=worst, pass_=bool(perr < 1e-5 and rerr < 1e-5))
    print(f"[1] FK   {len(common)} links × {args.n_fk} configs: max pos err {perr:.2e} m ({worst}), "
          f"max rot err {rerr:.2e} rad -> {'PASS' if rep['fk']['pass_'] else 'FAIL'}")

    # 2. self contact at home -----------------------------------------------
    mujoco.mj_resetData(m, d)
    set_q(m, d, home, cfg)
    mujoco.mj_forward(m, d)
    pairs = {}
    for c in d.contact[:d.ncon]:
        if robot_geom(m, c.geom1) and robot_geom(m, c.geom2):
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])
            k = tuple(sorted((b1, b2)))
            pairs[k] = min(pairs.get(k, 0.0), float(c.dist))
    rep['self_contact_home'] = [dict(bodies=list(k), depth_m=v) for k, v in sorted(pairs.items())]
    print(f"[2] self-contacts at home: {len(pairs)}", *[f'{k[0]} <-> {k[1]} ({v*1000:.1f} mm)' for k, v in pairs.items()], sep='\n      ')

    # 3. hold ----------------------------------------------------------------
    d.ctrl[:] = ctrl_from(m, home, cfg)
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = {a['joint']: abs(get_q(m, d, a['joint']) - home[a['joint']]) for a in cfg['actuators']}
    rep['hold'] = dict(max_drift=max(drift.values()), per_joint=drift, pass_=bool(max(drift.values()) < 5e-3))
    print(f"[3] hold 3 s: max drift {max(drift.values()):.2e} ({max(drift, key=drift.get)}) -> {'PASS' if rep['hold']['pass_'] else 'FAIL'}")
    render(m, d, 'overview', f'{out_dir}/home_overview.png')
    for cam in cfg['cameras']:
        render(m, d, cam['name'], f"{out_dir}/home_{cam['name']}.png")

    # 4. step response -------------------------------------------------------
    step = {}
    for a in cfg['actuators'][:7]:
        j = a['joint']
        mujoco.mj_resetData(m, d)
        set_q(m, d, home, cfg)
        tgt = dict(home)
        lo, hi = ref.joints[j]['limit']
        tgt[j] = float(np.clip(home[j] + 0.3, lo, hi)) if home[j] + 0.3 < hi else home[j] - 0.3
        d.ctrl[:] = ctrl_from(m, tgt, cfg)
        for _ in range(int(1.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        step[j] = abs(get_q(m, d, j) - tgt[j])
    rep['step_0.3rad_err_after_1s'] = step
    print('[4] step 0.3 rad, err after 1 s:', ' '.join(f'{k}={v:.1e}' for k, v in step.items()))

    # 5. grasp ---------------------------------------------------------------
    lift0 = 0.35
    arm = [f'joint{i}' for i in range(1, 8)]
    q0 = dict(home, lift_joint=lift0)
    base_T = ref.fk(q0)['arm_base']
    # gripper pointing down, 0.45 m in front of the arm base, 0.10 m below it
    target = base_T[:3, 3] + np.array([0.45, 0.0, -0.10])
    best = ik_down(ref, q0, arm, target, rng)
    if best is None:
        rep['grasp'] = dict(pass_=False, reason='IK found no gripper-down config')
        print('[5] grasp: IK found no gripper-down config')
    else:
        q, T = best
        tcp = T[:3, 3]
        mujoco.mj_resetData(m, d)
        q['virtual_gripper_joint'] = 0.1
        set_q(m, d, q, cfg)
        mujoco.mj_forward(m, d)
        # table top 2 mm below the fingertips, cube standing on it between the fingers
        top = min(body_min_z(m, d, f) for f in ('left_finger_tip', 'right_finger_tip')) - 0.002
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table')
        m.geom_pos[tid] = [tcp[0], tcp[1], top - m.geom_size[tid][2]]
        cj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free')
        adr = m.jnt_qposadr[cj]
        cube_half = cfg['scene']['cube']['half']
        d.qpos[adr:adr + 3] = [tcp[0], tcp[1], top + cube_half + 1e-4]
        d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        d.ctrl[:] = ctrl_from(m, q, cfg)
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        render(m, d, 'overview', f'{out_dir}/grasp_0_before.png')
        for cam in cfg['cameras']:
            render(m, d, cam['name'], f"{out_dir}/grasp_0_{cam['name']}.png")
        z0 = float(d.qpos[adr + 2])
        q['virtual_gripper_joint'] = 0.0
        d.ctrl[:] = ctrl_from(m, q, cfg)
        for _ in range(int(1.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        width = get_q(m, d, 'virtual_gripper_joint')
        q['lift_joint'] = lift0 + 0.10
        d.ctrl[:] = ctrl_from(m, q, cfg)
        for _ in range(int(3.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        dz = float(d.qpos[adr + 2]) - z0
        render(m, d, 'overview', f'{out_dir}/grasp_1_lifted.png')
        render(m, d, 'wrist_rgb', f'{out_dir}/grasp_1_wrist_rgb.png')
        rep['grasp'] = dict(arm_q={j: q[j] for j in arm}, tcp=tcp.tolist(), gripper_width_closed=width,
                            cube_dz=dz, lift_dz=0.10, pass_=bool(abs(dz - 0.10) < 0.01 and width > 0.03))
        print(f"[5] grasp: gripper closed to {width*1000:.1f} mm (cube 40 mm), cube rose {dz*1000:.1f} mm "
              f"for a 100 mm lift -> {'PASS' if rep['grasp']['pass_'] else 'FAIL'}")

    json.dump(rep, open(f'{out_dir}/report.json', 'w'), indent=2, default=float)
    print('report + images ->', out_dir)


if __name__ == '__main__':
    main()
