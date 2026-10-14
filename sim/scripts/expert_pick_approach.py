#!/usr/bin/env python3
"""Scripted expert for PICK phase A, "reach to view" (VLA data): from kcare's ready pose of the mode (movej::give, the
head panned to the mode side and tilted down), bring the finger tips 0.10-0.15 m from the target object (the D405 sits ~0.15 m
behind them, so the wrist camera is ~0.25-0.30 m away), the object centred in the wrist image, coordinating arm + lift + a small forward base shift. fine_move takes over from there on
the robot (wrist-camera grasp pose, close, grasp_succeed).

    /sim/.venv/bin/python scripts/expert_pick_approach.py [--seed 3] [--n 4]   # episodes -> out/pick_approach/*.mp4

Planning (per episode, before moving): the viewpoint = camera position + orientation looking at the target centre from
the robot's side and above (rand['view']: distance, elevation, yaw); 7-DoF IK on the wrist camera frame for every
(lift, base shift) candidate — lift by kcare's rule (surface height + 0.1 m, clipped to the lift range) first, then
+-0.1 m; base shift 0, then 0.1 / 0.2 m toward the object along the heading (never sideways, never turning) — and the
first candidate whose goal pose is collision-free wins. A via point 0.12 m above (and 0.05 m back from) the goal keeps
the arm off the table on the way. ~20 % of episodes (rand['retry']) first stop at a viewpoint 4-8 cm off and then
correct (recovery data). Min-jerk interpolation in joint space, 30 Hz.

State (17) / action (10) — see STATE_NAMES / ACTION_NAMES. The action is absolute: joint targets, gripper width,
lift_joint target [m], base forward offset target [m] since the episode start (|.| <= 0.2).
Success: the target centre projects inside the central 30 % box of the wrist image, finger tips 0.08-0.18 m from it, no
robot contact with the table or any object during the episode, and the target did not move (> 1 cm).
"""
import argparse, math, os, sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(__file__))
import check_sim as cs
import pick_scene

SIM = os.environ.get('SIM_ROOT', '/sim')
HZ = 30
ARM = [f'joint{i}' for i in range(1, 8)]
MODES = ['front', 'left', 'right']
STATE_NAMES = ARM + ['gripper_width', 'lift', 'head_pan', 'head_tilt', 'base_dx', 'base_dy', 'base_dyaw',
                     'mode_front', 'mode_left', 'mode_right']
ACTION_NAMES = ARM + ['gripper_width', 'lift', 'base_forward']

# kcare_robot/configs/tasks.py
GIVE = {'front': [90, 0, 20, 12, -40, 97, -10], 'left': [96.85, 18.01, -49.85, 14.47, -33.29, 53.71, -14.79],
        'right': [-78.91, 24.26, 45.87, 24.81, -318.36, 52.09, 20.36]}
APPROACH_STANDING = {'right': [36.43, 33.89, -31.2, 51.74, -295.65, 99.97, 13.2],
                     'left': [-39.72, 30.57, 36.3, 45.17, -58.42, 99.99, -5.61],
                     'front': [0.0, 5.83, 0.0, 5.97, -180.0, 75.41, 0.0]}
# worker B's front poses (sim ARM_CONFIGS): the arm out of the head camera's view
PRE_PICK_FLOOR = {'front': [0.0, 64.64, 0.0, 79.37, -180.0, -14.72, 0.0],
                  'left': [-33.54, 106.3, -70.37, 91.75, -253.32, 70.67, -37.49],
                  'right': [33.54, 106.3, 70.37, 91.75, -106.68, 70.67, 37.49]}
PRE_PICK_HIGH = {'front': [0.0, -13.39, 0.0, 6.35, -180.0, 70.27, 0.0],
                 'left': [-44.63, 10.06, -40.33, 31.39, -174.03, 65.8, 4.06],
                 'right': [45.28, 9.96, 39.7, 31.39, -185.93, 65.81, -3.93]}
POSES = dict(give=GIVE, approach_standing=APPROACH_STANDING, pre_pick_floor=PRE_PICK_FLOOR, pre_pick_high=PRE_PICK_HIGH)
HEAD_RZ = {'front': 0, 'left': -70, 'right': 70}
LIFT = dict(range=(0.56, 1.1), mobile_height=0.43, home={'front': 0.6, 'left': 0.8, 'right': 0.8})
CAM_LINK = 'camera_color_optical_frame'
EXPERT = dict(head_tilt=-50, gripper=0.1, base_max=0.2, via_up=0.12, via_back=0.05, arm_speed=0.8, lift_speed=0.08,
              base_speed=0.15, hold=0.3, descend=1.3, miss=1.3, pause=0.4, correct=1.3, settle=0.6, ik_tol=0.004)


def lift_joint(x):
    """kcare lift::x -> lift_joint [m]."""
    return float(np.clip(x, *LIFT['range']) - LIFT['mobile_height'])


def minjerk(s):
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class CamIK:
    """7-DoF IK for the wrist camera frame (position + optical z + y axes) with lift / base fixed."""

    def __init__(self, ref, q_fixed):
        self.ref, self.q = ref, dict(q_fixed)
        self.lo = np.array([max(ref.joints[j]['limit'][0], -2 * np.pi) for j in ARM])
        self.hi = np.array([min(ref.joints[j]['limit'][1], 2 * np.pi) for j in ARM])

    def T(self, x, link=CAM_LINK):
        return self.ref.fk(dict(self.q, **dict(zip(ARM, x))))[link]

    def solve(self, R, p, seeds):
        best = None
        for x0 in seeds:
            def res(x):
                T = self.T(x)
                return np.concatenate([(T[:3, 3] - p) * 20, (T[:3, 2] - R[:, 2]) * 2, (T[:3, 1] - R[:, 1]) * 2,
                                       (x - x0) * 0.005])
            sol = least_squares(res, np.clip(x0, self.lo + 1e-4, self.hi - 1e-4), bounds=(self.lo, self.hi),
                                xtol=1e-9, ftol=1e-9, max_nfev=400)
            T = self.T(sol.x)
            err = np.linalg.norm(T[:3, 3] - p) + 0.05 * (np.linalg.norm(T[:3, 2] - R[:, 2]) + np.linalg.norm(T[:3, 1] - R[:, 1]))
            if best is None or err < best[1]:
                best = (sol.x, err)
            if err < EXPERT['ik_tol']:
                break
        return best


def cam_tcp_offset(ref, q):
    """Distance from the wrist camera to the finger tips along the optical axis (the D405 sits behind the fingers)."""
    T = ref.fk(q)
    return float(np.dot(T['tool_tcp_link'][:3, 3] - T[CAM_LINK][:3, 3], T[CAM_LINK][:3, 2]))


def view_frame(c, view_dir, view, cam_off=0.0):
    """Camera rotation (optical: x right, y down, z forward) + position looking at c along the robot's working
    direction view_dir (base frame: front (1,0,0), left (0,-1,0), right (0,1,0)), from above (view['elev'], 0 =
    horizontal for a side grasp on a high shelf). view['dist'] is finger tips -> object centre; the camera sits
    cam_off further back on the same axis."""
    h = np.array(view_dir, float)
    cy, sy = math.cos(view['yaw']), math.sin(view['yaw'])
    h = np.array([cy * h[0] - sy * h[1], sy * h[0] + cy * h[1], 0.0])
    z = math.cos(view['elev']) * h + math.sin(view['elev']) * np.array([0, 0, -1.0])
    z /= np.linalg.norm(z)
    g = np.array([0, 0, -1.0])
    y = g - np.dot(g, z) * z
    y /= np.linalg.norm(y)
    R = np.stack([np.cross(y, z), y, z], 1)
    return R, c - z * (view['dist'] + cam_off), h


def obj_centre(m, d, geom):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
    return d.geom_xpos[gid] + d.geom_xmat[gid].reshape(3, 3) @ m.geom_aabb[gid][:3]


def robot_bodies(m):
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_footprint')
    out = set()
    for b in range(m.nbody):
        p = b
        while p > 0 and p != root:
            p = m.body_parentid[p]
        if p == root:
            out.add(b)
    return out


def bad_contacts(m, d, robot, env_geoms):
    hits = []
    for c in d.contact[:d.ncon]:
        g1, g2 = c.geom1, c.geom2
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        if (b1 in robot and g2 in env_geoms) or (b2 in robot and g1 in env_geoms):
            hits.append((m.geom(g1).name or m.body(b1).name, m.geom(g2).name or m.body(b2).name))
    return hits


def self_contacts(m, d, robot):
    """Robot-robot contacts (arm vs base / lift column / head ...); pairs excluded in the model never show up."""
    out = []
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in robot and b2 in robot and b1 != b2:
            out.append(tuple(sorted((m.geom(c.geom1).name or m.body(b1).name, m.geom(c.geom2).name or m.body(b2).name))))
    return out


# clearance (owner, 2026-10-09): the arm keeps >= 0.05 m from the head / head camera / neck / lift column and >= 0.03 m
# from the base and the furniture at every sampled point (contacts alone let it graze them). A pair already closer at
# kcare's own start pose may not get closer than it started.
ARM_BODIES = ('link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand_base_link', 'gripper_body_link',
              'right_finger_tip', 'left_finger_tip', 'vision_frame_link', 'camera_link', 'camera_usb_plug_link')
HEAD_BODIES = ('head_base', 'head_link1', 'head_link2', 'head_link3', 'femto_base_link', 'lift_link', 'lift_base_link')
BASE_BODIES = ('slamtec_base_link',)
MARGIN = dict(head=0.05, base=0.03, furniture=0.03)


def clearance_sets(m, d, robot, around, radius=1.3):
    """Geom ids: arm, head + column, base, furniture / walls within `radius` m of the base (xy) — objects excluded."""
    def geoms(bodies):
        ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids and (m.geom_contype[g] or m.geom_conaffinity[g])]
    furn = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b in robot or m.geom(g).name == 'floor' or m.body(b).name.startswith('obj_'):
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if np.linalg.norm(d.geom_xpos[g][:2] - np.array(around[:2])) - m.geom_rbound[g] < radius:
            furn.append(g)
    return dict(arm=geoms(ARM_BODIES), head=geoms(HEAD_BODIES), base=geoms(BASE_BODIES), furniture=furn)


def clearance(m, d, sets, distmax=0.12):
    """Smallest distance [m] from the arm to each category (capped at distmax)."""
    out, ft = {}, np.zeros(6)
    for cat in ('head', 'base', 'furniture'):
        best = distmax
        for ga in sets['arm']:
            for gb in sets[cat]:
                best = min(best, mujoco.mj_geomDistance(m, d, ga, gb, distmax, ft))
        out[cat] = float(best)
    return out


def clear_ok(cl, start):
    """Violated categories: closer than the margin AND closer than at the start pose."""
    return [c for c, v in cl.items() if v < MARGIN[c] and v < start[c] - 0.01]


def head_sees(m, d, geom, W=320, H=180):
    """How many of 9 points of `geom` (box centre + corners pulled halfway in) the head camera sees: in the image and
    not behind anything nearer in its depth render (the arm in front of the target -> hidden). (MuJoCo's
    segmentation render indexes past ngeom in this house model, so depth is used instead.)"""
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'head_rgb')
    r = mujoco.Renderer(m, H, W)
    try:
        r.enable_depth_rendering()
        r.update_scene(d, 'head_rgb')
        depth = r.render()
    finally:
        r.close()
    Rg, pg = d.geom_xmat[gid].reshape(3, 3), d.geom_xpos[gid]
    c, h = m.geom_aabb[gid][:3], m.geom_aabb[gid][3:]
    pts = [c] + [c + 0.5 * h * np.array(sg) for sg in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]]
    pos, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
    f = 0.5 * H / math.tan(math.radians(m.cam_fovy[cid]) / 2)
    n = 0
    for p in pts:
        q = R.T @ (pg + Rg @ p - pos)
        z = -q[2]
        if z <= 0.05:
            continue
        u, v = int(W / 2 + f * q[0] / z), int(H / 2 - f * q[1] / z)
        if 0 <= u < W and 0 <= v < H and depth[v, u] > z - 0.03:      # planar depth of the nearest surface
            n += 1
    return n


def wrist_view(m, d, c, W=848, H=480):
    """(u, v, distance) of world point c in the wrist camera image."""
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_rgb')
    pos, R = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3)
    p = R.T @ (c - pos)
    depth = -p[2]
    f = 0.5 * H / math.tan(math.radians(m.cam_fovy[cid]) / 2)
    if depth <= 1e-4:
        return None, None, float(np.linalg.norm(c - pos))
    return W / 2 + f * p[0] / depth, H / 2 - f * p[1] / depth, float(np.linalg.norm(c - pos))


def to_base(p, robot):
    """World point -> the start base frame (x heading, y left, z up)."""
    x, y, yaw = robot
    c, s_ = math.cos(yaw), math.sin(yaw)
    dx, dy = p[0] - x, p[1] - y
    return np.array([c * dx + s_ * dy, -s_ * dx + c * dy, p[2]])


PLAN_STATS = __import__('collections').Counter()   # why candidates were rejected (debugging)


def plan(m, d, ref, q0, rand, info, scratch, robot, env_geoms, log=print, ignore=(), self_ok=(), csets=None, cstart=None):
    """Pick lift / base shift / arm goal + via + (retry) miss, all in the start base frame. Returns dict or None."""
    mode = rand['mode']
    tgt = info['objects'][info['target']]
    c_w = obj_centre(m, d, tgt['body'])
    rb = info.get('robot', (0.0, 0.0, 0.0))
    c = to_base(c_w, rb)
    R, cam_p, h = view_frame(c, info['view_dir'], rand['view'], cam_tcp_offset(ref, q0))
    base_lift = info['top_z'] + 0.1                        # kcare's rule, then the band's own (reach sweep) lifts
    lifts = [base_lift, base_lift + 0.1, base_lift - 0.1, base_lift + 0.2] + list(rand.get('lifts') or [])
    if rand.get('lifts_only'):                             # e.g. near the floor: the lift stays at its lowest
        lifts = list(rand.get('lifts') or [])
    lifts = list(dict.fromkeys(round(lift_joint(x), 3) for x in lifts))
    # base shift along the heading: the smallest of 0 / 0.1 / 0.2 m that brings the target into the arm's comfortable
    # band (x in [-0.15, 0.8] of the base); reaching far behind makes the arm run into the robot's own column
    lo_c, hi_c = rand.get('comfort', (-0.15, 0.8))
    sgn = 1.0 if c[0] > (lo_c + hi_c) / 2 else -1.0
    need = max(0.0, lo_c - c[0], c[0] - hi_c)
    shifts = [b for b in (0.0, 0.1, 0.2) if b >= need - 1e-9 or b == 0.2]
    shifts = [sgn * b for b in shifts]
    seeds0 = [np.array([q0[j] for j in ARM])]
    for pz in (APPROACH_STANDING, PRE_PICK_HIGH, PRE_PICK_FLOOR):
        if mode in pz:
            seeds0.append(np.radians(pz[mode]))
    for bx in shifts:
        tried = []
        # every (lift, IK seed): different seeds land on different arm branches, and the first branch that solves
        # may still hit the robot (near the floor link5 folds onto the base) while another one is clear
        for lj, seed in [(l_, sd) for l_ in lifts for sd in seeds0]:
            q = dict(q0, lift_joint=lj)
            ik = CamIK(ref, q)
            shift = np.array([bx, 0.0, 0.0])
            sol = ik.solve(R, cam_p - shift, [seed])
            if sol is None or sol[1] > EXPERT['ik_tol']:
                PLAN_STATS['ik_goal'] += 1; continue
            goal = sol[0]
            if any(lj == t_[0] and np.max(np.abs(goal - t_[1])) < 0.05 for t_ in tried):
                continue                                    # the same solution as a rejected one
            tried.append((lj, goal))
            vu, vb_ = rand.get('via', (EXPERT['via_up'], EXPERT['via_back']))
            via_p = cam_p + np.array([0, 0, vu]) - h * vb_
            via = ik.solve(R, via_p - shift, [goal] + seeds0)
            if via is None or via[1] > 0.01:
                PLAN_STATS['ik_via'] += 1; continue
            # goal pose must not touch the table / objects
            cs.set_q(m, scratch, dict(q, **dict(zip(ARM, goal))), {'gripper': cs_cfg_gripper()})
            ax, ay = base_adr(m)
            scratch.qpos[ax] = rb[0] + bx * math.cos(rb[2])
            scratch.qpos[ay] = rb[1] + bx * math.sin(rb[2])
            mujoco.mj_forward(m, scratch)
            bad_ = [h_ for h_ in bad_contacts(m, scratch, robot, env_geoms) if h_ not in ignore] + \
                   [h_ for h_ in self_contacts(m, scratch, robot) if h_ not in self_ok]
            if bad_:
                PLAN_STATS['contact_goal'] += 1; PLAN_STATS[str(bad_[0])] += 1; continue
            if csets is not None and clear_ok(clearance(m, scratch, csets), cstart):
                PLAN_STATS['clear_goal'] += 1; continue  # the goal pose itself is too close
            if csets is not None:                           # the whole joint-space path, kinematically, before moving
                x0_ = np.array([q0[j] for j in ARM])
                legs = [(x0_, via[0], q0['lift_joint'], lj, 0.0, bx), (via[0], goal, lj, lj, bx, bx)]
                if path_bad(m, scratch, q, legs, rb, csets, cstart, robot, env_geoms, ignore, self_ok):
                    PLAN_STATS['path'] += 1; continue
            misses = []
            if rand.get('retry'):
                perp = np.array([-h[1], h[0], 0.0])             # sideways in the image
                offs = rand.get('retry_offsets') or [list(rand['retry_offset']) + [0.0]]
                prev = via[0]
                for o in offs:                                  # [sideways, vertical, along the view axis (+ = closer)]
                    off = perp * o[0] + np.array([0, 0, o[1] * 0.5]) + R[:, 2] * (o[2] if len(o) > 2 else 0.0)
                    ms = ik.solve(R, cam_p + off - shift, [goal])
                    if ms is None or ms[1] > 0.01 or (csets is not None and path_bad(
                            m, scratch, q, [(prev, ms[0], lj, lj, bx, bx), (ms[0], goal, lj, lj, bx, bx)],
                            rb, csets, cstart, robot, env_geoms, ignore, self_ok)):
                        break
                    misses.append(ms[0]); prev = ms[0]
            return dict(lift=lj, base=bx, goal=goal, via=via[0], miss=misses[0] if misses else None, misses=misses,
                        c=c_w, R=R, cam_p=cam_p)
    return None


def path_bad(m, scratch, q, legs, rb, csets, cstart, robot, env_geoms, ignore=(), self_ok=(), n=8):
    """Any contact / clearance violation along joint-space legs [(x_a, x_b, lift_a, lift_b, base_a, base_b)]."""
    ax, ay = base_adr(m)
    for xa, xb, la, lb, ba, bb in legs:
        for i in range(1, n + 1):
            s_ = minjerk(i / n)
            cs.set_q(m, scratch, dict(q, **dict(zip(ARM, xa + (xb - xa) * s_)), lift_joint=la + (lb - la) * s_),
                     {'gripper': cs_cfg_gripper()})
            bs = ba + (bb - ba) * s_
            scratch.qpos[ax], scratch.qpos[ay] = rb[0] + bs * math.cos(rb[2]), rb[1] + bs * math.sin(rb[2])
            mujoco.mj_forward(m, scratch)
            if [h for h in bad_contacts(m, scratch, robot, env_geoms) if h not in ignore] or \
                    [h for h in self_contacts(m, scratch, robot) if h not in self_ok] or \
                    clear_ok(clearance(m, scratch, csets), cstart):
                return True
    return False


_CFG_GRIPPER = {}


def cs_cfg_gripper():
    return _CFG_GRIPPER


def base_adr(m):
    return tuple(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ('base_x', 'base_y'))


def setup(m, d, task, cfg, rand, info):
    """Put the robot at the mode's start pose, let the objects settle, and run the start checks. Returns
    (ctx, '') or (None, reason). ctx holds everything stepping / scoring needs (the expert and eval_pick_policy.py)."""
    _CFG_GRIPPER.update(cfg['gripper'])
    mode = rand['mode']
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q0 = dict(cfg['home'], lift_joint=lift_joint(LIFT['home'][mode]),
              head_joint1=math.radians(rand.get('head_pan', HEAD_RZ[mode])) + task.get('head_pan_offset', 0.0),
              head_joint2=math.radians(rand.get('head_tilt', task.get('head_tilt', EXPERT['head_tilt']))),
              virtual_gripper_joint=EXPERT['gripper'])
    sp = rand.get('start_pose', 'give')                     # 'pose' (this mode's) or 'pose:mode' (another mode's arm)
    start = POSES[sp.split(':')[0]][sp.split(':')[1] if ':' in sp else mode]
    q0.update({f'joint{i + 1}': math.radians(a) for i, a in enumerate(start)})
    for j, dv in (rand.get('q_extra') or {}).items():
        q0[j] += dv
    hu = rand.get('human') or {}
    if hu.get('start_deg'):                                 # the arm is not exactly at the ready pose
        srng = np.random.default_rng(hu['start_seed'])
        for j in ARM:
            q0[j] += math.radians(srng.uniform(-1, 1) * hu['start_deg'])
    n_sub = max(1, round(1.0 / HZ / m.opt.timestep))
    m.opt.timestep = 1.0 / HZ / n_sub
    cs.set_q(m, d, q0, cfg)
    d.ctrl[:] = cs.ctrl_from(m, q0, cfg)
    ja = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ('base_x', 'base_y', 'base_yaw')}
    jv = {j: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ja}
    # let the objects settle: until every object is still (balls roll for a while), at most 3 s; the target's start
    # pose is measured after this, so a ball that rolled while settling is not counted as moved by the robot
    obj_dofs = [m.jnt_dofadr[j] for j in range(m.njnt) if m.joint(j).name.startswith('obj_')]
    for _k in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if _k * m.opt.timestep > 0.3 and _k % 25 == 0 and \
                all(np.linalg.norm(d.qvel[a:a + 6]) < 0.01 for a in obj_dofs):
            break
    robot = robot_bodies(m)
    # everything that is not the robot or the floor: furniture, walls, objects
    env_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] not in robot and m.geom(g).name != 'floor'}
    start_hits = set(bad_contacts(m, d, robot, env_geoms))
    if start_hits:                                          # the jittered stop overlaps furniture: not an episode
        return None, f'robot starts in contact {sorted(start_hits)[:2]}'
    tgt_body = info['objects'][info['target']]['body']
    if obj_centre(m, d, tgt_body)[2] < info['top_z'] - 0.02:
        return None, 'target fell off the surface while settling'
    start_self = set(self_contacts(m, d, robot))
    seen = head_sees(m, d, tgt_body)                      # the head camera must see the target at the start pose
    if seen < task.get('min_head_points', 4):
        return None, f'target hidden in the head image at the start ({seen}/9 points visible)'
    csets = clearance_sets(m, d, robot, info.get('robot', (0, 0, 0)))
    cstart = clearance(m, d, csets)
    base0 = np.array([d.qpos[ja['base_x']], d.qpos[ja['base_y']], d.qpos[ja['base_yaw']]])
    return dict(mode=mode, ref=ref, q0=q0, n_sub=n_sub, ja=ja, jv=jv, robot=robot, env_geoms=env_geoms,
                start_hits=start_hits, start_self=start_self, csets=csets, cstart=cstart, tgt_body=tgt_body,
                head_px=seen, base0=base0, hx=math.cos(base0[2]), hy=math.sin(base0[2]),
                onehot=[1.0 if mode == k else 0.0 for k in MODES], c_start=obj_centre(m, d, tgt_body).copy(),
                hits=[], min_cl=dict(cstart), k=0), ''


def base_delta(d, ctx):
    """Base motion since the start, in the start base frame: [forward, left, yaw]."""
    ja, b0 = ctx['ja'], ctx['base0']
    dx, dy = d.qpos[ja['base_x']] - b0[0], d.qpos[ja['base_y']] - b0[1]
    return [ctx['hx'] * dx + ctx['hy'] * dy, -ctx['hy'] * dx + ctx['hx'] * dy, d.qpos[ja['base_yaw']] - b0[2]]


def state_vec(m, d, ctx):
    """The 17-D observation.state."""
    return ([cs.get_q(m, d, j) for j in ARM] + [cs.get_q(m, d, 'virtual_gripper_joint'), cs.get_q(m, d, 'lift_joint'),
            cs.get_q(m, d, 'head_joint1'), cs.get_q(m, d, 'head_joint2')] + base_delta(d, ctx) + ctx['onehot'])


def apply_step(m, d, ctx, cfg, x, grip, lift, base):
    """One 30 Hz control step: arm / gripper / lift position targets, the base driven kinematically along its heading
    to `base` [m since the start], then the contact + clearance checks (clearance every 3rd step = 10 Hz)."""
    q_cmd = dict(ctx['q0'], **dict(zip(ARM, x)), lift_joint=lift, virtual_gripper_joint=grip)
    d.ctrl[:] = cs.ctrl_from(m, q_cmd, cfg)
    ja, jv, b0, hx, hy, n_sub = ctx['ja'], ctx['jv'], ctx['base0'], ctx['hx'], ctx['hy'], ctx['n_sub']
    b_prev = base_delta(d, ctx)[0]
    vb = (base - b_prev) * HZ
    for i in range(n_sub):
        f = b_prev + (base - b_prev) * (i + 1) / n_sub
        d.qpos[ja['base_x']], d.qpos[ja['base_y']] = b0[0] + f * hx, b0[1] + f * hy
        d.qvel[jv['base_x']], d.qvel[jv['base_y']] = vb * hx, vb * hy
        mujoco.mj_step(m, d)
    ctx['hits'] += [h_ for h_ in bad_contacts(m, d, ctx['robot'], ctx['env_geoms']) if h_ not in ctx['start_hits']]
    ctx['hits'] += [('self',) + h_ for h_ in self_contacts(m, d, ctx['robot']) if h_ not in ctx['start_self']]
    if ctx['k'] % 3 == 0:
        cl = clearance(m, d, ctx['csets'])
        for cat in clear_ok(cl, ctx['cstart']):
            ctx['hits'].append(('clearance', cat, round(cl[cat], 3)))
        for cat, v in cl.items():
            ctx['min_cl'][cat] = min(ctx['min_cl'][cat], v)
    ctx['k'] += 1


def score(m, d, ctx, last_action):
    """The success test: target centred in the wrist view, finger tips 0.08-0.18 m away, no contact / clearance
    violation, the target not moved (> 1 cm), the arm where it was commanded (< 0.05 rad)."""
    for _k in range(int(0.2 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    c_end = obj_centre(m, d, ctx['tgt_body'])
    u, v, dist = wrist_view(m, d, c_end)
    centred = u is not None and abs(u - 424) < 0.15 * 848 and abs(v - 240) < 0.15 * 480
    moved = float(np.linalg.norm(c_end - ctx['c_start']))
    tip = dist - cam_tcp_offset(ctx['ref'], ctx['q0'])
    track = float(np.max(np.abs(np.array(last_action[:7]) - np.array([cs.get_q(m, d, j) for j in ARM]))))
    hits = ctx['hits']
    ok = bool(centred and 0.08 <= tip <= 0.18 and not hits and moved < 0.01 and track < 0.05)
    reason = '' if ok else (f'collision / clearance {sorted(set(map(str, hits)))[:2]}' if hits else
                            f'arm blocked (tracking {track:.2f} rad)' if track >= 0.05 else
                            f'moved {moved:.3f}' if moved >= 0.01 else f'view u={u} v={v} tip={tip:.3f}')
    return dict(success=ok, reason=reason, metric=dist, u=u, v=v, moved=moved, tip_dist=tip, track_err=track,
                clearance={k: round(v_, 3) for k, v_ in ctx['min_cl'].items()},
                collisions=sorted(set(map(str, hits)))[:5], head_px=ctx['head_px'])


def run_episode(m, d, task, cfg, rand=None, info=None, render=None, log=print, **_):
    rec = dict(action=[], state=[], phase=[], success=False, metric=float('nan'), reason='')
    ctx, why = setup(m, d, task, cfg, rand, info)
    if ctx is None:
        rec['reason'] = why
        return rec
    rec['head_px'] = ctx['head_px']
    q0 = ctx['q0']
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos
    p = plan(m, d, ctx['ref'], q0, rand, info, scratch, ctx['robot'], ctx['env_geoms'], log, self_ok=ctx['start_self'],
             csets=ctx['csets'], cstart=ctx['cstart'])
    if p is None:
        rec['reason'] = 'no reachable collision-free viewpoint'
        return rec
    ctx['c_start'] = p['c'].copy()
    hu = rand.get('human') or {}
    sp = float(hu.get('speed', 1.0))                        # > 1 = faster
    x_start = np.array([q0[j] for j in ARM])
    phases = [('hold', EXPERT['hold'], x_start, x_start, q0['lift_joint'], q0['lift_joint'], 0.0, 0.0)]
    t_reach = max(2.0 / sp, np.max(np.abs(p['via'] - x_start)) / (EXPERT['arm_speed'] * sp),
                  abs(p['lift'] - q0['lift_joint']) / EXPERT['lift_speed'], abs(p['base']) / EXPERT['base_speed'])
    phases.append(('reach', t_reach, x_start, p['via'], q0['lift_joint'], p['lift'], 0.0, p['base']))
    last = p['via']
    L, B = p['lift'], p['base']
    if hu.get('hesitate'):                                  # a human stops for a moment before going in
        phases.append(('hesitate', hu['hesitate'], last, last, L, L, B, B))
    for ms in p.get('misses') or ([p['miss']] if p['miss'] is not None else []):
        phases += [('miss', EXPERT['miss'] / sp, last, ms, L, L, B, B), ('pause', EXPERT['pause'], ms, ms, L, L, B, B)]
        last = ms
    if last is not p['via']:
        phases.append(('correct', EXPERT['correct'] / sp, last, p['goal'], L, L, B, B))
    else:
        phases.append(('descend', EXPERT['descend'] / sp, last, p['goal'], L, L, B, B))
    phases.append(('settle', float(hu.get('settle', EXPERT['settle'])), p['goal'], p['goal'], L, L, B, B))
    # noise on what is EXECUTED (smooth, Ornstein-Uhlenbeck), faded out over the final approach; the recorded action
    # stays the clean expert target, so the data shows how to get back on track from where the noise left the robot
    nz = hu.get('noise')
    n_total = sum(max(1, int(round(T * HZ))) for _n, T, *_r in phases)
    n_settle = max(1, int(round(phases[-1][1] * HZ))); n_final = max(1, int(round(phases[-2][1] * HZ)))
    noise = np.zeros((n_total, 9))
    if nz:
        nrng = np.random.default_rng(nz['seed'])
        sig = np.array([math.radians(nz['arm_deg'])] * 7 + [nz['lift'], nz['base']])
        a_ = 1.0 / (HZ * nz['tau'])
        for k in range(1, n_total):
            noise[k] = noise[k - 1] * (1 - a_) + sig * math.sqrt(2 * a_) * nrng.standard_normal(9)
        env = np.ones(n_total)
        f0 = n_total - n_settle - n_final                       # fade over the last approach phase, 0 while settling
        env[f0:n_total - n_settle] = np.linspace(1, 0, n_final)
        env[n_total - n_settle:] = 0.0
        h0 = int(EXPERT['hold'] * HZ); n_in = int(1.2 * HZ)     # none while holding, then ramps in: the ready pose
        env[:h0] = 0.0                                           # is close to the head (head clearance 5 cm)
        env[h0:h0 + n_in] *= np.linspace(0, 1, len(env[h0:h0 + n_in]))
        noise *= env[:, None]
    kk = 0
    for name, T, xa, xb, la, lb, ba, bb in phases:
        steps = max(1, int(round(T * HZ)))
        for k in range(steps):
            s = minjerk((k + 1) / steps)
            x = xa + (xb - xa) * s
            lift = la + (lb - la) * s
            base = ba + (bb - ba) * s
            state = state_vec(m, d, ctx)
            action = list(x) + [EXPERT['gripper'], lift, base]
            rec['state'].append(state); rec['action'].append(action); rec['phase'].append(name)
            if render:
                render(m, d, name, float('nan'), state=state, action=action)
            nk = noise[kk]; kk += 1
            apply_step(m, d, ctx, cfg, x + nk[:7], EXPERT['gripper'], float(np.clip(lift + nk[7], 0.13, 0.67)),
                       float(np.clip(base + nk[8], -EXPERT['base_max'], EXPERT['base_max'])))
    rec.update(score(m, d, ctx, rec['action'][-1]))
    rec['plan'] = dict(lift=p['lift'], base=p['base'], retry=p['miss'] is not None, misses=len(p.get('misses') or []),
                       noise=bool(nz), speed=round(sp, 2), hesitate=round(float(hu.get('hesitate', 0.0)), 2))
    rec = {k: (np.array(v_) if k in ('action', 'state') else v_) for k, v_ in rec.items()}
    return rec


# ── generator hooks (gen_dataset.py) ─────────────────────────────────────────
def sample_rand(rng, rcfg, task):
    return pick_scene.sample(rng, rcfg, task)


def build_scene(task, rand):
    """Marks this expert as one with its own scene for gen_dataset; the model itself comes from episode_model."""
    return None, None


def episode_model(task, rand, wid=0):
    return pick_scene.episode_model(task, rand, wid)


def main():
    import imageio, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--task', default=f'{SIM}/config/task_pick_approach.yaml')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(a.task))
    rcfg = yaml.safe_load(open(task.get('rand_config', f'{SIM}/config/rand_pick.yaml')))
    out = f'{SIM}/out/pick_approach'
    os.makedirs(out, exist_ok=True)
    ok = 0
    for e in range(a.n):
        rng = np.random.default_rng(a.seed * 1000 + e)
        rand = sample_rand(rng, rcfg, task)
        m, d, info = episode_model(task, rand)
        rw = mujoco.Renderer(m, 240, 424)
        rh = mujoco.Renderer(m, 240, 424)
        frames = []

        def on_step(m, d, phase, metric, state=None, action=None):
            if len(frames) % 2:
                frames.append(None); return
            rh.update_scene(d, 'head_rgb'); rw.update_scene(d, 'wrist_rgb')
            frames.append(np.concatenate([rh.render(), rw.render()], 1))
        rec = run_episode(m, d, task, cfg, rand=rand, info=info, render=on_step)
        ok += rec['success']
        print(f"ep {e}: {rand['band']} {rand['spot']} mode {rand['mode']} band {rand['base_band']} target {rand['target']} -> {'OK' if rec['success'] else 'FAIL'} "
              f"{rec.get('reason')} frames {len(rec['action'])} plan {rec.get('plan')}", flush=True)
        fr = [f for f in frames if f is not None]
        if fr:
            imageio.mimwrite(f'{out}/ep{e}.mp4', fr, fps=15, quality=6, macro_block_size=1)
        rw.close(); rh.close()
    print(f'{ok}/{a.n} OK')


if __name__ == '__main__':
    main()
