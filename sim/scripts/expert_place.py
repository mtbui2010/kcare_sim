#!/usr/bin/env python3
"""Scripted expert for PLACE ("carry -> place", VLA data step 6), in the house episodes of pick_scene.

The robot starts as at the end of a pick: kcare's carry pose (movej::give of the mode) with the object held upright
between the fingers (a weld to the gripper, house_bridge's grasp-assist idea; the fingers closed on it). The place
point comes from the real skill's find_place and is given as 3 extra state dims (x, y, z in the start base frame).
The expert moves arm + lift + base (forward <= 0.2 m) to above the point, lowers until the object is ~1 cm above the
surface, opens the gripper (weld off), and lifts the hand 0.1 m.

    /sim/.venv/bin/python scripts/expert_place.py [--n 6] [--seed 1]      # episodes -> out/place/*.mp4

Planning (before moving): TCP IK (the start pose's tool orientation, turned about the vertical by 0 / +-20 / +-40 deg
so the object stays upright) for every (base shift, lift, IK seed); the transfer / lower / lift legs are checked
kinematically with the object carried along: robot contacts, object contacts (furniture, other objects), self contacts
and the clearance margins (head / column 0.05 m, base / furniture 0.03 m; the support surface itself is exempt from
the furniture margin, contacts with it still count while the object is held).
Success: the object rests within 5 cm (xy) of the point, upright (tilt < 20 deg), on the surface, the gripper off it,
no contact / clearance violation during the episode, arm tracking < 0.05 rad.
"""
import argparse, math, os, re, sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import check_sim as cs
import expert_pick_approach as pk
import pick_scene

SIM = os.environ.get('SIM_ROOT', '/sim')
HZ = pk.HZ
ARM = pk.ARM
STATE_NAMES = pk.STATE_NAMES + ['place_x', 'place_y', 'place_z']
ACTION_NAMES = pk.ACTION_NAMES
PLAN_STATS = __import__('collections').Counter()   # why plan candidates were rejected (debugging)
EXPERT = dict(above=0.10, clear=0.01, lift_after=0.10, hold=0.3, lower=1.3, release=0.8, up=1.0, settle=0.6,
              arm_speed=0.8, lift_speed=0.08, base_speed=0.15, ik_tol=0.004, yaws_deg=(0, 20, -20, 40, -40),
              max_width=0.085, h_range=(0.05, 0.25))


class TcpIK(pk.CamIK):
    def T(self, x, link='tool_tcp_link'):
        return self.ref.fk(dict(self.q, **dict(zip(ARM, x))))[link]


def _held_ok(oid):
    lo, hi = pick_scene.mesh_box(oid)
    ext = hi - lo
    return min(ext[0], ext[1]) <= EXPERT['max_width'] and max(ext[0], ext[1]) <= 0.12 and EXPERT['h_range'][0] <= ext[2] <= EXPERT['h_range'][1]


def sample_rand(rng, rcfg, task):
    """pick_scene's episode; objects[0] (put at the reachable point) becomes the held object, its spot the place point."""
    for _ in range(40):
        rand = pick_scene.sample(rng, rcfg, task)
        o = rand['objects'][0]
        if _held_ok(o['id']):
            break
    o['lying'] = False                                      # carried upright
    # the hand needs room around the point: no other object within 0.15 m of it (they were only kept off its footprint)
    rand['objects'] = [o] + [q for q in rand['objects'][1:] if math.hypot(q['x'] - o['x'], q['y'] - o['y']) > 0.15]
    surf = rand['spot'].split('@')[0]
    phr = rcfg.get('place_prompts', ['place the {obj} on the {surf}'])
    rand.update(kind='place', place=[o['x'], o['y'], rand['top_z']], held=o['name'],
                prompt=phr[int(rng.integers(len(phr)))].format(obj=o['name'], surf=surf))
    return rand


def build_scene(task, rand):
    return None, None


def episode_model(task, rand, wid=0):
    return pick_scene.episode_model(task, rand, wid, welds=True)


def _set_obj(m, d, body, T):
    """Free-joint object `body` to the 4x4 world pose T."""
    jid = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)]
    a, v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
    d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = T[:3, 3], q
    d.qvel[v:v + 6] = 0.0


def _body_T(m, d, body):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
    return T


def _grip_point(m, d):
    """Between the finger tips: where the held object's centre sits (the TCP is ~3 cm beyond the tips)."""
    return (_body_T(m, d, 'left_finger_tip')[:3, 3] + _body_T(m, d, 'right_finger_tip')[:3, 3]) / 2


def _contacts_of(m, d, geom, exclude):
    """Geoms touching `geom`, minus `exclude` (ids)."""
    out = []
    for c in d.contact[:d.ncon]:
        if c.geom1 == geom and c.geom2 not in exclude:
            out.append(c.geom2)
        elif c.geom2 == geom and c.geom1 not in exclude:
            out.append(c.geom1)
    return out


def setup(m, d, task, cfg, rand, info):
    """Carry pose with the object welded upright between the fingers, the rest settled; start checks. (ctx, '')."""
    pk._CFG_GRIPPER.update(cfg['gripper'])
    mode = rand['mode']
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    o = rand['objects'][0]
    lo, hi = pick_scene.mesh_box(o['id'])
    ext = hi - lo
    narrow = int(ext[1] < ext[0])                           # the object's narrow horizontal axis (mesh x or y)
    width = float(np.clip(ext[narrow] + 0.002, 0.01, 0.1))  # the fingers just around it (the weld holds it)
    q0 = dict(cfg['home'], lift_joint=pk.lift_joint(pk.LIFT['home'][mode]),
              head_joint1=math.radians(pk.HEAD_RZ[mode]) + task.get('head_pan_offset', 0.0),
              head_joint2=math.radians(rand.get('head_tilt', -50)), virtual_gripper_joint=width)
    q0.update({f'joint{i + 1}': math.radians(a) for i, a in enumerate(pk.GIVE[mode])})
    for j, dv in (rand.get('q_extra') or {}).items():
        q0[j] += dv
    n_sub = max(1, round(1.0 / HZ / m.opt.timestep))
    m.opt.timestep = 1.0 / HZ / n_sub
    cs.set_q(m, d, q0, cfg)
    d.ctrl[:] = cs.ctrl_from(m, q0, cfg)
    mujoco.mj_forward(m, d)
    held_body = info['objects'][o['name']]['body']
    held_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, held_body)
    # object upright, its box centre between the finger tips, its narrow axis along the finger axis
    tcp = _body_T(m, d, 'tool_tcp_link')
    grip = _grip_point(m, d)
    fa = _body_T(m, d, 'left_finger_tip')[:3, 3] - _body_T(m, d, 'right_finger_tip')[:3, 3]
    yaw = math.atan2(fa[1], fa[0]) - (math.pi / 2 if narrow else 0.0)
    c_local = (lo + hi) / 2
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    T_obj = np.eye(4)
    T_obj[:3, :3] = Rz
    # held between the tips, but low enough that it does not reach into the palm: a tall object is held near its top
    # (in the palm, the released object was thrown over); lowered in 5 mm steps until it touches nothing of the robot
    rgeoms = {gg for gg in range(m.ngeom) if m.geom_bodyid[gg] in pk.robot_bodies(m)}
    appr = tcp[:3, 3] - _body_T(m, d, 'gripper_body_link')[:3, 3]
    appr /= np.linalg.norm(appr)                            # the hand's approach direction (palm -> TCP)
    fit = None
    for out in np.arange(0.0, 0.0301, 0.005):               # slid out along the fingers (the palm housing is wide) ...
        for dz in np.arange(0.0, max(0.0, ext[2] / 2 - 0.015) + 1e-9, 0.005):   # ... and/or held nearer its top
            T_obj[:3, 3] = grip + out * appr - Rz @ c_local + np.array([0, 0, -dz])
            _set_obj(m, d, held_body, T_obj)
            mujoco.mj_forward(m, d)
            if not _contacts_of(m, d, held_geom, set(range(m.ngeom)) - rgeoms):
                fit = (out, dz)
                break
        if fit:
            break
    if not fit:
        return None, 'held object does not fit in the hand'
    grip = grip + fit[0] * appr                             # the grip point from here on
    grip_dz = float(grip[2] - (T_obj[2, 3] + lo[2]))        # grip point above the object's bottom
    g = _body_T(m, d, 'gripper_body_link')
    grip_off = tcp[:3, :3].T @ (tcp[:3, 3] - grip)          # the TCP seen from the grip point, in the TCP frame
    T_rel = np.linalg.inv(g) @ T_obj                        # object in the gripper frame (kept while welded)
    e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f'grasp_{held_body[4:]}')
    qg, qo, qi, qr = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(qg, g[:3, :3].flatten()); mujoco.mju_mat2Quat(qo, T_obj[:3, :3].flatten())
    mujoco.mju_negQuat(qi, qg); mujoco.mju_mulQuat(qr, qi, qo)
    m.eq_data[e][:3] = 0.0; m.eq_data[e][3:6] = T_rel[:3, 3]; m.eq_data[e][6:10] = qr; m.eq_data[e][10] = 1.0
    d.eq_active[e] = 1
    ja = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ('base_x', 'base_y', 'base_yaw')}
    jv = {j: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ja}
    obj_dofs = [m.jnt_dofadr[j] for j in range(m.njnt) if m.joint(j).name.startswith('obj_') and
                m.joint(j).name != f'{held_body}_free']
    for _k in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if _k * m.opt.timestep > 0.4 and _k % 25 == 0 and all(np.linalg.norm(d.qvel[a:a + 6]) < 0.01 for a in obj_dofs):
            break
    robot = pk.robot_bodies(m)
    env_geoms = {gg for gg in range(m.ngeom) if m.geom_bodyid[gg] not in robot and m.geom(gg).name != 'floor'
                 and gg != held_geom}
    start_hits = set(pk.bad_contacts(m, d, robot, env_geoms))
    if start_hits:
        return None, f'robot starts in contact {sorted(start_hits)[:2]}'
    if _contacts_of(m, d, held_geom, set(gg for gg in range(m.ngeom) if m.geom_bodyid[gg] in robot)):
        return None, 'held object touches something at the start'
    if np.linalg.norm(_body_T(m, d, held_body)[:3, 3] - T_obj[:3, 3]) > 0.02:
        return None, 'held object slipped while settling'
    P = np.array([*rand['place'][:2], o['top']], float)    # o['top']: the surface height measured by episode_model
    # the support surface under the point: exempt from the furniture margin (the hand must come close to it)
    gid = np.array([-1], np.int32)
    geomgroup = None
    dist = mujoco.mj_ray(m, d, P + np.array([0, 0, 0.05]), np.array([0, 0, -1.0]), geomgroup, 1, -1, gid)
    support = int(gid[0]) if dist >= 0 else -1
    # every geom of the support's body (a stool's top + edge boards): released objects may touch them
    # (furniture in the world body: the geoms sharing the name prefix, 'stool_top' -> 'stool_*')
    sup = {support}
    if support >= 0:
        bid = m.geom_bodyid[support]
        pre = (m.geom(support).name or '').split('_')[0] + '_'
        sup = {gg for gg in range(m.ngeom) if (m.geom_bodyid[gg] == bid if bid else
                                                (m.geom(gg).name or '').startswith(pre) and len(pre) > 1)} | {support}
    csets = pk.clearance_sets(m, d, robot, info.get('robot', (0, 0, 0)))

    def _top(gg):
        R = d.geom_xmat[gg].reshape(3, 3)
        return float((d.geom_xpos[gg] + R @ m.geom_aabb[gg][:3])[2] + np.abs(R[2]) @ m.geom_aabb[gg][3:])
    top = {gg for gg in sup if abs(_top(gg) - P[2]) < 0.025} | {support}   # the boards at the surface height only
    csets['furniture'] = [gg for gg in csets['furniture'] if gg not in top]
    cstart = pk.clearance(m, d, csets)
    base0 = np.array([d.qpos[ja['base_x']], d.qpos[ja['base_y']], d.qpos[ja['base_yaw']]])
    robot_geoms = {gg for gg in range(m.ngeom) if m.geom_bodyid[gg] in robot}
    return dict(mode=mode, ref=ref, q0=q0, n_sub=n_sub, ja=ja, jv=jv, robot=robot, env_geoms=env_geoms,
                start_hits=start_hits, start_self=set(pk.self_contacts(m, d, robot)), csets=csets, cstart=cstart,
                base0=base0, hx=math.cos(base0[2]), hy=math.sin(base0[2]), onehot=[1.0 if mode == k else 0.0 for k in pk.MODES],
                hits=[], min_cl=dict(cstart), k=0, held_body=held_body, held_geom=held_geom, T_rel=T_rel, weld=e,
                width=width, support=sup, grip_off=grip_off, grip_dz=grip_dz, robot_geoms=robot_geoms, P=P, h=float(ext[2]),
                P_base=list(pk.to_base(P, (base0[0], base0[1], base0[2])))), ''


def _carry(m, d, ctx):
    """Kinematic: put the held object where the weld would hold it (planning only)."""
    _set_obj(m, d, ctx['held_body'], _body_T(m, d, 'gripper_body_link') @ ctx['T_rel'])
    mujoco.mj_forward(m, d)


def _leg_bad(m, scratch, ctx, q, legs, rb, n=12):
    ax, ay = pk.base_adr(m)
    others = ctx['robot_geoms']                         # carried: nothing but the hand, not even the support
    for xa, xb, la, lb, ba, bb in legs:
        for i in range(1, n + 1):
            s_ = pk.minjerk(i / n)
            cs.set_q(m, scratch, dict(q, **dict(zip(ARM, xa + (xb - xa) * s_)), lift_joint=la + (lb - la) * s_),
                     {'gripper': pk.cs_cfg_gripper()})
            bs = ba + (bb - ba) * s_
            scratch.qpos[ax], scratch.qpos[ay] = rb[0] + bs * math.cos(rb[2]), rb[1] + bs * math.sin(rb[2])
            mujoco.mj_forward(m, scratch)
            _carry(m, scratch, ctx)
            if [h for h in pk.bad_contacts(m, scratch, ctx['robot'], ctx['env_geoms']) if h not in ctx['start_hits']]:
                return 'path: robot contact'
            if [h for h in pk.self_contacts(m, scratch, ctx['robot']) if h not in ctx['start_self']]:
                return 'path: self contact'
            if _contacts_of(m, scratch, ctx['held_geom'], others):
                return 'path: object contact'
            bad = pk.clear_ok(pk.clearance(m, scratch, ctx['csets']), ctx['cstart'])
            if bad:
                return f'path: clearance {bad[0]}'
    return ''


def plan(m, d, ctx, rand):
    ref, q0, P = ctx['ref'], ctx['q0'], ctx['P']
    rb = (ctx['base0'][0], ctx['base0'][1], ctx['base0'][2])
    Pb = pk.to_base(P, rb)
    tcp0 = TcpIK(ref, q0).T(np.array([q0[j] for j in ARM]))
    grip_b = Pb + np.array([0, 0, EXPERT['clear'] + ctx['grip_dz']])       # grip point, the bottom 1 cm up
    lifts = [P[2] + 0.1, P[2] + 0.2, P[2], P[2] + 0.3]
    lifts = list(dict.fromkeys(round(pk.lift_joint(x), 3) for x in lifts))
    lo_c, hi_c = rand.get('comfort', (-0.15, 0.8))
    sgn = 1.0 if Pb[0] > (lo_c + hi_c) / 2 else -1.0
    need = max(0.0, lo_c - Pb[0], Pb[0] - hi_c)
    shifts = [sgn * b for b in (0.0, 0.1, 0.2) if b >= need - 1e-9 or b == 0.2]
    seeds = [np.array([q0[j] for j in ARM])] + [np.radians(pz[ctx['mode']]) for pz in (pk.APPROACH_STANDING, pk.PRE_PICK_HIGH)
                                                if ctx['mode'] in pz]
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos
    scratch.eq_active[:] = 0
    x0 = np.array([q0[j] for j in ARM])
    for bx in shifts:
        sh = np.array([bx, 0, 0])
        for lj in lifts:
            q = dict(q0, lift_joint=lj)
            ik = TcpIK(ref, q)
            for yd in EXPERT['yaws_deg']:
                a = math.radians(yd)
                Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                R = Rz @ tcp0[:3, :3]                     # turned about the vertical: the object stays upright
                goal_b = grip_b + R @ ctx['grip_off']
                for sd in seeds:
                    g = ik.solve(R, goal_b - sh, [sd])
                    if g is None or g[1] > EXPERT['ik_tol']:
                        PLAN_STATS['ik goal'] += 1
                        continue
                    ab = ik.solve(R, goal_b + np.array([0, 0, EXPERT['above']]) - sh, [g[0]])
                    if ab is None or ab[1] > 0.006:
                        PLAN_STATS['ik above'] += 1
                        continue
                    legs = [(x0, ab[0], q0['lift_joint'], lj, 0.0, bx), (ab[0], g[0], lj, lj, bx, bx)]
                    why = _leg_bad(m, scratch, ctx, q, legs, rb)
                    if why:
                        PLAN_STATS[why] += 1
                        continue
                    return dict(lift=lj, base=bx, goal=g[0], above=ab[0], yaw=yd)
    return None


def apply(m, d, ctx, cfg, x, grip, lift, base, held):
    """One step + the held-object checks: while carried it may touch only the hand; while released, also its support."""
    pk.apply_step(m, d, ctx, cfg, x, grip, lift, base)
    if held is False:
        return
    hit = _contacts_of(m, d, ctx['held_geom'], ctx['robot_geoms'] | (ctx['support'] if held is None else set()))
    if hit:
        ctx['hits'].append(('held', m.geom(hit[0]).name))


def run_episode(m, d, task, cfg, rand=None, info=None, render=None, log=print, **_):
    rec = dict(action=[], state=[], phase=[], success=False, metric=float('nan'), reason='')
    ctx, why = setup(m, d, task, cfg, rand, info)
    if ctx is None:
        rec['reason'] = why
        return rec
    p = plan(m, d, ctx, rand)
    if p is None:
        rec['reason'] = 'no reachable collision-free place path'
        return rec
    q0, w = ctx['q0'], ctx['width']
    x0 = np.array([q0[j] for j in ARM])
    t_tr = max(2.0, np.max(np.abs(p['above'] - x0)) / EXPERT['arm_speed'],
               abs(p['lift'] - q0['lift_joint']) / EXPERT['lift_speed'], abs(p['base']) / EXPERT['base_speed'])
    L, B = p['lift'], p['base']
    # (name, T, x_a, x_b, lift_a, lift_b, base_a, base_b, grip_a, grip_b, held)
    phases = [('hold', EXPERT['hold'], x0, x0, q0['lift_joint'], q0['lift_joint'], 0, 0, w, w, True),
              ('transfer', t_tr, x0, p['above'], q0['lift_joint'], L, 0, B, w, w, True),
              ('lower', EXPERT['lower'], p['above'], p['goal'], L, L, B, B, w, w, True),
              ('release', EXPERT['release'], p['goal'], p['goal'], L, L, B, B, w, 0.1, None),
              ('up', EXPERT['up'], p['goal'], p['above'], L, L, B, B, 0.1, 0.1, False),
              ('settle', EXPERT['settle'], p['above'], p['above'], L, L, B, B, 0.1, 0.1, False)]
    for name, T, xa, xb, la, lb, ba, bb, ga, gb, held in phases:
        if name == 'release':
            d.eq_active[ctx['weld']] = 0                     # let go (the fingers open over the phase)
        steps = max(1, int(round(T * HZ)))
        for k in range(steps):
            s = pk.minjerk((k + 1) / steps)
            x, lift, base, grip = xa + (xb - xa) * s, la + (lb - la) * s, ba + (bb - ba) * s, ga + (gb - ga) * s
            state = pk.state_vec(m, d, ctx) + ctx['P_base']
            action = list(x) + [grip, lift, base]
            rec['state'].append(state); rec['action'].append(action); rec['phase'].append(name)
            if render:
                render(m, d, name, float('nan'), state=state, action=action)
            apply(m, d, ctx, cfg, x, grip, lift, base, held)
        T_h = _body_T(m, d, ctx['held_body'])
        rec.setdefault('phase_end', []).append((name, round(math.degrees(math.acos(float(np.clip(T_h[2, 2], -1, 1)))), 1),
                                                np.round(T_h[:3, 3] - ctx['P'], 3).tolist()))
    rec.update(score(m, d, ctx, rec['action'][-1]))
    rec['plan'] = dict(lift=p['lift'], base=p['base'], yaw=p['yaw'], retry=False)
    return {k: (np.array(v_) if k in ('action', 'state') else v_) for k, v_ in rec.items()}


def score(m, d, ctx, last_action):
    for _k in range(int(0.3 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    T = _body_T(m, d, ctx['held_body'])
    gid = ctx['held_geom']
    centre = d.geom_xpos[gid] + d.geom_xmat[gid].reshape(3, 3) @ m.geom_aabb[gid][:3]
    err = float(np.linalg.norm(centre[:2] - ctx['P'][:2]))
    tilt = math.degrees(math.acos(float(np.clip(T[2, 2], -1, 1))))
    on = abs(centre[2] - (ctx['P'][2] + ctx['h'] / 2)) < 0.03
    touching = bool(_contacts_of(m, d, gid, set(range(m.ngeom)) - ctx['robot_geoms']))
    jid = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ctx['held_body'])]
    still = float(np.linalg.norm(d.qvel[m.jnt_dofadr[jid]:m.jnt_dofadr[jid] + 3])) < 0.02
    track = float(np.max(np.abs(np.array(last_action[:7]) - np.array([cs.get_q(m, d, j) for j in ARM]))))
    hits = ctx['hits']
    ok = bool(err <= 0.05 and tilt < 20 and on and not touching and still and not hits and track < 0.05)
    reason = '' if ok else (f'collision / clearance {sorted(set(map(str, hits)))[:2]}' if hits else
                            f'arm blocked ({track:.2f} rad)' if track >= 0.05 else
                            'gripper still on the object' if touching else
                            f'tilted {tilt:.0f} deg' if tilt >= 20 else f'off the surface' if not on else
                            f'{err * 100:.1f} cm from the point' if err > 0.05 else 'still moving')
    return dict(success=ok, reason=reason, metric=err, place_err=err, tilt=tilt, track_err=track,
                clearance={k: round(v_, 3) for k, v_ in ctx['min_cl'].items()}, collisions=sorted(set(map(str, hits)))[:5])


def main():
    import imageio, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--task', default=f'{SIM}/config/task_place.yaml')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(a.task))
    rcfg = yaml.safe_load(open(task['rand_config']))
    out = f'{SIM}/out/place'
    os.makedirs(out, exist_ok=True)
    ok = 0
    for e in range(a.n):
        rand = sample_rand(np.random.default_rng(a.seed * 1000 + e), rcfg, task)
        m, d, info = episode_model(task, rand)
        rh, rw = mujoco.Renderer(m, 240, 424), mujoco.Renderer(m, 240, 424)
        frames = []

        def on_step(m, d, phase, metric, state=None, action=None):
            if len(frames) % 2:
                frames.append(None); return
            rh.update_scene(d, 'head_rgb'); rw.update_scene(d, 'wrist_rgb')
            frames.append(np.concatenate([rh.render(), rw.render()], 1))
        rec = run_episode(m, d, task, cfg, rand=rand, info=info, render=on_step)
        ok += rec['success']
        print(f"ep {e}: {rand['spot']} mode {rand['mode']} held {rand['held']} -> {'OK' if rec['success'] else 'FAIL'} "
              f"{rec.get('reason')} err {rec.get('place_err')} frames {len(rec['action'])} plan {rec.get('plan')}", flush=True)
        fr = [f for f in frames if f is not None]
        if fr:
            imageio.mimwrite(f'{out}/ep{e}.mp4', fr, fps=15, quality=6, macro_block_size=1)
        rh.close(); rw.close()
    print(f'{ok}/{a.n} OK')


if __name__ == '__main__':
    main()
