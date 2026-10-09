#!/usr/bin/env python3
"""Scripted expert for "close drawer" (VLA data), from the same kcare ready pose as open drawer.

The target drawer starts `drawer0` open (task drawer0_range). Phases, d = into the cabinet:
  hold      keep the ready pose
  clear     back off along -d with the fingers open (an open drawer's bar sits between them at the ready pose)
  pregrasp  turn the wrist like open-drawer (fingers vertical), close the gripper, stop `pre_d` in front of the
            handle bar (which now sticks out by drawer0)
  push      push along d until the drawer front is flush (+ `over` m), the closed fingers on the bar
  release   open the gripper (the fingers close around the bar on the way in)
  retreat   back off along -d
Same 8-D action / 30 Hz as expert_open_drawer.run_episode; success = the drawer ends < `closed_tol` open.
"""
import os, sys

import mujoco, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import cabinet, check_sim as cs, drawer_ready as dr
from expert_open_drawer import ArmIK, smooth, HZ, ARM

SIM = os.environ.get('SIM_ROOT', '/sim')
EXPERT = dict(pre_d=0.06, over=0.01, retreat=0.10, closed_width=0.0, finger_axis_z=1, closed_tol=0.02,
              durations=dict(hold=0.5, clear=1.5, pregrasp=2.5, push=3.0, release=1.0, retreat=1.5))


def run_episode(m, d, task, cfg, mode='left', render=None, log=print, q_extra=None, drawer0=0.18, target=None):
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q_ready = dr.ready_q(task, mode, cfg['home'])
    for j, dv in (q_extra or {}).items():
        q_ready[j] += dv
    ik = ArmIK(ref, q_ready)
    n_sub = max(1, round(1.0 / HZ / m.opt.timestep))
    m.opt.timestep = 1.0 / HZ / n_sub
    cs.set_q(m, d, q_ready, cfg)
    d.ctrl[:] = cs.ctrl_from(m, q_ready, cfg)
    drawer = f"cab_drawer{cabinet.STYLES[task['cabinet']]['target'] if target is None else target}"
    slide = m.joint(f'{drawer}_slide')
    adr = m.jnt_qposadr[slide.id]
    d.qpos[adr] = drawer0
    mujoco.mj_forward(m, d)

    T0 = ref.fk(q_ready)['tool_tcp_link']
    pull_axis = d.xmat[m.jnt_bodyid[slide.id]].reshape(3, 3) @ m.jnt_axis[slide.id]
    d_app = -pull_axis; d_app[2] = 0; d_app /= np.linalg.norm(d_app)
    handle = d.geom_xpos[m.geom(f'{drawer}_handle').id].copy()        # already pulled out by drawer0
    z_goal, y_goal = d_app, np.array([0, 0, float(EXPERT['finger_axis_z'])])
    pre_pos = handle - d_app * EXPERT['pre_d']
    push_pos = handle + d_app * (drawer0 + EXPERT['over'])
    retreat_pos = push_pos - d_app * (EXPERT['retreat'] + drawer0)
    open_w = task['ready_pose']['gripper']
    # an open top / middle drawer puts its bar right between the open fingers of the ready pose: back off first
    # with the fingers open (closing there grabbed the bar and the push turned into a pull)
    clear_pos = T0[:3, 3] - d_app * (drawer0 + 0.05)
    phases = [('hold', T0[:3, 3], T0[:3, 3], open_w), ('clear', T0[:3, 3], clear_pos, open_w),
              ('pregrasp', clear_pos, pre_pos, EXPERT['closed_width']),
              ('push', pre_pos, push_pos, EXPERT['closed_width']),
              # the closing fingers end up around the bar: open before backing off, or the drawer comes along
              ('release', push_pos, push_pos, open_w), ('retreat', push_pos, retreat_pos, open_w)]
    x_ready = np.array([q_ready[j] for j in ARM])
    rec = dict(action=[], state=[], drawer=[], phase=[])
    x, width = x_ready.copy(), open_w
    for name, p_from, p_to, w_end in phases:
        steps = int(round(EXPERT['durations'][name] * HZ))
        w0 = width
        for k in range(steps):
            s = smooth((k + 1) / steps)
            pos = p_from + (p_to - p_from) * s
            if name == 'hold':
                x_tgt = x_ready
            elif name == 'clear':
                x_tgt, _ = ik.solve(pos, T0[:3, 2], T0[:3, 1], x)
            elif name == 'pregrasp':
                z = T0[:3, 2] * (1 - s) + z_goal * s; z /= np.linalg.norm(z)   # from the ready orientation
                y = T0[:3, 1] * (1 - s) + y_goal * s; y -= z * np.dot(y, z); y /= np.linalg.norm(y)
                x_tgt, _ = ik.solve(pos, z, y, x)
            else:
                x_tgt, _ = ik.solve(pos, z_goal, y_goal, x)
            x = x_tgt
            width = w0 + (w_end - w0) * s
            q_cmd = dict(q_ready, **dict(zip(ARM, x)), virtual_gripper_joint=width)
            d.ctrl[:] = cs.ctrl_from(m, q_cmd, cfg)
            state = [cs.get_q(m, d, j) for j in ARM] + [cs.get_q(m, d, 'virtual_gripper_joint')]
            rec['state'].append(state); rec['action'].append(list(x) + [width]); rec['phase'].append(name)
            if render:
                render(m, d, name, float(d.qpos[adr]), state=state, action=rec['action'][-1])
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            rec['drawer'].append(float(d.qpos[adr]))
        log(f'  {name:9s} done: drawer {d.qpos[adr] * 100:5.1f} cm')
    rec = {k: np.array(v) for k, v in rec.items()}
    rec['success'] = bool(rec['drawer'][-1] < EXPERT['closed_tol'])
    return rec
