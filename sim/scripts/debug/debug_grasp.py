"""At the check_sim grasp pose: where are the fingertips vs the cube, and what touches what."""
import json, sys, mujoco, numpy as np, yaml
sys.path.insert(0, '/sim/scripts'); import check_sim as cs
v = 'slamtec'; cfg = yaml.safe_load(open('/sim/config/mjcf.yaml'))
r = json.load(open(f'/sim/out/{v}/report.json'))['grasp']
m = mujoco.MjModel.from_xml_path(f'/sim/mjcf/scene_tabletop_{v}.xml'); d = mujoco.MjData(m)
q = dict(cfg['home'], lift_joint=0.35, **r['arm_q']); q['virtual_gripper_joint'] = 0.1
cs.set_q(m, d, q, cfg); tcp = np.array(r['tcp'])
mujoco.mj_forward(m, d)
top = min(cs.body_min_z(m, d, f) for f in ('left_finger_tip', 'right_finger_tip')) - 0.002
d.qpos[m.jnt_qposadr[m.joint('cube_free').id]:][:3] = [tcp[0], tcp[1], top + 0.02 + 1e-4]
m.geom_pos[m.geom('table').id] = [tcp[0], tcp[1], top - m.geom_size[m.geom('table').id][2]]
mujoco.mj_forward(m, d)
print('tcp', np.round(tcp, 4), 'body tool_tcp_link', np.round(d.body('tool_tcp_link').xpos, 4))
print('cube z range', np.round(d.body('cube').xpos[2] - 0.02, 4), np.round(d.body('cube').xpos[2] + 0.02, 4))
for f in ['left_finger_tip', 'right_finger_tip', 'gripper_body_link']:
    bid = m.body(f).id
    zs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_contype[g]:
            c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); a = m.geom_aabb[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * a[3:] + a[:3]
            w = (R @ corners.T).T + c; zs.append(w)
    w = np.vstack(zs)
    print(f, 'world z [%.4f, %.4f]  y [%.4f, %.4f]' % (w[:, 2].min(), w[:, 2].max(), w[:, 1].min(), w[:, 1].max()))

# ── simulate close + lift, log forces ──
d.ctrl[:] = cs.ctrl_from(m, q, cfg)
for _ in range(250): mujoco.mj_step(m, d)
q['virtual_gripper_joint'] = 0.0; d.ctrl[:] = cs.ctrl_from(m, q, cfg)
def log(tag):
    f6 = np.zeros(6); n = 0; tot = 0.0
    for i, c in enumerate(d.contact[:d.ncon]):
        names = {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name}
        if 'cube' in names and names & {'left_finger_tip', 'right_finger_tip'}:
            mujoco.mj_contactForce(m, d, i, f6); n += 1; tot += abs(f6[0])
    print(f'{tag}: t={d.time:.2f} width={d.joint("virtual_gripper_joint").qpos[0]:.4f} '
          f'lift={d.joint("lift_joint").qpos[0]:.4f} cube_z={d.body("cube").xpos[2]:.4f} '
          f'finger-cube contacts={n} sum normal={tot:.1f}N  grip actuator force={d.actuator_force[m.actuator("gripper").id]:.1f} '
          f'fingers L={d.joint("left_finger_tip_joint").qpos[0]:.4f} R={d.joint("right_finger_tip_joint").qpos[0]:.4f} cube_y={d.body("cube").xpos[1]:.4f}')
for k in range(750):
    mujoco.mj_step(m, d)
    if k % 50 == 49: log('close')
q['lift_joint'] += 0.1; d.ctrl[:] = cs.ctrl_from(m, q, cfg)
for k in range(1500):
    mujoco.mj_step(m, d)
    if k % 300 == 299: log('lift ')
