import json, sys, mujoco, numpy as np, yaml
sys.path.insert(0, '/sim/scripts'); import check_sim as cs
v = 'slamtec'; cfg = yaml.safe_load(open('/sim/config/mjcf.yaml'))
r = json.load(open(f'/sim/out/{v}/report.json'))['grasp']
m = mujoco.MjModel.from_xml_path(f'/sim/mjcf/scene_tabletop_{v}.xml'); d = mujoco.MjData(m)
q = dict(cfg['home'], lift_joint=0.35, **r['arm_q']); q['virtual_gripper_joint'] = 0.1
cs.set_q(m, d, q, cfg); tcp = np.array(r['tcp'])
d.qpos[m.jnt_qposadr[m.joint('cube_free').id]:][:3] = [tcp[0], tcp[1], tcp[2] + 0.01 + 0.02 + 1e-4]
m.geom_pos[m.geom('table').id] = [tcp[0], tcp[1], tcp[2] + 0.01 - m.geom_size[m.geom('table').id][2]]
mujoco.mj_forward(m, d)
cube = m.geom('cube').id
print('cube contype/conaff', m.geom_contype[cube], m.geom_conaffinity[cube])
for f in ['left_finger_tip']:
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == m.body(f).id:
            print(' ', m.geom(g).name, 'type', m.geom_type[g], 'contype', m.geom_contype[g], 'conaff', m.geom_conaffinity[g], 'group', m.geom_group[g], 'rbound', round(m.geom_rbound[g], 4))
q['virtual_gripper_joint'] = 0.0; d.ctrl[:] = cs.ctrl_from(m, q, cfg)
for k in range(1000):
    mujoco.mj_step(m, d)
seen = set()
for c in d.contact[:d.ncon]:
    seen.add((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))
print('t', d.time, 'width', d.joint('virtual_gripper_joint').qpos, 'contacts:', seen)
print('exclude pairs', m.nexclude, [(m.body(m.exclude_signature[i] >> 16).name, m.body(m.exclude_signature[i] & 0xFFFF).name) for i in range(m.nexclude)])
print('cube xpos', np.round(d.body('cube').xpos, 4))
ft = np.zeros(6)
for f in ['left_finger_tip', 'right_finger_tip']:
    ds = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == m.body(f).id and m.geom_contype[g]:
            ds.append(mujoco.mj_geomDistance(m, d, g, cube, 0.1, ft))
    print(f, 'min dist to cube', np.round(min(ds), 4), 'finger xpos', np.round(d.body(f).xpos, 4))
print('bodyweld cube', m.body_weldid[m.body('cube').id], 'finger', m.body_weldid[m.body('left_finger_tip').id])
print('opt disableflags', m.opt.disableflags, 'nconmax?', d.ncon)
