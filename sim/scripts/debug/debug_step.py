"""Step one arm joint from home and list the robot contacts that stop it."""
import sys, mujoco, numpy as np, yaml
sys.path.insert(0, '/sim/scripts'); import check_sim as cs
v, j, dq = sys.argv[1], sys.argv[2], float(sys.argv[3])
cfg = yaml.safe_load(open('/sim/config/mjcf.yaml'))
m = mujoco.MjModel.from_xml_path(f'/sim/mjcf/scene_tabletop_{v}.xml'); d = mujoco.MjData(m)
q = dict(cfg['home']); cs.set_q(m, d, q, cfg); q[j] += dq
d.ctrl[:] = cs.ctrl_from(m, q, cfg)
for _ in range(500): mujoco.mj_step(m, d)
print(j, 'target', q[j], 'reached', d.joint(j).qpos[0])
for c in d.contact[:d.ncon]:
    print('  ', m.body(m.geom_bodyid[c.geom1]).name, '<->', m.body(m.geom_bodyid[c.geom2]).name, f'{c.dist*1000:.1f} mm')
