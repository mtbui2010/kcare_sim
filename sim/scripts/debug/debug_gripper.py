"""Close the gripper in free space at home and list what the fingers touch."""
import sys, mujoco, numpy as np, yaml
v = sys.argv[1] if len(sys.argv) > 1 else 'slamtec'
m = mujoco.MjModel.from_xml_path(f'/sim/mjcf/scene_tabletop_{v}.xml'); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[m.jnt_qposadr[m.joint('cube_free').id]] = 5  # move cube away
d.ctrl[:] = m.key_ctrl[0]; d.ctrl[m.actuator('gripper').id] = 0.0
for _ in range(1500): mujoco.mj_step(m, d)
print('free-space closed width', d.joint('virtual_gripper_joint').qpos, 'fingers', d.joint('left_finger_tip_joint').qpos, d.joint('right_finger_tip_joint').qpos)
for c in d.contact[:d.ncon]:
    b = lambda g: m.body(m.geom_bodyid[g]).name
    print(' contact', b(c.geom1), m.geom(c.geom1).name, '<->', b(c.geom2), m.geom(c.geom2).name, f'{c.dist*1000:.1f}mm')
# finger geometry extents in the gripper body frame
for f in ['left_finger_tip', 'right_finger_tip']:
    bid = m.body(f).id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_contype[g]:
            print(f, 'geom', g, 'aabb', np.round(m.geom_aabb[g], 4), 'pos', np.round(m.geom_pos[g], 4))
tcp = d.body('tool_tcp_link'); lf = d.body('left_finger_tip')
R = tcp.xmat.reshape(3, 3)
print('left finger origin in tcp frame', np.round(R.T @ (lf.xpos - tcp.xpos), 4))
