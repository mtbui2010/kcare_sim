"""Head-camera views of the house drawer scene at several head tilts -> out/head_tilts.jpg"""
import os, sys, copy
import cv2, mujoco, numpy as np, yaml
sys.path.insert(0, '/sim/scripts')
import check_sim as cs, drawer_ready as dr
cfg = yaml.safe_load(open('/sim/config/mjcf.yaml')); task0 = yaml.safe_load(open('/sim/config/task_open_drawer_house.yaml'))
ref = cs.UrdfFK('/sim/urdf/raw/kaair_slamtec.urdf')
tiles = []
for ry in (-41, -55, -65, -75):
    task = copy.deepcopy(task0); task['ready_pose']['head']['ry'] = ry
    q = dr.ready_q(task, task['mode'], cfg['home'])
    xml, *_ = dr.build_scene(task, ref.fk(q)['tool_tcp_link'])
    open('/sim/mjcf/_tilt.xml', 'w').write(xml)
    m = mujoco.MjModel.from_xml_path('/sim/mjcf/_tilt.xml'); d = mujoco.MjData(m)
    cs.set_q(m, d, q, cfg); mujoco.mj_forward(m, d)
    for k in (0, 1, 2):
        d.qpos[m.jnt_qposadr[m.joint(f'cab_drawer{k}_slide').id]] = 0.0
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 360, 640); r.update_scene(d, 'head_rgb')
    im = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR); cv2.putText(im, f'head ry {ry}', (10, 30), 0, 1, (0, 0, 255), 2); tiles.append(im); r.close()
cv2.imwrite('/sim/out/head_tilts.jpg', np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
