"""At the first 'grasp' frame: where are the fingers relative to the handle bar?"""
import sys, mujoco, numpy as np, yaml
sys.path.insert(0, '/sim/scripts'); import expert_open_drawer as e
cfg = yaml.safe_load(open('/sim/config/mjcf.yaml')); task = yaml.safe_load(open('/sim/config/task_open_drawer.yaml'))
m = mujoco.MjModel.from_xml_path('/sim/mjcf/scene_open_drawer_left.xml'); d = mujoco.MjData(m)
done = []
def probe(m, d, phase, drawer):
    if phase != 'grasp' or done: return
    done.append(1)
    h = m.geom('cab_drawer2_handle').id
    print('handle geom pos', np.round(d.geom_xpos[h], 4), 'contype', m.geom_contype[h], 'conaff', m.geom_conaffinity[h])
    tcp = d.body('tool_tcp_link'); R = tcp.xmat.reshape(3, 3)
    print('TCP', np.round(tcp.xpos, 4), 'tool z', np.round(R[:, 2], 2), 'tool y (finger axis)', np.round(R[:, 1], 2))
    print('handle in TCP frame', np.round(R.T @ (d.geom_xpos[h] - tcp.xpos), 4))
    ft = np.zeros(6)
    for f in ('left_finger_tip', 'right_finger_tip'):
        b = m.body(f).id
        ds = [mujoco.mj_geomDistance(m, d, g, h, 0.2, ft) for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
        print(f, 'pos in TCP frame', np.round(R.T @ (d.xpos[b] - tcp.xpos), 4), 'min dist to handle', np.round(min(ds), 4))
    print('robot geom contype/conaff sample', m.geom_contype[m.geom('left_finger_tip_col0').id], m.geom_conaffinity[m.geom('left_finger_tip_col0').id])
    print('front geom contype', m.geom_contype[m.geom('cab_drawer2_front').id])
e.run_episode(m, d, task, cfg, 'left', render=probe, log=lambda *a: None)
