"""Which robot body draws the grey band across the wrist camera image?"""
import sys, mujoco, numpy as np, yaml
sys.path.insert(0, '/sim/scripts'); import check_sim as cs, drawer_ready as dr
cfg = yaml.safe_load(open('/sim/config/mjcf.yaml')); task = yaml.safe_load(open('/sim/config/task_open_drawer.yaml'))
m = mujoco.MjModel.from_xml_path('/sim/mjcf/scene_open_drawer_left.xml'); d = mujoco.MjData(m)
cs.set_q(m, d, dr.ready_q(task, 'left', cfg['home']), cfg); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 848)
def band():
    r.update_scene(d, 'wrist_rgb'); img = r.render().astype(float)
    return img[240:260].mean()
base = band(); print('band mean (all visible)', round(base, 1))
cam_pos = d.cam_xpos[m.camera('wrist_rgb').id]
bodies = sorted({m.geom_bodyid[g] for g in range(m.ngeom) if np.linalg.norm(d.geom_xpos[g] - cam_pos) < 0.25})
for b in bodies:
    saved = m.geom_rgba.copy()
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b: m.geom_rgba[g, 3] = 0
    v = band(); m.geom_rgba[:] = saved
    print(f'  hide {m.body(b).name:28s} -> {v:6.1f}{"   <== changes" if abs(v - base) > 3 else ""}')
print('camera pos', np.round(cam_pos, 3), 'znear', m.vis.map.znear, 'extent', round(m.stat.extent, 2))
