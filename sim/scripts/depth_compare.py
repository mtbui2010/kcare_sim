#!/usr/bin/env python3
"""Real vs simulated depth at the recorded pose of a real capture (office scene, nominal placement).

    python3 scripts/depth_compare.py /nas/real/20260928_scene1  -> /sim/out/real_compare/depth_compare.jpg

Row 1: real D405 depth | sim wrist depth, perfect | sim wrist depth through camera_model.DepthModel.
Row 2: sim head (Femto) depth, perfect | through DepthModel (no real Femto depth frame recorded yet).
Prints invalid fraction and local noise per distance for the real and the modelled wrist depth.
"""
import json, os, sys, time

import cv2
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import camera_model, check_sim as cs, drawer_ready as dr  # noqa: E402


def viz(z, zmax):
    v = cv2.applyColorMap(cv2.convertScaleAbs(z, alpha=255 / zmax), cv2.COLORMAP_JET)
    v[z <= 0] = 0
    return v


def stats(name, z_mm):
    z = z_mm.astype(np.float32); v = (z > 0).astype(np.float32)
    mu = cv2.blur(z, (7, 7)); sd = np.sqrt(np.maximum(cv2.blur(z * z, (7, 7)) - mu * mu, 0))
    ok = cv2.blur(v, (7, 7)) > 0.999
    s = [f'{name}: invalid {(z == 0).mean():.2f}']
    for z0 in (150, 300, 500, 800, 1500):
        k = ok & (mu > z0 * 0.8) & (mu < z0 * 1.2)
        if k.sum() > 200:
            s.append(f'{z0}mm sd {np.median(sd[k]):.1f}')
    print('  '.join(s))


def main():
    real_dir = sys.argv[1]
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
    cams, dms = camera_model.load_camera_models(), camera_model.load_depth_models()
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q_nom = dr.ready_q(task, task['mode'], cfg['home'])
    tcp_T = ref.fk(q_nom)['tool_tcp_link']
    st = json.load(open(f'{real_dir}/robot_state.json'))['joints']
    q = dict(q_nom, **{n: p for n, p in zip(st['name'], st['position']) if n in q_nom})
    q['head_joint1'] += task['ready_pose'].get('head_pan_offset', 0.0)
    xml, *_ = dr.build_scene(task, tcp_T, rand=dict(textured=True, tex_tint=(1.0, 1.0, 1.0), q_extra={}, drawer0=0.0))
    path = f'{SIM}/mjcf/_depth_compare.xml'; open(path, 'w').write(xml)
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
    cs.set_q(m, d, q, cfg); mujoco.mj_forward(m, d)
    rng = np.random.default_rng(0)
    out = {}
    for cam, name, (w, h) in (('head', 'head_rgb', (1280, 720)), ('wrist', 'wrist_rgb', (848, 480))):
        r = mujoco.Renderer(m, h, w); r.update_scene(d, name); rgb = r.render().copy()
        r.enable_depth_rendering(); z = r.render().copy(); r.close()
        z[z > 10] = 0
        cm = cams[cam]
        z = cv2.remap(z, cm.map_x, cm.map_y, cv2.INTER_NEAREST, borderValue=0.0)
        lum = cv2.remap(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cm.map_x, cm.map_y, cv2.INTER_LINEAR)
        t0 = time.time(); zm = dms[cam].apply(z, lum, rng); dt = time.time() - t0
        print(f'{cam}: DepthModel {dt * 1000:.0f} ms')
        out[cam] = (z, zm)
    os.remove(path)
    real = cv2.imread(f'{real_dir}/wrist_depth.png', -1)
    stats('REAL D405', real); stats('SIM  D405 model', out['wrist'][1] * 1000); stats('SIM  D405 perfect', out['wrist'][0] * 1000)
    zmax = float(np.percentile(real[real > 0], 99))
    lab = lambda im, t: cv2.putText(im, t, (10, 30), 0, 0.9, (255, 255, 255), 2) if True else im
    row1 = np.concatenate([lab(viz(real.astype(np.float32), zmax), 'REAL D405'),
                           lab(viz(out['wrist'][0] * 1000, zmax), 'SIM perfect'),
                           lab(viz(out['wrist'][1] * 1000, zmax), 'SIM D405 model')], 1)
    hz = lambda z: cv2.resize(viz(z * 1000, 4000), (848, 480), interpolation=cv2.INTER_NEAREST)
    row2 = np.concatenate([lab(hz(out['head'][0]), 'SIM head perfect'), lab(hz(out['head'][1]), 'SIM Femto model'),
                           np.zeros((480, 848, 3), np.uint8)], 1)
    os.makedirs(f'{SIM}/out/real_compare', exist_ok=True)
    cv2.imwrite(f'{SIM}/out/real_compare/depth_compare.jpg', np.concatenate([row1, row2], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('->', f'{SIM}/out/real_compare/depth_compare.jpg')


if __name__ == '__main__':
    main()
