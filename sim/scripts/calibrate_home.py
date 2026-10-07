#!/usr/bin/env python3
"""Fit the cameras' exposure / white balance / saturation to the robot's own images at home.

    python3 scripts/calibrate_home.py   -> /sim/config/camera_home.yaml  (+ out/house/calib_home.jpg)

Real: sim/assets/real_home/robot_cams/{femto,d405}_drawer.jpg (the robot at the drawer, 2026-10-06).
Sim : the house open-drawer scene, cabinet at the `fridge` site (where those images were taken), nominal ready
      pose; MuJoCo segmentation gives the exact pixels of each surface, the real image hand-picked patches.
  ae_target   = mean luminance of the real image
  color_gain  = white balance from NEUTRAL surfaces only (white fridge / cabinets / walls, grey floor):
                per-channel ratio of real to sim chromaticity, normalised to keep luminance
  saturation  = the value that gives the wood the real chromaticity (saturation, then gain, as apply() does)
calibrate_color.py fitted the gain on the wood alone (office), which also absorbed the sim-vs-real wood hue and
tinted every white surface yellow at home. camera_model.load_camera_models(profile='home') applies this file
over camera_real.yaml.
"""
import os, sys

import cv2
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import camera_model, check_sim as cs, drawer_ready as dr  # noqa: E402

REAL = f'{SIM}/assets/real_home/robot_cams'
Y = np.array([0.299, 0.587, 0.114])
# camera -> (real image, {surface: ((x0, y0, x1, y1) in the real image, sim geom name prefixes, neutral?)})
TARGETS = {
    'head': ('femto_drawer.jpg', {'wood': ((260, 230, 480, 360), ['cab_top'], False),
                                  'floor': ((800, 320, 1200, 650), ['floor_'], True),
                                  'white': ((470, 10, 620, 200), ['fridge_body', 'wall'], True)}),
    'wrist': ('d405_drawer.jpg', {'wood': ((268, 232, 368, 330), ['cab_drawer1_front'], False),
                                  'white': ((180, 20, 420, 130), ['fridge_body', 'wall', 'upper_cabinets', 'kshelf'], True),
                                  'floor': ((650, 380, 840, 470), ['floor_'], True)}),
}


def main():
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer_house.yaml'))
    base = camera_model._load()
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q = dr.ready_q(task, task['mode'], cfg['home'])
    xml, *_ = dr.build_scene(task, ref.fk(q)['tool_tcp_link'],
                             rand=dict(textured=True, tex_tint=(1.0, 1.0, 1.0), q_extra={}, drawer0=0.0, site='fridge'))
    path = f'{SIM}/mjcf/_calib_home.xml'; open(path, 'w').write(xml)
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m); os.remove(path)
    cs.set_q(m, d, q, cfg); mujoco.mj_forward(m, d)
    names = [m.geom(i).name for i in range(m.ngeom)]
    out, rows = {}, []
    for cam, cname, (w, h) in (('head', 'head_rgb', (1280, 720)), ('wrist', 'wrist_rgb', (848, 480))):
        r = mujoco.Renderer(m, h, w)
        r.update_scene(d, cname); rgb = r.render().copy()
        r.enable_segmentation_rendering(); r.update_scene(d, cname); seg = r.render()[..., 0].copy(); r.close()
        img, surf = TARGETS[cam]
        real = cv2.cvtColor(cv2.imread(f'{REAL}/{img}'), cv2.COLOR_BGR2RGB).astype(float)
        ae = round(float((real @ Y).mean()), 2)
        cm = camera_model.CameraModel(dict(base[cam], ae_target=ae, color_gain=[1, 1, 1], saturation=1.0))
        sim = cm.apply(rgb, None, stages=('ae',)).astype(float)
        seg = cv2.remap(seg.astype(np.float32), cm.map_x, cm.map_y, cv2.INTER_NEAREST).astype(int)
        pairs = {}
        for k, ((x0, y0, x1, y1), prefixes, neutral) in surf.items():
            ids = [i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)]
            mask = np.isin(seg, ids)
            if mask.sum() < 300:
                print(f'{cam} {k}: only {mask.sum()} px visible in sim, skipped'); continue
            pairs[k] = (real[y0:y1, x0:x1].reshape(-1, 3).mean(0), sim[mask].mean(0), neutral)
        chrom = lambda c: c / (c @ Y)
        neutral = [chrom(rr) / chrom(ss) for rr, ss, n in pairs.values() if n]
        gain = np.mean(neutral, 0); gain = gain / (gain @ Y)
        rr, ss, _ = pairs['wood']                       # saturation: match the wood's chromaticity, in apply()'s order
        lum = ss @ Y
        cands = np.linspace(0.5, 5.0, 451)
        err = [np.linalg.norm(chrom((lum + c * (ss - lum)) * gain) - chrom(rr)) for c in cands]
        sat = float(cands[int(np.argmin(err))])
        out[cam] = dict(ae_target=ae, color_gain=[round(float(g), 3) for g in gain], saturation=round(sat, 3))
        for k, (rr, ss, n) in pairs.items():
            lum = ss @ Y
            pred = (lum + sat * (ss - lum)) * gain
            rows.append(f'{cam:5s} {k:5s} real {rr.round(0)}  sim before {ss.round(0)}  after {pred.round(0)}')
        print(f'{cam}: {out[cam]}   (office: gain {base[cam]["color_gain"]}, saturation {base[cam]["saturation"]})')
        # side by side: real | sim with the office colour | sim with the home colour
        office = camera_model.CameraModel(base[cam]).apply(rgb, None)
        home = camera_model.CameraModel(dict(base[cam], **out[cam])).apply(rgb, None)
        sz = (848, 480)
        tile = [cv2.resize(cv2.cvtColor(real.astype(np.uint8), cv2.COLOR_RGB2BGR), sz)] + \
               [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), sz) for x in (office, home)]
        for t, s_ in zip(tile, ('REAL ' + cam, 'SIM office colour', 'SIM home colour')):
            cv2.putText(t, s_, (10, 30), 0, 0.9, (0, 0, 255), 2)
        out.setdefault('_tiles', []).append(np.concatenate(tile, 1))
    print('\n'.join(rows))
    cv2.imwrite(f'{SIM}/out/house/calib_home.jpg', np.concatenate(out.pop('_tiles'), 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    out['_source'] = 'calibrate_home.py: sim/assets/real_home/robot_cams/*_drawer.jpg, fridge site, 2026-10-07'
    yaml.safe_dump(out, open(f'{SIM}/config/camera_home.yaml', 'w'), sort_keys=False)
    print('->', f'{SIM}/config/camera_home.yaml')


if __name__ == '__main__':
    main()
