#!/usr/bin/env python3
"""Fit each camera's exposure / saturation / white balance to the real images.

    python3 scripts/calibrate_color.py /nas/real/20260928_scene1
      -> ae_target, saturation, color_gain in /sim/config/camera_real.yaml

The robot is posed at the recorded joint state in the nominal office scene.
MuJoCo's segmentation render gives the exact pixels of each surface; the real
image gives hand-picked patches of the same surfaces (2026-09-28 capture).
  ae_target   = mean luminance of the real image (what the camera's AE settles on)
  saturation  = |chroma real| / |chroma sim after AE|   (chroma = colour - luminance)
  color_gain  = remaining per-channel ratio, normalised to keep luminance (white balance)
Head: cabinet top + floor; wrist: drawer fronts.
"""
import json, os, sys

import cv2
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import check_sim as cs, drawer_ready as dr  # noqa: E402

# real patches (x0, y0, x1, y1) in the native image, and the sim geoms showing the same surface
TARGETS = {
    'head': [((230, 420, 330, 480), ['cab_top']),                     # cabinet top (beech)
             ((700, 450, 1000, 650), ['floor'])],                      # laminate floor
    'wrist': [((472, 238, 578, 338), ['cab_drawer1_front', 'cab_drawer2_front', 'cab_drawer0_front'])],
}


def main():
    real_dir = sys.argv[1]
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
    cam_path = f'{SIM}/config/camera_real.yaml'
    cam_cfg = yaml.safe_load(open(cam_path))
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q_nom = dr.ready_q(task, task['mode'], cfg['home'])
    st = json.load(open(f'{real_dir}/robot_state.json'))['joints']
    q = dict(q_nom, **{n: p for n, p in zip(st['name'], st['position']) if n in q_nom})
    q['head_joint1'] += task['ready_pose'].get('head_pan_offset', 0.0)       # URDF head-zero calibration
    nominal = dict(textured=True, tex_tint=(1.0, 1.0, 1.0), q_extra={}, drawer0=0.0)
    xml, *_ = dr.build_scene(task, ref.fk(q_nom)['tool_tcp_link'], rand=nominal)
    path = f'{SIM}/mjcf/_calib_scene.xml'
    open(path, 'w').write(xml)
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
    cs.set_q(m, d, q, cfg); mujoco.mj_forward(m, d)
    os.remove(path)

    import camera_model
    Y = np.array([0.299, 0.587, 0.114])
    for cam, name, (w, h) in (('head', 'head_rgb', (1280, 720)), ('wrist', 'wrist_rgb', (848, 480))):
        r = mujoco.Renderer(m, h, w)
        r.update_scene(d, name); rgb = r.render()
        r.enable_segmentation_rendering(); r.update_scene(d, name); seg = r.render()[..., 0]; r.close()
        real = cv2.cvtColor(cv2.imread(f'{real_dir}/{cam}.jpg'), cv2.COLOR_BGR2RGB).astype(float)
        cam_cfg[cam]['ae_target'] = round(float((real @ Y).mean()), 2)
        cm = camera_model.CameraModel(dict(cam_cfg[cam], color_gain=[1, 1, 1], saturation=1.0))
        sim = cm.apply(rgb, None, stages=('ae',)).astype(float)          # distortion + AE only
        seg = cv2.remap(seg.astype(np.float32), cm.map_x, cm.map_y, cv2.INTER_NEAREST).astype(int)
        pairs = []
        for (x0, y0, x1, y1), geoms in TARGETS[cam]:
            mask = np.isin(seg, [m.geom(g).id for g in geoms])
            if mask.sum() < 200:
                print(f'{cam}: {geoms} barely visible ({mask.sum()} px), skipped'); continue
            pairs.append((real[y0:y1, x0:x1].reshape(-1, 3).mean(0), sim[mask].mean(0), geoms[0]))
        sats = [np.linalg.norm(rr - rr @ Y) / max(np.linalg.norm(ss - ss @ Y), 1e-3) for rr, ss, _ in pairs]
        sat = float(np.mean(sats))
        ratios = []
        for rr, ss, g in pairs:
            pred = (ss @ Y) + sat * (ss - ss @ Y)
            ratios.append(rr / pred)
            print(f'{cam} {g:18s} real {rr.round(1)}  sim(AE) {ss.round(1)}  sim(AE+sat) {pred.round(1)}')
        gain = np.mean(ratios, 0)
        gain = gain / (gain @ Y)                                         # white balance only, keep luminance
        cam_cfg[cam].update(saturation=round(sat, 3), color_gain=[round(float(g), 3) for g in gain],
                            color_source='calibrate_color.py at the recorded pose (2026-09-28 capture)')
        cam_cfg[cam].pop('color_gain_source', None)
        print(f"{cam}: ae_target {cam_cfg[cam]['ae_target']}  saturation {sat:.2f}  color_gain {cam_cfg[cam]['color_gain']}")
    yaml.safe_dump(cam_cfg, open(cam_path, 'w'), sort_keys=False)


if __name__ == '__main__':
    main()
