#!/usr/bin/env python3
"""Sim vs real, side by side — the check before generating data.

    python3 scripts/render_real_compare.py /nas/real/20260928_scene1
      -> /sim/out/real_compare/compare.png   real | sim for the head and wrist cameras,
                                              sim posed at the recorded joint state
         /sim/out/real_compare/samples.png   randomized training scenes (config/randomization_real.yaml),
                                              incl. both ends of the lift::0.4..0.6 range
Both go through camera_model.py exactly as the dataset will (distortion, colour
response, noise, JPEG; head then downsized to 848x480 like the recorder).
"""
import json, os, sys

import cv2
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import camera_model, check_sim as cs, drawer_ready as dr, randomize  # noqa: E402

IMG_WH = (848, 480)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return img


def render_pair(task, cfg, tcp_T, rand, q, cams, rng):
    xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
    path = f'{SIM}/mjcf/_compare_scene.xml'
    open(path, 'w').write(xml)
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
    cs.set_q(m, d, q, cfg)
    mujoco.mj_forward(m, d)
    out = {}
    for cam, name, (w, h) in (('head', 'head_rgb', (1280, 720)), ('wrist', 'wrist_rgb', (848, 480))):
        r = mujoco.Renderer(m, h, w); r.update_scene(d, name); img = r.render(); r.close()
        out[cam] = cams[cam].apply(img, rng)
    os.remove(path)
    return out


def main():
    real_dir = sys.argv[1]
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
    rcfg = yaml.safe_load(open(f'{SIM}/config/randomization_real.yaml'))
    cams = camera_model.load_camera_models()
    out_dir = f'{SIM}/out/real_compare'
    os.makedirs(out_dir, exist_ok=True)
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q_nom = dr.ready_q(task, task['mode'], cfg['home'])
    tcp_T = ref.fk(q_nom)['tool_tcp_link']                  # cabinet placed from the nominal ready pose

    # 1. sim at the recorded real joint state, nominal scene
    st = json.load(open(f'{real_dir}/robot_state.json'))['joints']
    q_real = dict(q_nom, **{n: p for n, p in zip(st['name'], st['position']) if n in q_nom})
    q_real['head_joint1'] += task['ready_pose'].get('head_pan_offset', 0.0)   # URDF head-zero calibration
    # nominal = office default lighting, the same scene calibrate_color.py fits the cameras on
    nominal = dict(textured=True, tex_tint=(1.0, 1.0, 1.0), q_extra={}, drawer0=0.0)
    rng = np.random.default_rng(0)
    for c in cams.values():
        c.episode_jitter(None)
    sim = render_pair(task, cfg, tcp_T, nominal, q_real, cams, rng)
    real_h = cv2.cvtColor(cv2.imread(f'{real_dir}/head.jpg'), cv2.COLOR_BGR2RGB)
    real_w = cv2.cvtColor(cv2.imread(f'{real_dir}/wrist.jpg'), cv2.COLOR_BGR2RGB)
    rs = lambda im: cv2.resize(im, IMG_WH, interpolation=cv2.INTER_AREA)
    grid = np.concatenate([
        np.concatenate([label(rs(real_h), 'REAL head (Femto)'), label(rs(sim['head']), 'SIM head')], 1),
        np.concatenate([label(real_w, 'REAL wrist (D405)'), label(sim['wrist'], 'SIM wrist')], 1)], 0)
    cv2.imwrite(f'{out_dir}/compare.png', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print('->', f'{out_dir}/compare.png', ' real joints:', {k: round(v, 3) for k, v in q_real.items() if k.startswith(('lift', 'head'))})

    # 2. randomized training scenes; first two pinned to the lift range ends
    rows = []
    for k in range(6):
        r_rng = np.random.default_rng(2_000_000 + k)
        rand = randomize.sample(r_rng, rcfg)
        if k < 2:
            x = rcfg['lift_inputs'][k]
            kc = rcfg['lift_kcare']
            rand['lift_input'] = x
            rand['q_extra']['lift_joint'] = randomize.kcare_lift_joint(x, kc) - randomize.kcare_lift_joint(kc['nominal'], kc)
        q = dict(q_nom)
        for j, dv in rand['q_extra'].items():
            q[j] += dv
        for c in cams.values():
            c.episode_jitter(r_rng)
        img = render_pair(task, cfg, tcp_T, rand, q, cams, r_rng)
        tag = (f"lift::{rand['lift_input']:.2f} (joint {q['lift_joint']:.3f})  "
               f"{'texture' if rand['textured'] else 'plain colour'}")
        rows.append(np.concatenate([label(rs(img['head']), tag), label(img['wrist'], 'wrist')], 1))
    samples = np.concatenate(rows, 0)
    cv2.imwrite(f'{out_dir}/samples.png', cv2.cvtColor(samples, cv2.COLOR_RGB2BGR))
    print('->', f'{out_dir}/samples.png')


if __name__ == '__main__':
    main()
