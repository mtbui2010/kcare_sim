#!/usr/bin/env python3
"""Check the corners of the randomization box before generating data.

    python3 scripts/check_ranges.py [--rand-config randomization_real.yaml]
      -> /sim/out/real_compare/ranges.png   head | wrist view at every lift x pan x tilt extreme
         prints whether the expert still opens the drawer there

For each corner: the cabinet must stay in the head camera's view (segmentation
pixel count) and the expert must succeed — otherwise the range is too wide.
"""
import argparse, itertools, math, os, sys

import cv2
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import camera_model, check_sim as cs, drawer_ready as dr, expert_open_drawer as ex, randomize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rand-config', default='randomization_real.yaml')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
    rcfg = yaml.safe_load(open(f'{SIM}/config/{a.rand_config}'))
    cams = camera_model.load_camera_models()
    kc, hj = rcfg['lift_kcare'], rcfg['head_jitter_deg']
    q_nom = dr.ready_q(task, task['mode'], cfg['home'])
    tcp_T = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf').fk(q_nom)['tool_tcp_link']
    rows, ok_all = [], True
    targets = sorted(rcfg.get('drawer_targets', {}).get('weights', {None: 1}))
    names = {0: 'bottom', 1: 'middle', 2: 'top', None: 'default'}
    for target, lift, pan, tilt in itertools.product(targets, rcfg['lift_inputs'], hj['pan'], hj['tilt']):
        rand = dict(textured=True, tex_tint=(1.0, 1.0, 1.0), drawer0=0.0,
                    q_extra=dict(lift_joint=randomize.kcare_lift_joint(lift, kc) - randomize.kcare_lift_joint(kc['nominal'], kc),
                                 head_joint1=math.radians(pan), head_joint2=math.radians(tilt)))
        xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
        path = f'{SIM}/mjcf/_ranges_scene.xml'
        open(path, 'w').write(xml)
        m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
        q = dict(q_nom); [q.__setitem__(j, q[j] + v) for j, v in rand['q_extra'].items()]
        cs.set_q(m, d, q, cfg); mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 720, 1280)
        r.update_scene(d, 'head_rgb'); head = cams['head'].apply(r.render())
        r.enable_segmentation_rendering(); r.update_scene(d, 'head_rgb'); seg = r.render()[..., 0]; r.close()
        cab_ids = [i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('cab')]
        cab_px = int(np.isin(seg, cab_ids).sum())
        r = mujoco.Renderer(m, 480, 848); r.update_scene(d, 'wrist_rgb'); wrist = cams['wrist'].apply(r.render()); r.close()
        rec = ex.run_episode(m, d, task, cfg, task['mode'], log=lambda *x: None, q_extra=rand['q_extra'],
                             target=target)
        others = [float(d.qpos[m.jnt_qposadr[m.joint(f'cab_drawer{i}_slide').id]]) for i in range(3) if i != target]
        os.remove(path)
        ok = rec['success'] and cab_px > 20000 and max(others) < 0.03
        ok_all &= ok
        tag = f'{names[target]:6s} lift::{lift:.2f} pan {-70 + pan:+.0f} tilt {-41 + tilt:+.0f}'
        print(f"{tag:40s} cabinet {cab_px / seg.size * 100:4.1f}% of head image, expert opened "
              f"{rec['drawer'][-1] * 100:4.1f} cm, other drawers {max(others) * 100:3.1f} cm -> {'OK' if ok else 'PROBLEM'}",
              flush=True)
        h = cv2.resize(head, (424, 240), interpolation=cv2.INTER_AREA); w = cv2.resize(wrist, (424, 240))
        tile = np.concatenate([h, w], 1).copy()
        cv2.putText(tile, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(tile, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        rows.append(tile)
    rows = rows[::len(rows) // 8] if len(rows) > 8 else rows        # keep the image small: 8 of the corners
    grid = np.concatenate([np.concatenate(rows[i:i + 2], 1) for i in range(0, len(rows), 2)], 0)
    os.makedirs(f'{SIM}/out/real_compare', exist_ok=True)
    cv2.imwrite(f'{SIM}/out/real_compare/ranges.png', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print('all corners OK' if ok_all else 'SOME CORNERS FAIL', '->', f'{SIM}/out/real_compare/ranges.png')


if __name__ == '__main__':
    main()
