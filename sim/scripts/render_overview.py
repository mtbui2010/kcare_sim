#!/usr/bin/env python3
"""One global view of the sim scene, for eyeballing the model.

    python3 render_overview.py slamtec [--head-tilt -0.6]
      -> /sim/out/<variant>/overview.png

Robot at the check_sim grasp pose (gripper down over the cube on the table),
gripper closed on the cube; main 3/4 view with the head and wrist camera
images inset, so model, scene and both cameras are checked in one picture.
"""
import argparse, json, os, sys
import mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
import check_sim as cs

SIM = os.environ.get('SIM_ROOT', '/sim')


def shot(m, d, cam, w, h):
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, cam)
    img = r.render()
    r.close()
    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['slamtec', 'clober'])
    ap.add_argument('--head-tilt', type=float, default=-0.6,
                    help='head_joint2 [rad]; its axis is (0,-1,0), so negative looks down')
    a = ap.parse_args()
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    g = json.load(open(f'{SIM}/out/{a.variant}/report.json'))['grasp']
    m = mujoco.MjModel.from_xml_path(f'{SIM}/mjcf/scene_tabletop_{a.variant}.xml')
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    m.vis.quality.shadowsize = 8192
    d = mujoco.MjData(m)

    q = dict(cfg['home'], lift_joint=0.35, **g['arm_q'], head_joint2=a.head_tilt)
    q['virtual_gripper_joint'] = 0.1
    cs.set_q(m, d, q, cfg)
    mujoco.mj_forward(m, d)
    tcp = np.array(g['tcp'])
    top = min(cs.body_min_z(m, d, f) for f in ('left_finger_tip', 'right_finger_tip')) - 0.002
    tid = m.geom('table').id
    m.geom_pos[tid] = [tcp[0], tcp[1], top - m.geom_size[tid][2]]
    adr = m.jnt_qposadr[m.joint('cube_free').id]
    d.qpos[adr:adr + 7] = [tcp[0], tcp[1], top + cfg['scene']['cube']['half'] + 1e-4, 1, 0, 0, 0]
    d.ctrl[:] = cs.ctrl_from(m, q, cfg)
    for _ in range(500):
        mujoco.mj_step(m, d)
    q['virtual_gripper_joint'] = 0.0          # close on the cube
    d.ctrl[:] = cs.ctrl_from(m, q, cfg)
    for _ in range(1000):
        mujoco.mj_step(m, d)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.35, 0.0, 0.75]
    cam.distance, cam.azimuth, cam.elevation = 2.4, 135, -18
    main_img = shot(m, d, cam, 1920, 1080)
    head = shot(m, d, 'head_rgb', 640, 360)
    wrist = shot(m, d, 'wrist_rgb', 640, 360)

    canvas = main_img.copy()
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
    except OSError:
        font = ImageFont.load_default()
    for i, (img, label) in enumerate([(head, 'head_rgb  (Femto Bolt)'), (wrist, 'wrist_rgb  (D405)')]):
        x, y = 1920 - 660, 80 + i * 390
        dr.rectangle([x - 4, y - 4, x + 644, y + 364], fill=(255, 255, 255))
        canvas.paste(img, (x, y))
        dr.text((x + 10, y + 8), label, fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    width = d.joint('virtual_gripper_joint').qpos[0]
    dr.text((24, 20), f'kaair_{a.variant}  |  MuJoCo {mujoco.__version__}  |  lift {q["lift_joint"]:.2f} m, '
            f'head tilt {a.head_tilt:.2f} rad, gripper closed on 4 cm cube (width {width*1000:.0f} mm)',
            fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    out = f'{SIM}/out/{a.variant}/overview.png'
    canvas.save(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
