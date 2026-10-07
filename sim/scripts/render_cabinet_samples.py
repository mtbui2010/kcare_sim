#!/usr/bin/env python3
"""Render every cabinet style in cabinet.STYLES next to the robot, for choosing one.

    python3 render_cabinet_samples.py [--open 0.12] [--head-tilt -0.6]
      -> /sim/mjcf/scene_drawer_<style>.xml, /sim/out/cabinets/samples.png

Each panel: 3/4 view of robot + cabinet (target drawer pulled out `--open` m so
it is obvious which one it is) with the head camera image inset, and the
target handle height printed.
"""
import argparse, os, sys
import mujoco, numpy as np, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
import cabinet, check_sim as cs

SIM = os.environ.get('SIM_ROOT', '/sim')


def scene_xml(style_key, variant='slamtec', pos=(0.80, 0.0, 0.0)):
    body, target, H = cabinet.cabinet_xml(cabinet.STYLES[style_key], pos=pos)
    xml = f'''<mujoco model="drawer_{style_key}">
  <include file="kaair_{variant}.xml"/>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
    <global offwidth="1920" offheight="1080"/><quality shadowsize="8192"/></visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1=".55 .55 .52" rgb2=".48 .48 .45" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1.5 3" dir="-0.4 0.4 -1" directional="true" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    {body}
  </worldbody>
</mujoco>'''
    return xml, target, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--open', type=float, default=0.12)
    ap.add_argument('--head-tilt', type=float, default=-0.6)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    out = f'{SIM}/out/cabinets'
    os.makedirs(out, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    except OSError:
        font = ImageFont.load_default()

    panels = []
    for key, st in cabinet.STYLES.items():
        xml, target, H = scene_xml(key)
        path = f'{SIM}/mjcf/scene_drawer_{key}.xml'
        open(path, 'w').write(xml)
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
        # arm folded to the side (SRDF left_home) so the head camera sees the cabinet
        q = dict(cfg['home'], joint1=-1.5708, head_joint2=a.head_tilt)
        cs.set_q(m, d, q, cfg)
        d.qpos[m.jnt_qposadr[m.joint(f'{target}_slide').id]] = a.open
        mujoco.mj_forward(m, d)
        # handle height = highest-z handle geom of the target drawer
        hz = max(d.geom_xpos[g][2] for g in range(m.ngeom)
                 if m.geom_bodyid[g] == m.body(target).id and ('handle' in m.geom(g).name or 'knob' in m.geom(g).name or 'lip' in m.geom(g).name))

        cam = mujoco.MjvCamera()
        # from the robot's front-right, looking at the drawer fronts (-x face)
        cam.lookat[:] = [0.45, 0.0, 0.55]
        cam.distance, cam.azimuth, cam.elevation = 2.2, 50, -18
        r = mujoco.Renderer(m, 540, 960); r.update_scene(d, cam); view = Image.fromarray(r.render()); r.close()
        r = mujoco.Renderer(m, 720, 1280); r.update_scene(d, 'head_rgb'); head = Image.fromarray(r.render()); r.close()
        head = head.resize((384, 216))
        dr = ImageDraw.Draw(view)
        view.paste(head, (960 - 384 - 8, 8))
        dr.rectangle([960 - 384 - 8, 8, 960 - 8, 8 + 216], outline=(255, 255, 255), width=2)
        dr.text((970 - 384, 12), 'head_rgb', fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        dr.text((12, 540 - 64), st['label'], fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        dr.text((12, 540 - 34), f"W {st['width']:.2f} × D {st['depth']:.2f} × H {H:.2f} m   handle z = {hz:.2f} m",
                fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        panels.append(view)
        print(f'{key}: H={H:.3f} handle z={hz:.3f}  -> {path}')

    grid = Image.new('RGB', (1920, 1080))
    for i, p in enumerate(panels):
        grid.paste(p, ((i % 2) * 960, (i // 2) * 540))
    grid.save(f'{out}/samples.png')
    print('wrote', f'{out}/samples.png')


if __name__ == '__main__':
    main()
