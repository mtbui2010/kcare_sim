"""Render every r## object on a table (offline) -> out/real_objects_preview.jpg"""
import glob, json, os, sys
import cv2, mujoco, numpy as np
SIM = '/sim'
ids = sorted(os.path.basename(os.path.dirname(os.path.dirname(p))) for p in glob.glob(f'{SIM}/assets/ycb/r*_*/google_16k/textured.obj'))
ids += ['013_apple', '017_orange', '011_banana', '030_fork', '031_spoon']
assets, bodies = [], []
for i, k in enumerate(ids):
    d = f'{SIM}/assets/ycb/{k}/google_16k'
    try:
        rgba = json.load(open(f'{SIM}/assets/ycb/{k}/props.json')).get('rgba')
    except OSError:
        rgba = None
    rg = f' rgba="{" ".join(map(str, rgba))}"' if rgba else ''
    assets.append(f'<mesh name="m{i}" file="{d}/textured.obj"/><texture name="t{i}" type="2d" file="{d}/texture_map.png"/><material name="mt{i}" texture="t{i}"{rg}/>')
    x, y = -0.6 + 0.2 * (i % 7), -0.3 + 0.22 * (i // 7)
    bodies.append(f'<body pos="{x} {y} 0.75"><freejoint/><geom type="mesh" mesh="m{i}" material="mt{i}" density="400"/></body>')
xml = f'''<mujoco><visual><global offwidth="1600" offheight="900"/><headlight ambient=".5 .5 .5" diffuse=".5 .5 .5"/></visual>
<asset>{"".join(assets)}</asset><worldbody><light pos="0 -1 3" dir="0 0.3 -1"/>
<geom type="plane" size="3 3 .1" rgba=".85 .85 .85 1"/><geom type="box" pos="0 0 0.37" size="0.8 0.5 0.37" rgba=".93 .92 .9 1"/>
{"".join(bodies)}<camera name="c" pos="0 -1.25 1.45" xyaxes="1 0 0 0 0.6 0.8"/></worldbody></mujoco>'''
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
for _ in range(1500): mujoco.mj_step(m, d)
r = mujoco.Renderer(m, 900, 1600); r.update_scene(d, 'c')
img = r.render()
for i, k in enumerate(ids):
    pass
cv2.imwrite(f'{SIM}/out/real_objects_preview.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(len(ids), 'objects;', 'fell over:', [k for i, k in enumerate(ids) if d.xquat[i + 1][0] < 0.9 and not k.startswith('0')])
