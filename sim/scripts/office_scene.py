"""The room around the cabinet, dressed like the real one (drawer@living room, 2026-09-28).

    assets_xml, world_xml = office_xml(rand, cab_pos, cab_yaw, cab_depth, cab_width)

Textures are crops of the robot's own photos (sim/assets/real_textures/, cut
from the head camera — the wrist camera's colour cast is added by
camera_model.py, not baked in):
  wood_top               cabinet fronts + top and the desk top (same beech; cut from the head
                         view — a crop from the wrist view would bake in the D405 colour cast)
  gray_side              cabinet sides / back / plinth
  floor_wood             laminate floor
  office_bg              far background (windows, partitions) behind the cabinet — cut from the
                         wrist view, so its D405 saturation/WB are divided out first
Layout from the head image: the pedestal stands partly under the near end of a
desk that runs away from the robot, one desk leg beside the cabinet's right
side; the far wall carries the office photo.
`rand` (randomize.py) supplies texture tints and lighting; with
rand['textured'] False the cabinet falls back to plain random colours.
"""
import math

TEX_DIR = '../assets/real_textures'    # relative to /sim/mjcf, where the scene files are written


def _fmt(v):
    return ' '.join(f'{x:.4f}' for x in v)


def office_xml(rand, cab_pos, cab_yaw, cab_depth, cab_width):
    tint = rand.get('tex_tint', (1.0, 1.0, 1.0))
    # office light comes from the ceiling: tops bright, vertical fronts dimmer (as in the real head
    # image, front/top ≈ 0.77). Keep the camera-attached headlight weak for the same reason.
    li = rand.get('light', dict(dir=(-0.3, 0.3, -1), diffuse=0.55, ambient=0.32, head=0.12))
    t = lambda k: _fmt([min(1.0, c * k) for c in tint]) + ' 1'
    assets = f'''
    <texture name="tex_wood_top" type="2d" file="{TEX_DIR}/wood_top.png"/>
    <texture name="tex_gray_side" type="2d" file="{TEX_DIR}/gray_side.png"/>
    <texture name="tex_floor" type="2d" file="{TEX_DIR}/floor_wood.png"/>
    <texture name="tex_office" type="2d" file="{TEX_DIR}/office_bg.png"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".93 .94 .95" rgb2=".80 .82 .84" width="256" height="256"/>
    <material name="mat_wood_front" texture="tex_wood_top" texuniform="true" texrepeat="3 3" rgba="{t(1.0)}"/>
    <material name="mat_wood_top" texture="tex_wood_top" texuniform="true" texrepeat="3 3" rgba="{t(1.0)}"/>
    <material name="mat_gray_side" texture="tex_gray_side" texuniform="true" texrepeat="6 6" rgba="{t(1.0)}"/>
    <material name="mat_floor" texture="tex_floor" texuniform="true" texrepeat="0.35 0.35" rgba="{t(1.0)}"/>
    <material name="mat_office" texture="tex_office" rgba="{t(1.05)}"/>
    <material name="mat_desk_leg" rgba="0.62 0.64 0.67 1" specular="0.4"/>'''

    # desk + backdrop in the cabinet's frame (local -x = towards the robot, +x = away)
    c, s = math.cos(cab_yaw), math.sin(cab_yaw)
    W = lambda lx, ly, lz: (cab_pos[0] + c * lx - s * ly, cab_pos[1] + s * lx + c * ly, lz)
    quat = f'{math.cos(cab_yaw / 2):.6f} 0 0 {math.sin(cab_yaw / 2):.6f}'
    dl, dw, dh = 1.40, 0.70, 0.72                          # desk length (away from the robot), width, height
    # the desk's near end is at the cabinet's back (5 cm overlap),
    # its right leg just outside the cabinet's right side (head image)
    x0, y0 = cab_depth / 2 - 0.05, -cab_width / 2 - 0.09
    dc = (x0 + dl / 2, y0 + dw / 2)                        # desk centre in the cabinet frame
    legs = [(x0 + 0.05, y0 + 0.05), (x0 + dl - 0.05, y0 + 0.05),
            (x0 + 0.05, y0 + dw - 0.05), (x0 + dl - 0.05, y0 + dw - 0.05)]
    desk = f'<body name="desk" pos="{_fmt(W(dc[0], dc[1], 0))}" quat="{quat}">' \
           f'<geom name="desk_top" type="box" size="{dl / 2} {dw / 2} 0.0125" pos="0 0 {dh - 0.0125}" material="mat_wood_top"/>' + \
           ''.join(f'<geom name="desk_leg{i}" type="cylinder" size="0.025 {dh / 2}" pos="{lx - dc[0]:.4f} {ly - dc[1]:.4f} {dh / 2}" '
                   f'material="mat_desk_leg"/>' for i, (lx, ly) in enumerate(legs)) + '</body>'
    # far wall with the office photo, floor to 3.2 m, wide enough to fill both camera views
    backdrop = f'<body name="backdrop" pos="{_fmt(W(cab_depth / 2 + 3.0, 0.5, 1.6))}" quat="{quat}">' \
               f'<geom name="backdrop" type="box" size="0.02 5.0 1.6" material="mat_office" contype="0" conaffinity="0"/></body>'
    world = f'''
    <light name="ceiling1" pos="{_fmt(W(-0.5, 0, 2.6))}" dir="0 0 -1" diffuse="{_fmt([li['diffuse']] * 3)}" castshadow="false"/>
    <light name="ceiling2" pos="{_fmt(W(1.5, 1.0, 2.6))}" dir="0 0 -1" diffuse="{_fmt([li['diffuse'] * 0.6] * 3)}" castshadow="false"/>
    <light name="window" pos="{_fmt(W(3.0, 0, 1.8))}" dir="{_fmt(li['dir'])}" directional="true" diffuse="{_fmt([li['diffuse'] * 0.4] * 3)}" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="mat_floor"/>
    {desk}
    {backdrop}'''
    return assets, world, li
