#!/usr/bin/env python3
"""The owner's five top-down photos of the robot's white test table as scenes: config/scenes/table_photo_<n>.yaml.

    .venv/bin/python scripts/make_photo_scenes.py

Each object is (id, image x, image y, yaw or lying axis) read off the 848 x 480 head-camera photo: the table is
seen from the robot standing in front of it, so image right = world +x and image up = into the table (+y).
Scale from object sizes (Ace box 21 cm, water bottle 20.5 cm): ~1.27 mm/px; the table's near edge is at y ~350 px
and its centre column at x ~420 px. Lying bottles / cans / jars: `axis` is the image direction from the bottom to
the cap, the body origin (bottom centre) goes half a length behind the centre, axis horizontal at the radius.
Anything else lies on its usual face, turned by `yaw` (degrees, the mesh +x axis in image terms).
"""
import json, math, os, sys

import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
import house_scene as hs                                    # noqa: E402
import make_real_objects as mro                             # noqa: E402

MM = 0.00127
X0, Y_NEAR = 420, 350
TX, TY = hs.TEST_TABLE_AT
H = hs.TEST_TABLE[2]
NEAR = TY - hs.TEST_TABLE[1] / 2

# (id, px, py, kind, angle): kind 'yaw' = mesh +x along image angle, 'lie' = lying, cap along image angle,
# angles measured in the image with up = +90 deg (world +y), right = 0 (world +x)
PHOTOS = {
    1: [('r12_ace_box', 273, 95, 'yaw', 45), ('r21_black_cap', 378, 112, 'yaw', -30), ('r20_pine_box', 535, 165, 'yaw', -36),
        ('r08_juice_bottle', 324, 258, 'lie', 90), ('r10_brown_towel', 475, 270, 'yaw', -25)],
    2: [('r01_coke_can', 316, 135, 'lie', 52), ('r02_cider_can', 459, 140, 'lie', -52), ('r16_top_coffee', 385, 213, 'lie', 148),
        ('r09_water_bottle', 526, 203, 'lie', -157)],
    3: [('r23_wood_spoon', 300, 64, 'yaw', 35), ('r22_striped_ball', 375, 107, 'yaw', 0), ('r03_green_jar', 470, 146, 'lie', 92),
        ('r10_brown_towel', 296, 176, 'yaw', 15), ('017_orange', 530, 210, 'yaw', 0), ('r21_black_cap', 392, 290, 'yaw', -90),
        ('r24_wood_fork', 532, 285, 'yaw', 33)],
    4: [('011_banana', 397, 80, 'yaw', 40), ('r06_beige_towel', 300, 155, 'yaw', 5), ('r05_snack_pouch', 530, 116, 'yaw', -5),
        ('r22_striped_ball', 452, 182, 'yaw', 0), ('r13_phone', 431, 243, 'yaw', -40), ('017_orange', 348, 272, 'yaw', 0)],
    5: [('r14_remote', 343, 113, 'yaw', -56), ('r24_wood_fork', 268, 187, 'yaw', -83), ('013_apple', 428, 168, 'yaw', 0),
        ('r23_wood_spoon', 532, 112, 'yaw', 122), ('r04_gnc_bottle', 358, 258, 'lie', 131), ('r03_green_jar', 513, 246, 'lie', 135)],
}
YCB_NAMES = {'017_orange': 'orange', '013_apple': 'apple', '011_banana': 'banana'}


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def lying(oid):
    """(length, radius) of a body of revolution."""
    _, kind, size, *_ = mro.OBJECTS[oid]
    if kind == 'cyl':
        return size[1], size[0] / 2
    pts = size if kind == 'lathe' else [(0, size[0] / 2), (size[2], size[1] / 2)]
    return pts[-1][0], max(r for _, r in pts)


def main():
    os.makedirs(f'{SIM}/config/scenes', exist_ok=True)
    for n, items in PHOTOS.items():
        objs, used = [], {}
        for oid, px, py, kind, ang in items:
            name = mro.OBJECTS[oid][0] if oid in mro.OBJECTS else YCB_NAMES[oid]
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:
                name = f'{name} {used[name]}'
            x, y = TX + (px - X0) * MM, NEAR + (Y_NEAR - py) * MM
            a = math.radians(ang)
            if kind == 'lie':
                L, r = lying(oid)
                ox, oy = x - math.cos(a) * L / 2, y - math.sin(a) * L / 2
                q = qmul([math.cos(a / 2), 0, 0, math.sin(a / 2)], [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
                objs.append({'name': name, 'ycb': oid, 'at': [round(ox, 4), round(oy, 4), round(H + r + 0.003, 4)],
                             'quat': [round(v, 5) for v in q], 'bench': {'target': True, 'photo': n}})
            else:
                objs.append({'name': name, 'ycb': oid, 'at': [round(x, 4), round(y, 4)], 'yaw_deg': ang,
                             'bench': {'target': True, 'photo': n}})
        scene = {'photo': n, 'objects': objs}
        yaml.safe_dump(scene, open(f'{SIM}/config/scenes/table_photo_{n}.yaml', 'w'), sort_keys=False, allow_unicode=True)
        print(f'table_photo_{n}: ' + ', '.join(o['name'] for o in objs))


if __name__ == '__main__':
    main()
