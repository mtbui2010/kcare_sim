#!/usr/bin/env python3
"""Scenes for the visually-impaired demo (the owner's scenario sheet): config/scenes/vi_s<n>.yaml.

    .venv/bin/python scripts/make_vi_scenes.py

  s1 high-place info   kitchen tall-unit niche: mug + wine glass on its left half, tumbler + plate on its right half
  s2 ramen             the same niche: Shin Ramyun, Ansungtangmyun, Chapagetti
  s3 hot pot           the blue-handled pot on the left burner of the gas hob
  s4 dropped spoon     the wooden spoon on the floor beside the dining chair
  s5 clear the table   bowl, plate, mug, spoon and a cloth at the near end of the dining table
  s6 fall              a person lying in the hall in front of the restroom door
  s8 clothes / bags    white / black / checked shirts on a clothes rack in the living room; beige / blue / black bags on the chest
  s9 shoes             a black pair in the hall, apart and turned
Left / right are as seen by someone standing in front of the furniture. Offsets on a spot: [along the front edge
(+ = to the robot's left when it faces the furniture... see house_scene.objects_xml), into the surface from its
centre]; `at` = world position (z added from the surface below, or given).
"""
import os
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
O = lambda name, ycb, **k: dict(name=name, ycb=ycb, **k)
SHELF = 'shelf@kitchen'                    # niche floor 0.512; its front faces north, so offset u > 0 is west (= right
                                           # for a person facing the shelf)
SCENES = {
    'vi_s1': [O('mug', '025_mug', surface=SHELF, offset=[-0.18, -0.06], yaw_deg=150),
              O('wine glass', 'd05_wine_glass', surface=SHELF, offset=[-0.06, 0.06]),
              O('tumbler', 'd04_tumbler', surface=SHELF, offset=[0.09, -0.15]),
              O('plate', '029_plate', surface=SHELF, offset=[0.15, 0.1])],
    'vi_s2': [O('shin ramyun', 'd01_shin_ramyun', surface=SHELF, offset=[-0.17, 0.0], yaw_deg=90),
              O('ansungtangmyun', 'd02_ansung', surface=SHELF, offset=[0.0, 0.0], yaw_deg=90),
              O('chapagetti', 'd03_chapagetti', surface=SHELF, offset=[0.17, 0.0], yaw_deg=90)],
    'vi_s3': [O('pot', 'd06_pot', at=[-1.22, -3.65, 0.909], yaw_deg=0)],
    'vi_s4': [O('spoon', 'r23_wood_spoon', at=[0.93, -2.62], yaw_deg=35)],
    'vi_s5': [O('bowl', '024_bowl', surface='dining table@kitchen', offset=[0.12, -0.45]),
              O('plate', '029_plate', surface='dining table@kitchen', offset=[-0.14, -0.4]),
              O('mug', '025_mug', surface='dining table@kitchen', offset=[0.16, -0.22], yaw_deg=30),
              O('spoon', 'r23_wood_spoon', surface='dining table@kitchen', offset=[-0.05, -0.2], yaw_deg=90),
              O('cloth', 'r06_beige_towel', surface='dining table@kitchen', offset=[0.0, 0.05])],
    'vi_s6': [O('person', 'd15_person', at=[3.95, -0.12, 0.0], yaw_deg=0)],
    'vi_s8': [O('white shirt', 'd12_white_shirt', at=[0.56, 3.4, 1.585]),
              O('black shirt', 'd13_black_shirt', at=[0.9, 3.4, 1.585]),
              O('checked shirt', 'd14_check_shirt', at=[1.24, 3.4, 1.585]),
              O('beige bag', 'd07_beige_bag', surface='chest@bedroom', offset=[0.27, 0.0], yaw_deg=90),
              O('blue bag', 'd08_blue_bag', surface='chest@bedroom', offset=[0.0, 0.0], yaw_deg=90),
              O('black bag', 'd09_black_bag', surface='chest@bedroom', offset=[-0.27, 0.0], yaw_deg=90)],
    'vi_s9': [O('black shoe', 'd10_shoe_left', at=[4.95, 0.28], yaw_deg=75),
              O('black shoe 2', 'd11_shoe_right', at=[5.12, -0.08], yaw_deg=140)],
}


def main():
    os.makedirs(f'{SIM}/config/scenes', exist_ok=True)
    for name, objs in SCENES.items():
        scene = {'vi_demo': name, 'objects': objs}
        if name == 'vi_s8':                # a clothes rack in the living room (the dress room has no room to stand in)
            scene['add'] = [{'id': 'hanger', 'type': 'clothes rack', 'pos': [0.9, 3.4], 'yaw_deg': 0}]
        yaml.safe_dump(scene, open(f'{SIM}/config/scenes/{name}.yaml', 'w'),
                       sort_keys=False, allow_unicode=True)
        print(name, ', '.join(o['name'] for o in objs))


if __name__ == '__main__':
    main()
