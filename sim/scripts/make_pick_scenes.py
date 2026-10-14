#!/usr/bin/env python3
"""Scenes for pick / place tests at three heights (VLA pick plan, step 2):

    pick_floor_0..5   objects on the floor (rug patch 'floor@living room', height 0)
    pick_low_0..5     a low coffee table (0.42 m, 'coffee table@living room')
    pick_high_0..5    an open shelf, top board 1.30 m ('high shelf@living room')

Each scene removes the white test table and puts its piece in the same open spot of the living room, front to the
south (where the test table's ENV loc already navigates), with default_mode front; 3-5 YCB objects 6-16 cm from the
front edge, at least 12 cm apart.

    python3 scripts/make_pick_scenes.py [--seed 0]  ->  config/scenes/pick_*.yaml
"""
import argparse, math, os
import numpy as np, yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
AT, YAW = (0.4, 2.0), 0.0
BANDS = {
    'floor': dict(piece='floor mat', id='floor', width=0.7, depth=0.5, w=(0.08, 0.16), objs=[
        ('apple', '013_apple'), ('orange', '017_orange'), ('tennis ball', '056_tennis_ball'), ('sponge', '026_sponge'),
        ('mug', '025_mug'), ('tomato soup can', '005_tomato_soup_can'), ('foam brick', '061_foam_brick'),
        ('rubiks cube', '077_rubiks_cube'), ('banana', '011_banana'), ('lemon', '014_lemon'), ('baseball', '055_baseball'),
        ('tuna can', '007_tuna_fish_can'), ('coke can', 'r01_coke_can'), ('grey cup', 'r07_grey_cup'), ('remote control', 'r14_remote')]),
    'low': dict(piece='coffee table', id='coffee table', width=0.9, depth=0.5, w=(0.07, 0.15), objs=[
        ('apple', '013_apple'), ('orange', '017_orange'), ('mug', '025_mug'), ('bowl', '024_bowl'), ('red cup', '065-a_cups'),
        ('sponge', '026_sponge'), ('pudding box', '008_pudding_box'), ('tennis ball', '056_tennis_ball'), ('banana', '011_banana'),
        ('phone', 'r13_phone'), ('yellow cup', 'r11_yellow_cup'), ('coke can', 'r01_coke_can'), ('peach', '015_peach'),
        ('potted meat can', '010_potted_meat_can'), ('remote control', 'r14_remote')]),
    'high': dict(piece='high shelf', id='high shelf', width=0.76, depth=0.3, w=(0.06, 0.11), objs=[
        ('sponge', '026_sponge'), ('pudding box', '008_pudding_box'), ('gelatin box', '009_gelatin_box'), ('mug', '025_mug'),
        ('coke can', 'r01_coke_can'), ('red cup', '065-a_cups'), ('tomato soup can', '005_tomato_soup_can'),
        ('rubiks cube', '077_rubiks_cube'), ('tuna can', '007_tuna_fish_can'), ('small water bottle', 'r19_small_water'),
        ('foam brick', '061_foam_brick'), ('top coffee', 'r16_top_coffee'), ('orange', '017_orange'), ('apple', '013_apple')]),
}


def scene(band, k, rng):
    b = BANDS[band]
    n = int(rng.integers(3, 6))
    picks = rng.choice(len(b['objs']), n, replace=False)
    half = b['width'] / 2 - 0.08
    objs, pts = [], []
    for i in picks:
        for _ in range(200):
            u, w = float(rng.uniform(-half, half)), float(rng.uniform(*b['w']))
            if all(math.hypot(u - pu, w - pw) >= 0.12 for pu, pw in pts):
                break
        pts.append((u, w))
        name, ycb = b['objs'][i]
        # house_scene offsets are from the surface CENTRE (u along the front edge, + = into the surface):
        # w here is the distance from the front edge, so the object sits w in from the edge
        objs.append(dict(name=name, ycb=ycb, surface=f"{b['id']}@living room", offset=[round(u, 3), round(w - b['depth'] / 2, 3)],
                         yaw_deg=int(rng.integers(0, 360))))
    return {'furniture': {'test_table': {'remove': True}},
            'add': [dict(id=b['id'], type=b['piece'], pos=list(AT), yaw_deg=YAW, mode='front')],
            'objects': objs}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--n', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(f'{SIM}/config/scenes', exist_ok=True)
    for band in BANDS:
        for k in range(a.n):
            rng = np.random.default_rng(a.seed * 1000 + list(BANDS).index(band) * 100 + k)
            f = f'{SIM}/config/scenes/pick_{band}_{k}.yaml'
            yaml.safe_dump(scene(band, k, rng), open(f, 'w'), sort_keys=False, allow_unicode=True)
            print(f)


if __name__ == '__main__':
    main()
