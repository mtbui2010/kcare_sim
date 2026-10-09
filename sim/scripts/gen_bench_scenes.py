#!/usr/bin/env python3
"""Random scenes for the skill bench (find / move / pick / place): config/scenes/bench_<seed>.yaml.

    .venv/bin/python scripts/gen_bench_scenes.py [--n 12] [--seed 100]

Each scene puts 2-4 YCB objects (from the bench pool, sim/assets/ycb) on every test surface, at random
yaw, spaced by their footprints. Positions are drawn around where the robot stands at that surface (the
ENV loc projected on the front edge): 70 % within 0.25 m of the front edge (what kcare is expected to
reach), 30 % anywhere on the surface (to measure where reach ends). Every other scene also moves the
dining table / stool a little and may drop the dining chair, so ENV places and the occupancy map change.
Object names are what a user would say (they are the find / pick inputs).
"""
import argparse, json, math, os, random

import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
# ycb id -> (name the user says, pickable target?)
POOL = {
    '002_master_chef_can': ('coffee can', True), '003_cracker_box': ('cracker box', True),
    '004_sugar_box': ('sugar box', True), '005_tomato_soup_can': ('soup can', True),
    '006_mustard_bottle': ('mustard bottle', True), '007_tuna_fish_can': ('tuna can', True),
    '008_pudding_box': ('pudding box', True), '009_gelatin_box': ('jello box', True),
    '010_potted_meat_can': ('spam can', True), '011_banana': ('banana', True), '012_strawberry': ('strawberry', True),
    '013_apple': ('apple', True), '014_lemon': ('lemon', True), '015_peach': ('peach', True), '016_pear': ('pear', True),
    '017_orange': ('orange', True), '018_plum': ('plum', True), '019_pitcher_base': ('pitcher', False),
    '021_bleach_cleanser': ('bleach bottle', True), '022_windex_bottle': ('spray bottle', True),
    '024_bowl': ('bowl', True), '025_mug': ('mug', True), '026_sponge': ('sponge', True), '029_plate': ('plate', False),
    '030_fork': ('fork', True), '031_spoon': ('spoon', True), '040_large_marker': ('marker', True),
    '043_phillips_screwdriver': ('screwdriver', True), '053_mini_soccer_ball': ('soccer ball', True),
    '054_softball': ('softball', True), '055_baseball': ('baseball', True), '056_tennis_ball': ('tennis ball', True),
    '057_racquetball': ('blue ball', True), '061_foam_brick': ('foam brick', True), '065-a_cups': ('orange cup', True),
    '065-c_cups': ('green cup', True), '065-f_cups': ('purple cup', True), '077_rubiks_cube': ('rubiks cube', True),
}
def real_pool():
    """The owner's table objects (make_real_objects.py) + the YCB fruit / cutlery that are in the same photos."""
    try:
        made = json.load(open(f'{SIM}/assets/real_home/objects/objects.json'))
    except OSError:
        made = {}
    pool = {k: (v, True) for k, v in made.items() if k.startswith('r')}        # d##: the demo props, not bench objects
    pool.update({k: POOL[k] for k in ('013_apple', '017_orange', '011_banana')})        # cutlery: the wooden r23 / r24
    return pool


DENSE = {'dining table@kitchen', 'desk@study room', 'counter@kitchen', 'test table@living room'}        # cluttered tables, as in the photos
SURFACES = ['desk@study room', 'drawer@kitchen', 'counter@kitchen', 'shelf@kitchen', 'dining table@kitchen',
            'washer@laundry room', 'chest@bedroom', 'stool@bedroom', 'shelf@restroom']


def footprint(ycb):
    """Half-diagonal of the mesh's xy box (m)."""
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for line in open(f'{SIM}/assets/ycb/{ycb}/google_16k/textured.obj'):
        if line.startswith('v '):
            v = np.array(line.split()[1:4], float); lo, hi = np.minimum(lo, v), np.maximum(hi, v)
    return 0.5 * math.hypot(hi[0] - lo[0], hi[1] - lo[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--pool', choices=['ycb', 'real'], default='ycb', help="real: the owner's photo objects")
    ap.add_argument('--prefix', default='bench')
    ap.add_argument('--max-from-front', type=float, default=0.15, help='objects at most this far in from the front edge (m)')
    ap.add_argument('--deep', action='store_true', help='also 30 %% anywhere on the surface (the old reach test)')
    a = ap.parse_args()
    global POOL
    if a.pool == 'real':
        POOL = real_pool()
    lay = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))
    env = json.load(open(f'{SIM}/kcare_sim_configs/locations/sim_house/skill_configs_override.json'))['ENV']
    pool = [k for k in POOL if os.path.exists(f'{SIM}/assets/ycb/{k}/google_16k/textured.obj')]
    rad = {k: footprint(k) for k in pool}
    os.makedirs(f'{SIM}/config/scenes', exist_ok=True)
    for i in range(a.n):
        seed = a.seed + i
        rng = random.Random(seed)
        scene = {'bench': {'seed': seed}, 'objects': []}
        if i % 2 == 1:                                         # furniture variation
            fur = {'dining_table': {'pos': [round(1.18 + rng.uniform(-0.15, 0.15), 3), round(-2.2 + rng.uniform(-0.1, 0.1), 3)],
                                    'yaw_deg': round(180 + rng.uniform(-10, 10), 1)},
                   'stool': {'pos': [round(-4.25 + rng.uniform(-0.05, 0.1), 3), round(0.95 + rng.uniform(-0.1, 0.1), 3)],
                             'yaw_deg': round(rng.uniform(-20, 20), 1)}}
            if rng.random() < 0.5:
                fur['dchair0'] = {'remove': True}
            scene['furniture'] = fur
        names = set()
        for surf in SURFACES + (['test table@living room'] if 'test table@living room' in lay['spots'] else []):
            if surf not in lay['spots'] or surf not in env:
                continue
            sp, loc = lay['spots'][surf], env[surf]['loc']
            fy = math.radians(sp['front_yaw_deg'])
            along = np.array([-math.sin(fy), math.cos(fy)])
            u0 = float(np.dot(np.array([loc['x'], loc['y']]) - np.array(sp['centre']), along))
            n = rng.randint(1, 3) if sp['width'] * sp['depth'] < 0.15 else rng.randint(2, 4)
            if a.pool == 'real':
                n = rng.randint(5, 8) if surf in DENSE else rng.randint(1, 2)
            placed = []
            for _ in range(n):
                for _try in range(80):
                    k = rng.choice(pool)
                    if POOL[k][0] in names:
                        continue
                    r = rad[k]
                    if 2 * r > min(sp['width'], sp['depth']) - 0.02:
                        continue
                    hw, d = sp['width'] / 2 - r - 0.01, sp['depth']
                    u = float(np.clip(u0 + rng.uniform(-0.3, 0.3), -hw, hw)) if hw > 0 else 0.0
                    # near the front edge, where the robot reaches from its standing spot (real homes: put
                    # things within reach; re-positioning the base costs too much time). --deep keeps the old 30 %
                    # anywhere-on-the-surface draws to measure where reach ends.
                    near = rng.random() < (0.7 if a.deep else 1.0)
                    wf = rng.uniform(r + 0.02, min(a.max_from_front, d - r - 0.01)) if near else rng.uniform(r + 0.02, d - r - 0.01)
                    if wf <= 0 or any(math.hypot(u - pu, wf - pw) < r + pr + 0.02 for pu, pw, pr in placed):
                        continue
                    placed.append((u, wf, r)); names.add(POOL[k][0])
                    scene['objects'].append({'name': POOL[k][0], 'ycb': k, 'surface': surf,
                                             'offset': [round(u, 3), round(wf - d / 2, 3)],
                                             'yaw_deg': round(rng.uniform(-180, 180), 1),
                                             'bench': {'from_front': round(wf, 3), 'target': POOL[k][1]}})
                    break
        yaml.safe_dump(scene, open(f'{SIM}/config/scenes/{a.prefix}_{seed}.yaml', 'w'), sort_keys=False, allow_unicode=True)
        print(f'{a.prefix}_{seed}: {len(scene["objects"])} objects', 'furniture moved' if 'furniture' in scene else '')


if __name__ == '__main__':
    main()
