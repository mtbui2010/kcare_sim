"""Sample one episode's randomization from config/randomization.yaml.

    rand = sample(rng, cfg)   # -> dict for drawer_ready.build_scene + expert extras
"""
import math

import numpy as np


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def _clip(c):
    return tuple(float(x) for x in np.clip(c, 0.0, 1.0))


def kcare_lift_joint(x, kc):
    """kcare `lift::x` -> lift_joint [m] (skills/lift.py: clip to range, minus mobile_height)."""
    return float(np.clip(x, *kc['range']) - kc['mobile_height'])


def sample(rng, cfg):
    pl = cfg['placement']
    base = np.array(cfg['cabinet_palettes'][rng.choice(list(cfg['cabinet_palettes']))])
    body = base + rng.uniform(-cfg['color_jitter'], cfg['color_jitter'], 3)
    front = body + _u(rng, cfg['front_delta'])
    g = _u(rng, cfg['handle_gray'])
    floor = cfg['floor_palettes'][rng.choice(list(cfg['floor_palettes']))]
    lc = cfg['light']
    az, el = math.radians(_u(rng, lc['azimuth_deg'])), math.radians(_u(rng, lc['elevation_deg']))
    hj = cfg['head_jitter_deg']
    # draw order is part of the contract: the fixed evaluation scenes are seeds, so the
    # original fields must keep consuming the rng in the original order; new fields come last
    rand = dict(
        yaw=math.radians(_u(rng, pl['yaw_deg'])),
        lateral=_u(rng, pl['lateral']),
        depth=_u(rng, pl['depth']),
        body_rgba=_clip(body) + (1.0,),
        front_rgba=_clip(front) + (1.0,),
        handle_rgba=(g, g, g * 1.02, 1.0),
        floor_rgb=[_clip(c) for c in floor],
        light=dict(dir=(-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)),
                   diffuse=_u(rng, lc['diffuse']), ambient=_u(rng, lc['ambient']), head=_u(rng, lc['headlight'])),
        drawer0=_u(rng, cfg['drawer0']),
        q_extra=dict(head_joint1=math.radians(_u(rng, hj['pan'])), head_joint2=math.radians(_u(rng, hj['tilt']))),
    )
    if 'lift_inputs' in cfg:                                  # lift::X, clipped like the robot does
        kc = cfg['lift_kcare']
        x = _u(rng, cfg['lift_inputs'])
        rand['q_extra']['lift_joint'] = kcare_lift_joint(x, kc) - kcare_lift_joint(kc['nominal'], kc)
        rand['lift_input'] = x
    if 'textured_prob' in cfg:                                # real photo textures vs plain colours
        rand['textured'] = bool(rng.uniform() < cfg['textured_prob'])
        k = _u(rng, cfg['tex_tint'])
        rand['tex_tint'] = tuple(float(k * v) for v in rng.uniform(0.97, 1.03, 3))
    if 'drawer_targets' in cfg:                               # which drawer + the user's phrasing of it
        dt = cfg['drawer_targets']
        ids = sorted(dt['weights'])
        w = np.array([dt['weights'][i] for i in ids], float)
        target = int(ids[rng.choice(len(ids), p=w / w.sum())])
        phrasings = dt['prompts'][target]
        rand['target'] = target
        rand['prompt'] = phrasings[int(rng.integers(len(phrasings)))]
    if 'house_sites' in cfg:                                  # where the cabinet stands in the house (house_scene.CAB_SITES)
        names = sorted(cfg['house_sites'])
        w = np.array([cfg['house_sites'][k] for k in names], float)
        rand['site'] = names[int(rng.choice(len(names), p=w / w.sum()))]
    return rand
