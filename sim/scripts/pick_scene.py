#!/usr/bin/env python3
"""PICK phase A ("reach to view") episodes in the KETI apartment (house_scene): real furniture, floors, walls and the
house lighting stay fixed; only the objects and the robot's exact stop vary.

Per worker the house MJCF is built once with a POOL of objects parked out of sight (x = 40 m, behind the walls);
the pool is refreshed every `pool_refresh` episodes. Per episode (sample) a surface spot of the house is chosen, the
robot is put at that spot's ENV stop (spot centre + outward normal * (depth/2 + standoff), as make_kcare_sim_location
does) with a small along / standoff / yaw jitter and the mode's heading, and the target + distractors from the pool
are teleported onto the spot surface (random position in the reachable band, random yaw, tall ones sometimes lying).

    rand = sample(rng, rcfg, task)            # spot, robot pose, objects + poses, view, retry
    m, d, info = episode_model(task, rand)     # cached house model + fresh MjData with the episode's objects placed

Bands (config rand_pick.yaml `bands`): table = the house's tabletop spots (base house), floor / low / high = worker B's
pick_floor_0 / pick_low_0 / pick_high_0 scenes (floor mat, coffee table 0.42 m, high shelf top board 1.31 m at the
old test-table spot). Modes: in 'left' the arm works toward the robot's -y, in 'right' toward +y (head pan -70 / +70),
so the heading runs along the front edge; in 'front' the robot faces the surface (worker A's reach sweep: front
reaches about twice as much as left / right at every height).
"""
import json, math, os, re

import numpy as np

SIM = os.environ.get('SIM_ROOT', '/sim')
YCB = '../assets/ycb'
PARK_X = 40.0                                 # pool objects wait here, on the floor outside the house

YCB_NAMES = {
    '002_master_chef_can': 'coffee can', '003_cracker_box': 'cracker box', '004_sugar_box': 'sugar box',
    '005_tomato_soup_can': 'soup can', '006_mustard_bottle': 'mustard bottle', '007_tuna_fish_can': 'tuna can',
    '008_pudding_box': 'pudding box', '009_gelatin_box': 'jello box', '010_potted_meat_can': 'spam can',
    '011_banana': 'banana', '012_strawberry': 'strawberry', '013_apple': 'apple', '014_lemon': 'lemon',
    '015_peach': 'peach', '016_pear': 'pear', '017_orange': 'orange', '018_plum': 'plum',
    '021_bleach_cleanser': 'bleach bottle', '022_windex_bottle': 'spray bottle', '024_bowl': 'bowl', '025_mug': 'mug',
    '026_sponge': 'sponge', '030_fork': 'fork', '031_spoon': 'spoon', '040_large_marker': 'marker',
    '043_phillips_screwdriver': 'screwdriver', '054_softball': 'softball', '055_baseball': 'baseball',
    '056_tennis_ball': 'tennis ball', '057_racquetball': 'blue ball', '061_foam_brick': 'foam brick',
    '065-a_cups': 'orange cup', '065-c_cups': 'green cup', '065-f_cups': 'purple cup', '077_rubiks_cube': 'rubiks cube',
    # 2026-10-12 extras (fetched for pick v2; names from the texture colours)
    '019_pitcher_base': 'blue pitcher', '029_plate': 'red plate', '033_spatula': 'spatula', '035_power_drill': 'drill',
    '036_wood_block': 'wood block', '037_scissors': 'scissors', '038_padlock': 'padlock', '042_adjustable_wrench': 'wrench',
    '044_flat_screwdriver': 'flat screwdriver', '048_hammer': 'hammer', '050_medium_clamp': 'clamp',
    '053_mini_soccer_ball': 'soccer ball', '058_golf_ball': 'golf ball', '065-b_cups': 'blue cup', '065-e_cups': 'red cup',
    '070-b_colored_wood_blocks': 'blue block',
}
REAL_PROPS = ('d01', 'd02', 'd03', 'd04', 'd05')   # demo props that are graspable (ramen packs, tumbler, wine glass)
SIDE = {'left': -1.0, 'right': 1.0}

_MESH, _HOUSE, _MODEL, _POOL = {}, {}, {}, {}


def _obj_path(k):
    return f'{SIM}/assets/ycb/{k}/google_16k/textured.obj'


def mesh_box(k):
    if k not in _MESH:
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for line in open(_obj_path(k)):
            if line.startswith('v '):
                v = np.array(line.split()[1:4], float)
                lo, hi = np.minimum(lo, v), np.maximum(hi, v)
        _MESH[k] = (lo, hi)
    return _MESH[k]


def obj_box(k, scale=1.0):
    """mesh_box with the object's per-pool scale (objects.scale in the config)."""
    lo, hi = mesh_box(k)
    return lo * scale, hi * scale


def _props(k):
    try:
        return json.load(open(f'{SIM}/assets/ycb/{k}/props.json'))
    except (OSError, ValueError):
        return {}


def catalogue():
    """{id: (name, real?)} — YCB pickables + the owner's real objects (r##) + the graspable demo props."""
    out = {k: (v, False) for k, v in YCB_NAMES.items() if os.path.exists(_obj_path(k))}
    try:
        made = json.load(open(f'{SIM}/assets/real_home/objects/objects.json'))
    except (OSError, ValueError):
        made = {}
    for k, v in made.items():
        if (k.startswith('r') or k[:3] in REAL_PROPS) and os.path.exists(_obj_path(k)) and not _props(k).get('static'):
            out[k] = (v, True)
    return out


def house(task=None):
    """(house xml without objects, spots) — built once per process from the base house (or task['house_scene']).
    task['hide_geoms']: geoms (by name) dropped from the xml while their spots stay — the floor band keeps
    pick_floor_*'s 'floor@living room' spot (ENV stop) without the rug, which is not in the real room."""
    hide = tuple((task or {}).get('hide_geoms') or ())
    key = ((task or {}).get('house_scene', ''), hide)
    if key not in _HOUSE:
        import house_scene
        sc = house_scene.load_scene(key[0]) if key[0] else {}
        sc = dict(sc, objects=[])
        house_scene.use_scene(sc)
        xml, spots, _ = house_scene.house_xml('kaair_slamtec_mobile.xml')
        for g in hide:
            xml, n = re.subn(rf'<geom name="{re.escape(g)}"[^>]*/>', '', xml)
            if not n:
                raise KeyError(f'hide_geoms: no geom {g!r} in scene {key[0]!r}')
        _HOUSE[key] = (xml, spots)
    return _HOUSE[key]


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def _refresh_pool(rng, cfg):
    """Every pool_refresh episodes: a new set of pool objects (~real_prob from the real pool), unique names."""
    cat = catalogue()
    ycb = sorted(k for k, v in cat.items() if not v[1])
    real = sorted(k for k, v in cat.items() if v[1])
    n = int(cfg['objects'].get('pool', 24))
    ids, names = [], set()
    while len(ids) < n:
        src = real if (real and rng.random() < float(cfg['objects'].get('real_prob', 0.4))) else ycb
        k = src[int(rng.integers(len(src)))]
        if cat[k][0] not in names:
            ids.append(k); names.add(cat[k][0])
    sc = cfg['objects'].get('scale')                     # e.g. [0.85, 1.15]: one size per object per pool
    scale = {k: (round(float(rng.uniform(*sc)), 3) if sc else 1.0) for k in ids}
    _POOL.update(ids=ids, cat=cat, scale=scale, left=int(cfg['objects'].get('pool_refresh', 10)),
                 version=_POOL.get('version', 0) + 1)


def _frame(spot, mode, standoff, off, yaw_jit):
    """Base pose for a spot. Side modes: heading along the front edge, the surface on the mode's side (left: the
    robot's -y). Front: facing the surface. `off` moves the stop along the front edge (along the heading in the side
    modes, sideways in front mode). Returns (xy, yaw, x_base, y_base) with axes as 2D world vectors."""
    fy = math.radians(spot['front_yaw_deg'])
    n = np.array([math.cos(fy), math.sin(fy)])                # outward normal (toward where the robot stands)
    if mode == 'front':
        xb = -n
    else:
        yb_ = -SIDE[mode] * n                                 # surface at SIDE * y_base
        xb = np.array([yb_[1], -yb_[0]])
    yb = np.array([-xb[1], xb[0]])
    edge = np.array([-n[1], n[0]])
    move = xb if mode != 'front' else yb                      # the stop slides along the edge
    p = np.array(spot['centre']) + n * (spot['depth'] / 2 + standoff) + move * off * float(np.sign(np.dot(move, edge)) or 1)
    return p, math.atan2(xb[1], xb[0]) + yaw_jit, xb, yb


def _pick(rng, weights):
    names = sorted(weights)
    w = np.array([weights[k] for k in names], float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def sample(rng, cfg, task):
    """One episode: band (table / floor / low / high), spot, mode, robot stop, objects on the surface, view, retry."""
    if not _POOL or _POOL['left'] <= 0:
        _refresh_pool(rng, cfg)
    _POOL['left'] -= 1
    band = _pick(rng, {k: v['weight'] for k, v in cfg['bands'].items()})
    bc = cfg['bands'][band]
    scene = bc.get('scene', '')
    hide = list(bc.get('hide_geoms', []))
    _, spots = house(dict(task, house_scene=scene, hide_geoms=hide))
    avail = {k: v for k, v in bc['spots'].items() if k in spots}
    if not avail:
        raise KeyError(f'band {band}: none of {list(bc["spots"])} in scene {scene!r}')
    key = _pick(rng, avail)
    spot = spots[key]
    mode = _pick(rng, bc['modes'])
    mc = dict(cfg['modes'][mode], **(bc.get('mode_overrides', {}).get(mode, {})))
    standoff = _u(rng, mc['standoff'])
    yaw_jit = math.radians(_u(rng, cfg['robot']['yaw_jitter_deg']))
    W, D, H = spot['width'], spot['depth'], spot['height']
    fy = math.radians(spot['front_yaw_deg'])
    nrm = np.array([math.cos(fy), math.sin(fy)])
    edge = np.array([-nrm[1], nrm[0]])
    front_c = np.array(spot['centre']) + nrm * D / 2
    lo_c, hi_c = mc['comfort']                                # target x (heading) the arm reaches without the base
    base_band = rng.random() < float(bc.get('base_band_prob', cfg['robot'].get('base_band_prob', 0.0)))
    _, _, xb0, yb0 = _frame(spot, mode, standoff, 0.0, 0.0)
    sg = float(np.sign(np.dot(xb0 if mode != 'front' else yb0, edge)) or 1)   # robot axis along the edge: +edge or -edge
    for _try in range(100):
        v_t = _u(rng, (-W / 2 + 0.05, W / 2 - 0.05))
        if mode == 'front':                                   # x = distance ahead: standoff + u
            x_rel = _u(rng, (hi_c + 0.05, hi_c + 0.2)) if base_band else _u(rng, mc['target_x'])
            u_t = x_rel - standoff
            y_rel = _u(rng, mc['target_y'])
            off = v_t - y_rel * sg                            # the stop's offset along the edge puts the target at y_rel
        else:
            band_x = mc['base_band'][int(rng.integers(len(mc['base_band'])))] if base_band else mc['target_x']
            x_rel = _u(rng, band_x)
            u_t = _u(rng, (0.04, min(D - 0.04, mc['max_in'])))
            off = v_t - x_rel * sg
        if 0.03 < u_t < D - 0.03 and abs(off) <= cfg['robot']['max_offset']:
            break
    else:
        return sample(rng, cfg, task)
    p, yaw, xb, yb = _frame(spot, mode, standoff, off, yaw_jit)

    def on_surface(u, v):
        return front_c - nrm * u + edge * v

    pool, cat = list(_POOL['ids']), _POOL['cat']
    n = int(rng.integers(cfg['objects']['n'][0], cfg['objects']['n'][1] + 1))
    chosen = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    objs, placed = [], []
    for k, oid in enumerate(chosen):
        lo, hi = obj_box(oid, _POOL['scale'].get(oid, 1.0))
        ext = hi - lo
        r = 0.5 * math.hypot(ext[0], ext[1])
        tall = ext[2] > 1.6 * max(ext[0], ext[1])
        lying = bool(tall and rng.random() < float(cfg['objects'].get('lying_prob', 0.3)))
        rr = 0.5 * math.hypot(ext[0], ext[2]) if lying else r
        h_obj = ext[1] if lying else ext[2]                   # lying: Rx(90) puts the mesh's y extent up
        narrow = min(ext[0], ext[2]) if lying else min(ext[0], ext[1])          # top-down grasp width
        smin = cfg['objects'].get('target_standing_min_h')        # previews: only standing targets this tall
        if k == 0 and smin is not None and (lying or ext[2] < float(smin)):
            return sample(rng, cfg, task)
        if k == 0 and not lying and ext[2] > float(cfg['objects'].get('target_max_h', 9.0)):
            return sample(rng, cfg, task)                     # v3: tall standing ones are grasped from the side, not here
        if k == 0 and narrow > float(cfg['objects'].get('target_max_width', 9.0)):
            return sample(rng, cfg, task)                     # v3: the target must fit between the fingers
        if h_obj > bc.get('max_obj_h', 9.0) or h_obj < bc.get('min_obj_h', 0.0):
            if k == 0:
                return sample(rng, cfg, task)                 # the target must fit the band (e.g. >= 5 cm near the floor)
            continue
        for _t in range(60):
            u, v = (u_t, v_t) if k == 0 else (_u(rng, (0.04, max(0.06, D - 0.04))), _u(rng, (-W / 2 + 0.04, W / 2 - 0.04)))
            if not (rr * 0.45 < u < D - rr * 0.45 and abs(v) < W / 2 - rr * 0.45):
                if k == 0:
                    break
                continue
            xy = on_surface(u, v)
            if all(np.linalg.norm(xy - q[:2]) > rr + q[2] + 0.02 for q in placed):
                placed.append(np.array([xy[0], xy[1], rr]))
                objs.append(dict(id=oid, name=cat[oid][0], real=cat[oid][1], x=float(xy[0]), y=float(xy[1]),
                                 yaw=_u(rng, (-math.pi, math.pi)), lying=lying, top=H,
                                 scale=_POOL['scale'].get(oid, 1.0)))
                break
        if k == 0 and not objs:
            return sample(rng, cfg, task)
    if not objs:
        return sample(rng, cfg, task)
    vc, hj = bc['view'], cfg['head_jitter_deg']
    phr = cfg.get('prompts', ['reach the {obj}'])
    tgt = objs[0]
    extra = {}
    if cfg.get('appearance'):                            # colours / lights a little different per episode
        extra['appearance'] = dict(cfg['appearance'], seed=int(rng.integers(2 ** 31)))
    if cfg.get('human'):                                 # human-like expert: speed, hesitation, noise, retries
        extra['human'] = _human(rng, cfg['human'])
    retry_offsets = None
    if cfg.get('retry_kinds'):                           # several kinds of miss (+ sometimes a second one)
        retry_offsets = [_retry_offset(rng, cfg)]
        if rng.random() < float(cfg.get('retry2_prob', 0.0)):
            retry_offsets.append(_retry_offset(rng, cfg))
    return dict(**extra, **({'retry_offsets': retry_offsets} if retry_offsets else {}),
                pool_scale=[_POOL['scale'].get(k, 1.0) for k in pool],band=band, scene=scene, hide_geoms=hide, spot=key, mode=mode, start_pose=bc.get('start_pose', {}).get(mode, 'give'),
                head_tilt=bc.get('head_tilt', -50), **({'head_pan': float(bc['head_pan'])} if 'head_pan' in bc else {}),
                lifts=bc.get('lifts', []), lifts_only=bool(bc.get('lifts_only', False)),
                via=bc.get('via', [0.12, 0.05]),
                comfort=[lo_c, hi_c], robot=dict(x=float(p[0]), y=float(p[1]), yaw=float(yaw)), base_band=bool(base_band),
                top_z=H, objects=objs, target=tgt['name'], pool_version=_POOL['version'], pool=pool,
                prompt=phr[int(rng.integers(len(phr)))].format(obj=tgt['name']),
                view=dict(dist=_u(rng, cfg['viewpoint']['dist']), elev=math.radians(_u(rng, vc['elev_deg'])),
                          yaw=math.radians(_u(rng, vc['yaw_deg']))),
                q_extra=dict(head_joint1=math.radians(_u(rng, hj['pan'])), head_joint2=math.radians(_u(rng, hj['tilt']))),
                retry=bool(rng.random() < float(cfg.get('retry_prob', 0.0))),
                retry_offset=[_u(rng, (-1, 1)) * cfg.get('retry_miss', 0.08), _u(rng, (-1, 1)) * cfg.get('retry_miss', 0.08)])


def _retry_offset(rng, cfg):
    """[sideways, vertical, along the view axis] [m] of a miss: kinds side / vertical / near (too close) / far."""
    kinds = cfg['retry_kinds']
    kind = _pick(rng, kinds)
    lo, hi = cfg.get('retry_mag', [0.04, 0.09])
    mag = _u(rng, (lo, hi)) * (1 if rng.random() < 0.5 else -1)
    small = lambda: _u(rng, (-0.3, 0.3)) * abs(mag)
    return dict(side=[mag, small(), 0.0], vertical=[small(), mag, 0.0], near=[small(), small(), abs(mag) * 0.6],
                far=[small(), small(), -abs(mag)])[kind]


def _human(rng, hc):
    """Per-episode human-like variation of the scripted expert (expert_pick_approach.run_episode)."""
    noisy = rng.random() < float(hc.get('noise_prob', 0.0))
    return dict(speed=_u(rng, hc.get('speed', [1.0, 1.0])),
                hesitate=_u(rng, hc.get('hesitate_s', [0.2, 0.6])) if rng.random() < float(hc.get('hesitate_prob', 0.0)) else 0.0,
                settle=_u(rng, hc.get('settle_s', [0.6, 0.6])),
                look=_u(rng, hc.get('look_s', [0.4, 0.4])),
                noise=dict(arm_deg=_u(rng, hc['noise_arm_deg']), lift=_u(rng, hc['noise_lift_m']),
                           base=_u(rng, hc['noise_base_m']), tau=_u(rng, hc['noise_tau_s']),
                           seed=int(rng.integers(2 ** 31))) if noisy else None,
                start_deg=_u(rng, (0.0, float(hc.get('start_arm_deg', 0.0)))),
                start_seed=int(rng.integers(2 ** 31)))


def apply_appearance(m, ent, app):
    """Colours / lights of this episode, from the model's original values (the model is shared across episodes):
    object materials (pmt_*) tinted by up to +-obj_rgb per channel, everything else by +-furniture_rgb, lights scaled
    by light_scale and tinted by +-light_tint."""
    if 'mat0' not in ent:
        ent['mat0'] = m.mat_rgba.copy()
        ent['light0'] = (m.light_diffuse.copy(), m.light_specular.copy(), m.light_ambient.copy())
    m.mat_rgba[:] = ent['mat0']
    m.light_diffuse[:], m.light_specular[:], m.light_ambient[:] = (a.copy() for a in ent['light0'])
    if not app:
        return
    rng = np.random.default_rng(app['seed'])
    for i in range(m.nmat):
        a = float(app.get('obj_rgb', 0.0)) if m.material(i).name.startswith('pmt_') else float(app.get('furniture_rgb', 0.0))
        if a:
            t = 1 + rng.uniform(-a, a) * 0.5 + rng.uniform(-a, a, 3) * 0.5      # brightness + chroma
            m.mat_rgba[i, :3] = np.clip(ent['mat0'][i, :3] * t, 0, 1)
    ls = rng.uniform(*app.get('light_scale', [1.0, 1.0]))
    lt = 1 + rng.uniform(-1, 1, 3) * float(app.get('light_tint', 0.0))
    m.light_diffuse[:] = ent['light0'][0] * ls * lt
    m.light_specular[:] = ent['light0'][1] * ls
    m.light_ambient[:] = ent['light0'][2] * ls * lt


def _pool_xml(house_xml, pool, cat, welds=False, scales=None):
    assets, bodies = [], []
    for i, oid in enumerate(pool):
        tag = re.sub(r'[^a-z0-9]', '_', oid.lower())
        s_ = float(scales[i]) if scales else 1.0
        lo, hi = obj_box(oid, s_)
        rgba = _props(oid).get('rgba')
        assets.append(f'<mesh name="pm_{tag}" file="{YCB}/{oid}/google_16k/textured.obj"' +
                      (f' scale="{s_} {s_} {s_}"' if s_ != 1.0 else '') + '/>'
                      f'<texture name="pt_{tag}" type="2d" file="{YCB}/{oid}/google_16k/texture_map.png"/>'
                      f'<material name="pmt_{tag}" texture="pt_{tag}"' + (f' rgba="{" ".join(str(v) for v in rgba)}"' if rgba else '') + '/>')
        bodies.append(f'<body name="obj_{tag}" pos="{PARK_X + 0.6 * i:.3f} 0 {-lo[2] + 0.002:.4f}">'
                      f'<freejoint name="obj_{tag}_free"/><geom name="obj_{tag}" type="mesh" mesh="pm_{tag}" '
                      f'material="pmt_{tag}" density="400" friction="1.0 0.01 0.001" condim="4"/></body>')
    xml = house_xml.replace('  </asset>', ''.join(assets) + '\n  </asset>', 1)
    xml = xml.replace('  </worldbody>', ''.join(bodies) + '\n  </worldbody>', 1)
    if welds:      # one inactive weld per pool object to the gripper (place episodes start holding one; house_scene idea)
        eq = '<equality>' + ''.join(
            f'<weld name="grasp_{re.sub(r"[^a-z0-9]", "_", oid.lower())}" body1="gripper_body_link" '
            f'body2="obj_{re.sub(r"[^a-z0-9]", "_", oid.lower())}" active="false" solref="0.004 1" solimp="0.99 0.999 0.001"/>'
            for oid in pool) + '</equality>'
        xml = xml.replace('</mujoco>', eq + '\n</mujoco>', 1) if '<equality>' not in xml else \
            xml.replace('<equality>', eq[:-len('</equality>')], 1)
    return xml


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def episode_model(task, rand, wid=0, welds=False):
    """(m, d, info): the cached house + pool model (rebuilt when the pool changed), a fresh MjData with the robot base at
    rand['robot'] and the episode's objects on the surface, the rest parked."""
    import mujoco
    key = (rand.get('scene', ''), tuple(rand.get('hide_geoms') or ()), bool(welds))
    ent = _MODEL.get(key)
    if ent is None or ent['version'] != rand['pool_version'] or ent.get('scales') != rand.get('pool_scale'):     # one model per scene (band), rebuilt with the pool
        hx, _ = house(dict(task, house_scene=key[0], hide_geoms=key[1]))
        xml = _pool_xml(hx, rand['pool'], _POOL.get('cat') or catalogue(), welds, rand.get('pool_scale'))
        # unique per process (+ worker, + weld variant): parallel runs (a big run + tests / eval) once shared
        # _pick_house_w1.xml and deleted it under each other; the file only lives while MuJoCo parses it
        path = f'{SIM}/mjcf/_pick_house_{os.getpid()}_w{wid}{"_weld" if welds else ""}.xml'
        try:
            open(path, 'w').write(xml)
            mm = mujoco.MjModel.from_xml_path(path)
            # znear / zfar are fractions of the model extent, and the pool parked at x = 40 m makes that ~65 m: the
            # near plane was 13 cm, so the wrist camera cut through an object it was grasping. Fixed metres instead.
            mm.vis.map.znear = 0.005 / mm.stat.extent
            mm.vis.map.zfar = 40.0 / mm.stat.extent
            ent = _MODEL[key] = dict(m=mm, version=rand['pool_version'], scales=rand.get('pool_scale'))
        finally:
            if os.path.exists(path):
                os.remove(path)
    m = ent['m']
    apply_appearance(m, ent, rand.get('appearance'))
    d = mujoco.MjData(m)
    for j, v in (('base_x', rand['robot']['x']), ('base_y', rand['robot']['y']), ('base_yaw', rand['robot']['yaw'])):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    info_obj = {}
    mujoco.mj_forward(m, d)                                  # objects parked, robot at its stop: rays see furniture only
    for o in rand['objects']:
        # the real surface under the object (a spot's height is nominal: the sofa's two seat cushions are 0.47 / 0.44 m)
        gid = np.array([-1], np.int32)
        dist = mujoco.mj_ray(m, d, np.array([o['x'], o['y'], o['top'] + 0.15]), np.array([0, 0, -1.0]), None, 1, -1, gid)
        if 0 <= dist < 0.25:
            o['top'] = round(o['top'] + 0.15 - float(dist), 4)
        tag = re.sub(r'[^a-z0-9]', '_', o['id'].lower())
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'obj_{tag}_free')
        a = m.jnt_qposadr[jid]
        lo, hi = obj_box(o['id'], o.get('scale', 1.0))
        qz = (math.cos(o['yaw'] / 2), 0, 0, math.sin(o['yaw'] / 2))
        if o['lying']:                                       # tipped over: 90 deg about x, then the yaw
            q = _qmul(qz, (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0))
            z = o['top'] - lo[1] + 0.004                      # Rx(90): z' = y, lowest point = lo[y]
        else:
            q = qz
            z = o['top'] - lo[2] + 0.002
        d.qpos[a:a + 7] = [o['x'], o['y'], z, *q]
        info_obj[o['name']] = dict(body=f'obj_{tag}', ycb=o['id'], height=float(hi[2] - lo[2]), lying=bool(o['lying']))
    mujoco.mj_forward(m, d)
    rb = rand['robot']
    view_dir = (1.0, 0.0, 0.0) if rand['mode'] == 'front' else (0.0, SIDE[rand['mode']], 0.0)
    tgt = next((o for o in rand['objects'] if o['name'] == rand['target']), None)
    return m, d, dict(top_z=tgt['top'] if tgt else rand['top_z'], side=SIDE.get(rand['mode'], 0.0), view_dir=view_dir, objects=info_obj,
                      target=rand['target'], robot=(rb['x'], rb['y'], rb['yaw']))
