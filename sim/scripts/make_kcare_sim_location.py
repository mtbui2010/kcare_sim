#!/usr/bin/env python3
"""Create kcare_robot's `sim_house` site profile in the sim's own config dir (/sim/kcare_sim_configs).

The code runs read-only from the dev PC (/kcare_mnt/remote_dir, see kcare_sim_app.py); the real site
`kcare` is read from there, so connection changes on the robot side carry over at the next run of this script.

    python3 scripts/make_kcare_sim_location.py [--standoff 0.55]

  connections.json            the real site's ROS connections unchanged (house_bridge.py serves the same
                              names), VisionServe / Ollama pointed at the A6000 (127.0.0.1), the robot-LAN-only
                              entries dropped (vlms TCP on the dev PC, the external /drawer/open VLA action)
  skill_configs_override.json the real site's groups, with ENV / HOME_LOC replaced by the house
                              (config/house_layout.yaml spots)

ENV entries use kcare's 'left' mode, where the arm and head work toward the base's -y side: `move`
drives to `loc` facing `rz` = toward the furniture, then turns +90 deg, which puts the furniture on -y. MOBILE dforward/dshift are cancelled per entry
so the robot stops exactly at `loc`, `standoff` metres (base centre) in front of the surface's front edge.
"""
import argparse, json, math, os, re, shutil

import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
ROBOT_LOC = os.environ.get('KCARE_CODE', '/kcare_mnt/remote_dir') + '/kcare_robot/kcare_robot/configs/locations'
LOC = f'{SIM}/kcare_sim_configs/locations'
DROP = {'vlms'}                 # vla_drawer_open (/drawer/open) is served in the sim by vla_drawer_node.py
HOST = '127.0.0.1'


# Korean / Vietnamese names of the house's places (ENV `aliases`): what a user says, ko / vi. A spot's own
# `aliases` (house_scene) come first. No two places may share one (PUT /skill-configs/ENV refuses clashes).
SPOKEN = {
    'sofa@living room': ['소파', 'ghế sofa'],
    'test table@living room': ['거실 테이블', 'bàn phòng khách'],
    'desk@study room': ['책상', 'bàn làm việc'],
    'drawer@kitchen': ['부엌 서랍', 'ngăn kéo bếp'],
    'counter@kitchen': ['조리대', 'bàn bếp'],
    'shelf@kitchen': ['부엌 선반', 'kệ bếp'],
    'dining table@kitchen': ['식탁', 'bàn ăn'],
    'washer@laundry room': ['세탁기', 'máy giặt'],
    'chest@bedroom': ['서랍장', 'tủ ngăn kéo'],
    'stool@bedroom': ['스툴', 'ghế đẩu'],
    'shelf@restroom': ['화장실 선반', 'kệ nhà tắm'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--standoff', type=float, default=0.55)
    a = ap.parse_args()
    src, dst = f'{ROBOT_LOC}/kcare', f'{LOC}/sim_house'
    os.makedirs(dst, exist_ok=True)

    conns = []
    for c in json.load(open(f'{src}/connections.json')):
        if c['name'] in DROP:
            continue
        if c['type'] in ('visionserve', 'llm'):
            cfg = c['config']
            txt = cfg if isinstance(cfg, str) else json.dumps(cfg)
            # whatever host the robot's site points at (it moves between LANs),
            # the sim uses the VisionServe / Ollama servers on this machine; an unreachable host would block
            # kcare's device loading (the connections after it, e.g. vla_drawer_open, never come up)
            txt = re.sub(r'(https?://)[^/:"\']+(:\d+)', lambda m: m.group(1) + HOST + m.group(2), txt)
            c = dict(c, config=txt if isinstance(cfg, str) else json.loads(txt))
        conns.append(c)
    # the sim's follow camera, so the dashboard's camera grid can show the robot in the house
    arm_rgb = next(c for c in conns if c['name'] == 'arm_rgb')
    conns.append(dict(arm_rgb, name='sim_overview', config=dict(
        arm_rgb['config'], conn_name='/sim/overview/image_raw/compressed', agent_name='sim_overview')))
    conns.append(dict(arm_rgb, name='sim_topdown', config=dict(       # fixed camera above the whole house
        arm_rgb['config'], conn_name='/sim/topdown/image_raw/compressed', agent_name='sim_topdown')))
    # the simulation's own control (house_bridge.py /sim/command): reset, move objects, scenes — the Map tab's Sim bar
    conns.append({'type': 'ros_service', 'name': 'sim_command', 'config': {
        'conn_name': '/sim/command', 'conn_type': 'ros_service', 'is_client': True, 'agent_name': 'sim_command',
        'data_interface': 'sim_msgs/srv/SimCommand',
        'encode_func': "def encode_func(data, req):\n    import json\n    req.request = json.dumps({k: v for k, v in data.items() if k != 'wait'})\n    return req",
        'decode_func': "def decode_func(res):\n    import json\n    out = json.loads(res.response or '{}')\n    out['isdone'] = bool(out.get('ok'))\n    return out"}})
    json.dump(conns, open(f'{dst}/connections.json', 'w'), indent=2, ensure_ascii=False)

    ov = json.load(open(f'{src}/skill_configs_override.json'))
    lay = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))
    mob = ov['MOBILE_CONFIGS']
    env = {}
    for key, sp in lay['spots'].items():
        if 'robot_pose' in sp:      # a spot that needs an exact robot pose (the VLA open-drawer cabinet)
            rp = sp['robot_pose']
            x, y, rz = rp['x'], rp['y'], (rp['heading_deg'] - 90 + 180) % 360 - 180   # left mode turns +90 at the end
        else:
            phi = math.radians(sp['front_yaw_deg'])
            n = (math.cos(phi), math.sin(phi))
            d = sp['depth'] / 2 + a.standoff
            x, y = sp['centre'][0] + n[0] * d, sp['centre'][1] + n[1] * d
            rz = (sp['front_yaw_deg'] + 360) % 360 - 180
        e = {'default_mode': sp['default_mode'], 'height': sp['height'], 'placepose': {'dx': 0, 'dy': 0},
             'loc': {'x': round(x, 3), 'y': round(y, 3), 'rz': round(rz, 1)},
             'dforward': -mob['dforward'], 'dshift': -mob['dshift']}
        for k in ('handle', 'handle_height', 'aliases'):
            if k in sp:
                e[k] = sp[k]
        if key in SPOKEN:
            e['aliases'] = list(dict.fromkeys(list(e.get('aliases') or []) + SPOKEN[key]))
        env[key] = e
    # places saved from the dashboard's Map tab ("source": "map") survive a regeneration
    if os.path.exists(f'{dst}/skill_configs_override.json'):
        old = json.load(open(f'{dst}/skill_configs_override.json')).get('ENV', {})
        env.update({k: v for k, v in old.items() if isinstance(v, dict) and v.get('source') == 'map' and k not in env})
    ov['ENV'] = env
    ov['HOME_LOC'] = {'x': lay['start']['x'], 'y': lay['start']['y'], 'rz': lay['start']['yaw_deg']}
    ov['MAP'] = map_config(lay, dst)
    if os.path.exists(f'{dst}/skill_configs_override.json'):     # the user's origin (Map tab) survives too
        old_map = json.load(open(f'{dst}/skill_configs_override.json')).get('MAP') or {}
        if 'origin' in old_map:
            ov['MAP']['origin'] = old_map['origin']
    json.dump(ov, open(f'{dst}/skill_configs_override.json', 'w'), indent=2, ensure_ascii=False)
    for f in ('.env',):
        if os.path.exists(f'{src}/{f}'):
            shutil.copy(f'{src}/{f}', f'{dst}/{f}')
    os.makedirs(f'{LOC}/../common', exist_ok=True)
    open(f'{LOC}/../common/active_location', 'w').write('sim_house\n')
    print(f'-> {dst}: {len(conns)} connections, {len(env)} ENV locations; active_location = sim_house')
    for k, e in env.items():
        print(f"   {k:28s} loc ({e['loc']['x']:6.2f}, {e['loc']['y']:6.2f}) rz {e['loc']['rz']:6.1f}  height {e['height']}")


def map_config(lay, dst):
    """The Map tab's config (robotapp MapPanel): picture + layers in house_scene's frames, the skills it calls.
    Layer files go to <site>/map/ and are served by robot_agent's GET /config/locations/<site>/files/<path>
    (occupancy.png is written there by house_bridge.py when it starts)."""
    import house_scene
    b = lay['bounds']
    bounds = (*b['x'], *b['y'])
    os.makedirs(f'{dst}/map', exist_ok=True)
    hm = f'{SIM}/out/house/heightmap.json'
    layers = {'occupancy': {'file': 'map/occupancy.png', 'frame': house_scene.occupancy_frame(bounds),
                            'values': {'free': 0, 'inflated': 128, 'occupied': 255}}}
    if os.path.exists(hm):                                   # written by house_scene.py
        shutil.copy(f'{SIM}/out/house/heightmap.png', f'{dst}/map/heightmap.png')
        layers['height'] = {'file': 'map/heightmap.png', 'frame': json.load(open(hm)), 'unit': 'cm'}
    surfaces = [{'name': k, 'centre': sp['centre'], 'size': [sp['depth'], sp['width']],
                 'yaw_deg': sp['front_yaw_deg'], 'height': sp['height']} for k, sp in lay['spots'].items()]
    return {
        'image': {'connection': 'sim_topdown', 'frame': house_scene.map_frame(bounds)},
        'layers': layers,
        'surfaces': surfaces,
        'places': 'ENV',
        # read-only (GET /agent/<name>/get): polling must not clear a pending cancel the way POST /skill does
        'pose': {'agent': 'mobile_pose', 'x': 'pose.position.x', 'y': 'pose.position.y',
                 'orientation': 'pose.orientation', 'period_s': 0.5},
        'goal': {'skill': 'moveb', 'params': {'x': '{x}', 'y': '{y}'}},           # no rz: keep the heading
        # kcare: `turn` is relative (/mobile/rotate, ignores < 5 deg); `rotate` goes to an ABSOLUTE heading
        'rotate': {'skill': 'turn', 'params': {'inputs': '{deg}'}, 'step_deg': 15},
        'lift': {'skill': 'lift', 'params': {'inputs': '{h}'}, 'clearance': 0.10, 'reach_m': 0.8},
        'confirm_goal': False,
        'sim': {'agent': 'sim_command'},                     # simulation only: reset / objects / scenes in the Map tab
    }


if __name__ == '__main__':
    main()
