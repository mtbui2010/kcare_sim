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
        env[key] = e
    ov['ENV'] = env
    ov['HOME_LOC'] = {'x': lay['start']['x'], 'y': lay['start']['y'], 'rz': lay['start']['yaw_deg']}
    json.dump(ov, open(f'{dst}/skill_configs_override.json', 'w'), indent=2, ensure_ascii=False)
    for f in ('.env',):
        if os.path.exists(f'{src}/{f}'):
            shutil.copy(f'{src}/{f}', f'{dst}/{f}')
    os.makedirs(f'{LOC}/../common', exist_ok=True)
    open(f'{LOC}/../common/active_location', 'w').write('sim_house\n')
    print(f'-> {dst}: {len(conns)} connections, {len(env)} ENV locations; active_location = sim_house')
    for k, e in env.items():
        print(f"   {k:28s} loc ({e['loc']['x']:6.2f}, {e['loc']['y']:6.2f}) rz {e['loc']['rz']:6.1f}  height {e['height']}")


if __name__ == '__main__':
    main()
