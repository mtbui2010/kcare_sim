#!/usr/bin/env python3
"""Open-drawer scene at the kcare ready pose (moveh + lift + movej::approach_drawer).

    python3 drawer_ready.py [--mode left|right]
      -> /sim/mjcf/scene_open_drawer_<mode>.xml
         /sim/out/open_drawer/ready_<mode>_{overview,head,wrist}.png

The cabinet is placed from the ready TCP, not the other way round: its front
faces the gripper, the handle sits on the TCP's approach axis `standoff` m
ahead (+ kcare-style dforward/dshift base offsets), at whatever height style A
puts it (0.415 m, same as the real one).
The robot then holds the pose for 1 s under physics, so the images show the
settled state and any robot/cabinet contact is reported.
"""
import argparse, math, os, sys
import imageio, mujoco, numpy as np, yaml
sys.path.insert(0, os.path.dirname(__file__))
import cabinet, check_sim as cs

SIM = os.environ.get('SIM_ROOT', '/sim')


def ready_q(task, mode, home):
    rp = task['ready_pose']
    lift = float(np.clip(rp['lift']['inputs'], *rp['lift']['range']) - rp['lift']['mobile_height'])
    q = dict(home, lift_joint=lift, head_joint1=math.radians(rp['head']['rz'][mode]) + rp.get('head_pan_offset', 0.0),
             head_joint2=math.radians(rp['head']['ry']), virtual_gripper_joint=rp['gripper'])
    q.update({f'joint{i + 1}': math.radians(a) for i, a in enumerate(rp['arm'][mode])})
    return q


def build_scene(task, tcp_T, variant='slamtec', rand=None):
    """Scene XML with the cabinet placed from the ready TCP.

    `rand` (see randomize.py) perturbs the placement and the looks:
      yaw [rad], lateral / depth [m] offsets of the cabinet, and colours /
      lighting. None = the nominal scene.
    """
    rand = rand or {}
    st = dict(cabinet.STYLES[task['cabinet']])
    if not rand.get('textured', True):          # plain random colours instead of the real textures
        for k in ('body_material', 'front_material', 'top_material'):
            st.pop(k, None)
    if not st.get('front_material'):
        for k in ('body_rgba', 'front_rgba'):
            if k in rand:
                st[k] = rand[k]
    d = tcp_T[:3, 2].copy(); d[2] = 0; d /= np.linalg.norm(d)      # approach direction, horizontal
    side = np.array([-d[1], d[0], 0.0])                              # along the cabinet front
    yaw = math.atan2(d[1], d[0]) + rand.get('yaw', 0.0)              # cabinet local -x faces the robot
    face = np.array([math.cos(yaw), math.sin(yaw), 0.0])             # into the cabinet
    bar_from_centre = st['depth'] / 2 + cabinet.T + 0.035            # carcass centre -> handle bar
    pl = task['placement']
    handle = (tcp_T[:3, 3] + d * (pl['standoff'] + pl.get('dforward', 0.0) + rand.get('depth', 0.0))
              + side * (pl.get('dshift', 0.0) + rand.get('lateral', 0.0)))
    centre = handle + face * bar_from_centre
    body, target, H = cabinet.cabinet_xml(st, pos=(centre[0], centre[1], 0), yaw=yaw)
    if 'handle_rgba' in rand and task.get('scene') != 'office':
        body = body.replace(' '.join(f'{v:.3f}' for v in cabinet.HANDLE_RGBA),
                            ' '.join(f'{v:.3f}' for v in rand['handle_rgba']))
    fmt = lambda v: ' '.join(f'{x:.3f}' for x in v)
    if task.get('scene') == 'house':
        # the KETI apartment (house_scene.py) placed around the robot so its cabinet sits at (centre, yaw)
        import house_scene
        return house_scene.drawer_scene_xml(st, centre, yaw, rand), target, H, d
    if task.get('scene') == 'office':
        # the real room: textured cabinet/floor/desk/background (office_scene.py)
        import office_scene
        assets, world, li = office_scene.office_xml(rand, centre, yaw, st['depth'], st['width'])
        xml = f'''<mujoco model="open_drawer">
  <include file="kaair_{variant}.xml"/>
  <visual><headlight ambient="{fmt([li['ambient']] * 3)}" diffuse="{fmt([li['head']] * 3)}"/>
    <global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/>
    <map znear="0.002"/></visual>
  <asset>{assets}
  </asset>
  <worldbody>{world}
    {body}
  </worldbody>
</mujoco>'''
        return xml, target, H, d
    fl = rand.get('floor_rgb', ((.55, .55, .52), (.48, .48, .45)))
    li = rand.get('light', dict(dir=(-0.4, 0.4, -1), diffuse=0.6, ambient=0.35, head=0.5))
    xml = f'''<mujoco model="open_drawer">
  <include file="kaair_{variant}.xml"/>
  <visual><headlight ambient="{fmt([li['ambient']] * 3)}" diffuse="{fmt([li['head']] * 3)}"/>
    <global offwidth="1920" offheight="1080"/><quality shadowsize="8192"/></visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="{fmt(fl[0])}" rgb2="{fmt(fl[1])}" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8" reflectance="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".85 .87 .9" rgb2=".55 .58 .62" width="512" height="512"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1.5 3" dir="{fmt(li['dir'])}" directional="true" diffuse="{fmt([li['diffuse']] * 3)}" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    {body}
  </worldbody>
</mujoco>'''
    return xml, target, H, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['left', 'right'], default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
    mode = a.mode or task['mode']
    out = f'{SIM}/out/open_drawer'
    os.makedirs(out, exist_ok=True)

    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q = ready_q(task, mode, cfg['home'])
    tcp_T = ref.fk(q)['tool_tcp_link']
    xml, target, H, d_app = build_scene(task, tcp_T)
    path = f'{SIM}/mjcf/scene_open_drawer_{mode}.xml'
    open(path, 'w').write(xml)

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    cs.set_q(m, d, q, cfg)
    d.ctrl[:] = cs.ctrl_from(m, q, cfg)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    drift = max(abs(cs.get_q(m, d, j) - q[j]) for j in q if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0)
    contacts = set()
    for c in d.contact[:d.ncon]:
        b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        if b1 != 'world' or b2 != 'world':
            contacts.add(tuple(sorted((b1, b2))))
    tcp = d.body('tool_tcp_link').xpos
    hid = m.geom(f'{target}_handle').id
    handle = d.geom_xpos[hid]
    print(f'mode={mode}  lift_joint={q["lift_joint"]:.3f}  head pan/tilt={q["head_joint1"]:.3f}/{q["head_joint2"]:.3f} rad')
    print(f'TCP {np.round(tcp, 3)}  approach dir {np.round(d_app, 2)}')
    print(f'handle {np.round(handle, 3)}  -> gap along approach {np.dot(handle - tcp, d_app):.3f} m, '
          f'height diff {handle[2] - tcp[2]:+.3f} m, cabinet H {H:.3f} m')
    print(f'held 1 s: max joint drift {drift:.1e} rad;  contacts: {sorted(contacts) or "none"}')

    # renders: overview (behind-side of the robot, looking at the cabinet), head, wrist
    look = (tcp + handle) / 2
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [look[0] - 0.1, look[1], 0.55]
    side = -1 if d_app[1] < 0 else 1
    cam.azimuth = math.degrees(math.atan2(d_app[1], d_app[0])) + 50 * side
    cam.distance, cam.elevation = 2.3, -22
    for name, camera, (w, h) in [('overview', cam, (1920, 1080)),
                                 ('head', 'head_rgb', (1280, 720)),
                                 ('wrist', 'wrist_rgb', (848, 480))]:
        r = mujoco.Renderer(m, h, w)
        r.update_scene(d, camera)
        imageio.imwrite(f'{out}/ready_{mode}_{name}.png', r.render())
        r.close()
    print('images ->', out)


if __name__ == '__main__':
    main()
