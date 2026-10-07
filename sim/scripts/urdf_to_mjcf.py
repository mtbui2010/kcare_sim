#!/usr/bin/env python3
"""Compile kaair_<variant>_sim.urdf into MuJoCo MJCF.

    python3 urdf_to_mjcf.py slamtec
      -> /sim/mjcf/kaair_slamtec.xml         robot only (actuators, cameras, gripper coupling)
      -> /sim/mjcf/scene_tabletop_slamtec.xml robot + floor + table + cube

MuJoCo's URDF importer gives the bodies/joints/geoms; this script adds what a
URDF cannot say, all from /sim/config/mjcf.yaml:

  * position actuators on the joints the real robot position-controls through
    its forward controllers (joint1..7, lift_joint, head_joint1/2,
    virtual_gripper_joint), so an 11-D ctrl vector is the sim twin of
    /arm/xarm7_forward_controller + /body/*_forward_controller;
  * finger coupling  finger = 0.5 * virtual_gripper_joint  (tool.urdf.xacro
    <mimic>) as stiff equality constraints;
  * gravity compensation on the robot bodies (the xArm controller does its own);
  * RGB cameras on femto_color_optical_frame (head) and
    camera_color_optical_frame (D405, wrist). ROS optical frames look down +z
    with y down; MuJoCo cameras look down -z with y up, hence quat="0 1 0 0";
  * collision hulls moved to geom group 3 (simulated, not rendered);
  * a "home" keyframe = kaair_moveit_config/config/initial_positions.yaml.
"""

import argparse
import os
import tempfile

import sys

import mujoco
import yaml
from lxml import etree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIM = os.environ.get('SIM_ROOT', '/sim')


def sub(parent, tag, **attrs):
    return etree.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def compile_urdf(urdf_path):
    """Load the URDF with MuJoCo-specific compiler flags and return MJCF text."""
    tree = etree.parse(urdf_path)
    robot = tree.getroot()
    mj = etree.SubElement(robot, 'mujoco')
    sub(mj, 'compiler', meshdir=os.path.dirname(urdf_path), strippath='false',
        fusestatic='false', discardvisual='false', balanceinertia='true', angle='radian')
    with tempfile.NamedTemporaryFile('wb', suffix='.urdf', dir=os.path.dirname(urdf_path), delete=False) as f:
        tree.write(f)
        tmp = f.name
    try:
        mujoco.MjModel.from_xml_path(tmp)
        out = tmp + '.xml'
        mujoco.mj_saveLastXML(out, mujoco.MjModel.from_xml_path(tmp))
        text = open(out).read()
        os.remove(out)
    finally:
        os.remove(tmp)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['slamtec', 'clober'])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(SIM, 'config', 'mjcf.yaml')))

    urdf = os.path.join(SIM, 'urdf', f'kaair_{args.variant}_sim.urdf')
    root = etree.fromstring(compile_urdf(urdf).encode())
    root.set('model', f'kaair_{args.variant}')

    # mesh paths: make them absolute-free (relative to /sim/mjcf)
    comp = root.find('compiler')
    comp.set('meshdir', os.path.relpath(os.path.dirname(urdf), os.path.join(SIM, 'mjcf')))

    opt = root.find('option') if root.find('option') is not None else sub(root, 'option')
    for k, v in cfg['option'].items():
        opt.set(k, str(v))

    bodies = {b.get('name'): b for b in root.iter('body')}
    joints = {j.get('name'): j for j in root.iter('joint')}

    for b in bodies.values():
        b.set('gravcomp', '1')

    # collision geoms (convex hulls) go to group 3: they collide but are not
    # drawn. Left in group 0 they are rendered over the visual meshes — the
    # D405 hull sits 2 cm in front of its own lens and blanks the wrist camera.
    for geom in root.iter('geom'):
        if geom.get('contype') != '0':
            geom.set('group', '3')

    for jname, jc in cfg.get('joint_dynamics', {}).items():
        for k, v in jc.items():
            joints[jname].set(k, str(v))

    # actuators
    act = root.find('actuator') if root.find('actuator') is not None else sub(root, 'actuator')
    for a in cfg['actuators']:
        j = joints[a['joint']]
        lo, hi = j.get('range').split()
        sub(act, 'position', name=a['name'], joint=a['joint'], kp=a['kp'],
            dampratio=a.get('dampratio', 1.0), ctrlrange=f'{lo} {hi}',
            forcerange=f"{-a['force']} {a['force']}")

    # gripper coupling. MuJoCo >= 3.3 turns <mimic> into these itself; older
    # importers drop it, so add whichever is missing and stiffen both.
    eq = root.find('equality') if root.find('equality') is not None else sub(root, 'equality')
    g = cfg['gripper']
    have = {e.get('joint1'): e for e in eq.findall('joint')}
    for finger in g['fingers']:
        e = have.get(finger)  # (childless lxml elements are falsy: no `or` here)
        if e is None:
            e = sub(eq, 'joint', joint1=finger, joint2=g['driver'], polycoef=f"0 {g['ratio']} 0 0 0")
        e.set('solref', '0.005 1')

    # geom pairs that interpenetrate by design (nested lift column, pan-tilt
    # brackets): their convex hulls overlap even at rest
    contact = root.find('contact') if root.find('contact') is not None else sub(root, 'contact')
    for b1, b2 in cfg.get('contact_exclude', []):
        sub(contact, 'exclude', body1=b1, body2=b2)

    # finger pads: friction for grasping
    for fname in g['fingers']:
        for geom in joints[fname].getparent().findall('geom'):
            geom.set('friction', g['friction'])
            geom.set('condim', '4')

    # cameras: real pinhole intrinsics from config/camera_real.yaml (camera_info of the robot)
    # when available, else the datasheet fovy
    real_path = os.path.join(SIM, 'config', 'camera_real.yaml')
    real = yaml.safe_load(open(real_path)) if os.path.exists(real_path) else {}
    import camera_model
    for cam in cfg['cameras']:
        if cam.get('real') in real:
            attrs = camera_model.mjcf_intrinsics(real[cam['real']])
        else:
            attrs = dict(fovy=cam['fovy'], resolution=f"{cam['width']} {cam['height']}")
        sub(bodies[cam['body']], 'camera', name=cam['name'], mode='fixed', quat='0 1 0 0', **attrs)

    os.makedirs(os.path.join(SIM, 'mjcf'), exist_ok=True)
    robot_xml = os.path.join(SIM, 'mjcf', f'kaair_{args.variant}.xml')
    etree.ElementTree(root).write(robot_xml, pretty_print=True)

    # home keyframe (qpos in model joint order) = initial_positions.yaml
    home = cfg['home']
    m = mujoco.MjModel.from_xml_path(robot_xml)
    qpos = []
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos.append(g['ratio'] * home[g['driver']] if name in g['fingers'] else home.get(name, 0.0))
    ctrl = [home.get(a['joint'], 0.0) for a in cfg['actuators']]
    kf = sub(root, 'keyframe')
    sub(kf, 'key', name='home', qpos=' '.join(f'{v:.6g}' for v in qpos), ctrl=' '.join(f'{v:.6g}' for v in ctrl))
    etree.ElementTree(root).write(robot_xml, pretty_print=True)

    # tabletop scene
    scene = etree.Element('mujoco', model=f'kaair_{args.variant}_tabletop')
    sub(scene, 'include', file=os.path.basename(robot_xml))
    s = cfg['scene']
    vis = sub(scene, 'visual')
    sub(vis, 'headlight', ambient='0.4 0.4 0.4', diffuse='0.6 0.6 0.6')
    sub(vis, 'global', offwidth=1280, offheight=720)
    asset = sub(scene, 'asset')
    sub(asset, 'texture', name='grid', type='2d', builtin='checker', rgb1='.2 .3 .4', rgb2='.3 .4 .5', width=512, height=512)
    sub(asset, 'material', name='grid', texture='grid', texrepeat='4 4', reflectance='0.1')
    wb = sub(scene, 'worldbody')
    sub(wb, 'light', pos='0 0 3', dir='0 0 -1', directional='true')
    sub(wb, 'geom', name='floor', type='plane', size='0 0 0.05', material='grid')
    t = s['table']
    sub(wb, 'geom', name='table', type='box', pos=' '.join(map(str, t['pos'])),
        size=' '.join(map(str, t['half_size'])), rgba='0.6 0.45 0.3 1', friction='1 0.005 0.0001')
    c = s['cube']
    cb = sub(wb, 'body', name='cube', pos=' '.join(map(str, c['pos'])))
    sub(cb, 'freejoint', name='cube_free')
    sub(cb, 'geom', name='cube', type='box', size=' '.join([str(c['half'])] * 3), mass=c['mass'],
        rgba='0.9 0.2 0.2 1', friction='1 0.005 0.0001', condim='4')
    scene_xml = os.path.join(SIM, 'mjcf', f'scene_tabletop_{args.variant}.xml')
    etree.ElementTree(scene).write(scene_xml, pretty_print=True)

    m = mujoco.MjModel.from_xml_path(scene_xml)
    print(f'wrote {robot_xml}\n      {scene_xml}')
    print(f'nbody={m.nbody} njnt={m.njnt} nu={m.nu} ngeom={m.ngeom} ncam={m.ncam} neq={m.neq} '
          f'robot mass={sum(m.body_mass[:]) - c["mass"]:.2f} kg')


if __name__ == '__main__':
    main()
