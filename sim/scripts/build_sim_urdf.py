#!/usr/bin/env python3
"""Turn a flattened kaair URDF (flatten_urdf.sh) into a physics-ready URDF.

    python3 build_sim_urdf.py slamtec            # -> /sim/urdf/kaair_slamtec_sim.urdf
    python3 build_sim_urdf.py clober --audit     # report only, write nothing

What changes, driven by /sim/config/sim_urdf.yaml:

  * every mesh is resolved from package://, baked to metres (the kaair STLs are
    in mm with scale="0.001") and copied to /sim/assets/<variant>/, so the
    output is self-contained and loads in MuJoCo / Isaac without ROS;
  * links without <inertial> get the configured mass, with COM and inertia
    taken from the convex hull of their visual geometry;
  * collision is rebuilt per link: kept, a single convex hull, a CoACD
    decomposition (fingers, gripper body) or dropped;
  * joint limit fixes (finger velocity), hardware-only tags stripped.

The kinematics (joint origins/axes/limits) are never touched, so the FK of the
output must match the input exactly — check_sim.py verifies that.
"""

import argparse
import os
import subprocess

import numpy as np
import trimesh
import yaml
from lxml import etree

SIM = os.environ.get('SIM_ROOT', '/sim')
_pkg_cache = {}


def pkg_share(pkg):
    if pkg not in _pkg_cache:
        prefix = subprocess.check_output(['ros2', 'pkg', 'prefix', pkg], text=True).strip()
        _pkg_cache[pkg] = os.path.join(prefix, 'share', pkg)
    return _pkg_cache[pkg]


def resolve(uri):
    if uri.startswith('package://'):
        pkg, rel = uri[len('package://'):].split('/', 1)
        return os.path.join(pkg_share(pkg), rel)
    if uri.startswith('file://'):
        return uri[len('file://'):]
    return uri


def origin_tf(el):
    """4x4 transform of an <origin> element (URDF rpy = fixed-axis XYZ)."""
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get('xyz', '0 0 0').split()]
    r, p, y = [float(v) for v in el.get('rpy', '0 0 0').split()]
    T[:3, :3] = trimesh.transformations.euler_matrix(r, p, y, 'sxyz')[:3, :3]
    T[:3, 3] = xyz
    return T


def geometry_mesh(geom):
    """trimesh of a <geometry> (mesh or primitive), in the element's own frame."""
    m = geom.find('mesh')
    if m is not None:
        path = resolve(m.get('filename'))
        mesh = trimesh.load(path, force='mesh', process=True)
        if m.get('scale'):
            mesh.apply_scale([float(v) for v in m.get('scale').split()])
        return mesh
    b = geom.find('box')
    if b is not None:
        return trimesh.creation.box([float(v) for v in b.get('size').split()])
    c = geom.find('cylinder')
    if c is not None:
        return trimesh.creation.cylinder(float(c.get('radius')), float(c.get('length')))
    s = geom.find('sphere')
    if s is not None:
        return trimesh.creation.icosphere(radius=float(s.get('radius')))
    return None


def link_mesh(link, tag):
    """All <tag> geometry of a link merged into one mesh in the link frame."""
    parts = []
    for el in link.findall(tag):
        g = el.find('geometry')
        mesh = geometry_mesh(g) if g is not None else None
        if mesh is None:
            continue
        mesh = mesh.copy()
        mesh.apply_transform(origin_tf(el.find('origin')))
        parts.append(mesh)
    return trimesh.util.concatenate(parts) if parts else None


def fmt(v):
    return ' '.join(f'{x:.6g}' for x in v)


def add_origin(parent, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    etree.SubElement(parent, 'origin', xyz=fmt(xyz), rpy=fmt(rpy))


def add_mesh_geom(parent_tag_el, path):
    g = etree.SubElement(parent_tag_el, 'geometry')
    etree.SubElement(g, 'mesh', filename=path)


def export_mesh(mesh, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + '.stl')
    mesh.export(path)
    return path


def inertial_from(mesh, mass):
    """(com, inertia 3x3 about com) of a solid with the mesh's convex hull shape."""
    hull = mesh.convex_hull
    if hull.volume <= 1e-12:  # flat/degenerate: fall back to the bounding box
        hull = trimesh.creation.box(mesh.extents + 1e-3, trimesh.transformations.translation_matrix(mesh.bounds.mean(0)))
    hull.density = mass / hull.volume
    return hull.center_mass, hull.moment_inertia


def coacd_parts(mesh, threshold, max_hulls):
    import coacd
    coacd.set_log_level('error')
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces),
                            threshold=threshold, max_convex_hull=max_hulls)
    return [trimesh.Trimesh(v, f) for v, f in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['slamtec', 'clober'])
    ap.add_argument('--audit', action='store_true', help='report only')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(SIM, 'config', 'sim_urdf.yaml')))
    src = os.path.join(SIM, 'urdf', 'raw', f'kaair_{args.variant}.urdf')
    tree = etree.parse(src, etree.XMLParser(remove_blank_text=True))
    robot = tree.getroot()
    asset_dir = os.path.join(SIM, 'assets', args.variant)

    report = {'no_inertial': [], 'mesh_collision': [], 'mimic': [], 'limits': [], 'estimates': []}

    for tag in cfg.get('strip_tags', []):
        for el in robot.findall(tag):
            robot.remove(el)

    masses = cfg.get('masses', {})
    coll_cfg = cfg.get('collision', {})
    for link in robot.findall('link'):
        name = link.get('name')
        visual = link_mesh(link, 'visual')
        has_inertial = link.find('inertial') is not None
        if not has_inertial:
            report['no_inertial'].append(name)
        if any(c.find('geometry/mesh') is not None for c in link.findall('collision')):
            report['mesh_collision'].append(name)
        if args.audit:
            continue

        # visual: bake to metres, copy next to the URDF
        for i, v in enumerate(link.findall('visual')):
            m = v.find('geometry/mesh')
            if m is None:
                continue
            mesh = geometry_mesh(v.find('geometry'))
            path = export_mesh(mesh, os.path.join(asset_dir, 'visual'), f'{name}_{i}')
            m.set('filename', os.path.relpath(path, os.path.join(SIM, 'urdf')))
            m.attrib.pop('scale', None)

        # collision
        mode = coll_cfg.get(name, {}).get('mode', cfg.get('collision_default', 'hull'))
        if mode != 'keep':
            for c in link.findall('collision'):
                link.remove(c)
            if mode in ('hull', 'coacd') and visual is not None:
                if mode == 'hull':
                    parts = [visual.convex_hull]
                else:
                    c = coll_cfg[name]
                    parts = coacd_parts(visual, c.get('threshold', 0.05), c.get('max_convex_hull', 8))
                for i, part in enumerate(parts):
                    path = export_mesh(part, os.path.join(asset_dir, 'collision'), f'{name}_{i}')
                    col = etree.SubElement(link, 'collision', name=f'{name}_col{i}')
                    add_origin(col)
                    add_mesh_geom(col, os.path.relpath(path, os.path.join(SIM, 'urdf')))

        # inertial
        if not has_inertial:
            entry = masses.get(name)
            mass = entry['mass'] if entry else cfg.get('frame_mass', 0.0)
            if entry and entry.get('source') == 'estimate':
                report['estimates'].append(f'{name}={mass}kg')
            if mass <= 0:
                continue
            geom = visual if visual is not None else link_mesh(link, 'collision')
            if geom is not None:
                com, I = inertial_from(geom, mass)
            else:  # geometry-less: a 1 cm cube
                com, I = np.zeros(3), np.eye(3) * mass * (0.01 ** 2) / 6
            inert = etree.SubElement(link, 'inertial')
            add_origin(inert, com)
            etree.SubElement(inert, 'mass', value=f'{mass:.6g}')
            etree.SubElement(inert, 'inertia', ixx=f'{I[0,0]:.6g}', ixy=f'{I[0,1]:.6g}', ixz=f'{I[0,2]:.6g}',
                             iyy=f'{I[1,1]:.6g}', iyz=f'{I[1,2]:.6g}', izz=f'{I[2,2]:.6g}')

    jfix = cfg.get('joints', {})
    for j in robot.findall('joint'):
        name, lim = j.get('name'), j.find('limit')
        if j.find('mimic') is not None:
            mm = j.find('mimic')
            report['mimic'].append(f"{name} = {mm.get('multiplier', 1)} × {mm.get('joint')}")
        if lim is not None and j.get('type') in ('revolute', 'prismatic'):
            report['limits'].append(f"{name}: [{lim.get('lower')}, {lim.get('upper')}] v={lim.get('velocity')} e={lim.get('effort')}")
        if not args.audit and name in jfix and lim is not None:
            for k, v in jfix[name].items():
                lim.set(k, str(v))

    print(f'== {args.variant}: {len(robot.findall("link"))} links, {len(robot.findall("joint"))} joints')
    print(f'links without <inertial> ({len(report["no_inertial"])}):', ', '.join(report['no_inertial']))
    print(f'links with mesh collision ({len(report["mesh_collision"])}):', ', '.join(report['mesh_collision']))
    print('mimic joints:', *report['mimic'], sep='\n  ')
    print('actuated joint limits:', *report['limits'], sep='\n  ')
    if args.audit:
        return
    print('masses still estimates:', ', '.join(report['estimates']))
    out = os.path.join(SIM, 'urdf', f'kaair_{args.variant}_sim.urdf')
    tree.write(out, pretty_print=True, xml_declaration=True, encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
