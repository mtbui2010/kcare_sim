#!/usr/bin/env python3
"""The KETI demo apartment for the mobile kcare robot.

    python3 scripts/house_scene.py            -> /sim/mjcf/house.xml (+ kaair_slamtec_mobile.xml)
                                                 /sim/config/house_layout.yaml (rooms, furniture, spots, objects)
    python3 scripts/house_scene.py --render   also /sim/out/house/{top,living,kitchen,bedroom,restroom}.png,
                                                 photo_<room>.jpg (owner's photo | sim) and compare_real.jpg

Floor plan: docs/draws/floor.drawio (owner, 2026-10-07), 1 drawio unit = 2 cm, the apartment is 10 x 8 m.
World frame = the robot's `map` frame (x east, y north, metres), origin in the hall between the living room
and the kitchen (= drawio (420, -386)); the robot starts there.

    y  4 +-----------+--------------+------------------+
         |  BEDROOM  | sofa  LIVING TV|  STUDY  desk     |
         |  bed      | desk         |                  |
         |  chest    |  side table  |                  |
     0.6 |           +--------------+--door-------------+
         |                 HALL (open)                   |
    -0.6 +-door--+---+--------------+--+----+---door-----+
         | DRESS |WARE-|drawer      fridge| C  | RESTROOM  |
         | ROOM  |HOUSE door   KITCHEN    | O  | tub,      |
         |closets+-----+       dining tbl | R  | basin, WC |
         |       |LAUND door              | R  |           |
      -4 +-------+-----+ shelf/stove/sink-+----+-----------+
        -4.5    -3   -1.76               1.96 2.76        5.5  x
                                           (entrance below the corridor)

Furniture looks follow the owner's photos (sim/assets/real_home/photos, 2026-10-06) and the robot's own head /
wrist images (sim/assets/real_home/robot_cams). The robot body `base_footprint` is a mocap body: house_bridge.py
drives the base kinematically (x, y, yaw) like the real nav stack; everything above it (lift, head, arm,
gripper) is simulated as in the drawer scenes. Small objects are real scanned meshes (YCB, sim/assets/ycb) so
the open-vocabulary detector used by kcare's `find` sees realistic things.
"""
import argparse, math, os, re, sys

import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import cabinet  # noqa: E402

WALL_H, WALL_T, CEIL = 2.45, 0.1, 2.45
TEX = '../assets/real_textures'          # relative to /sim/mjcf
HOME_TEX = '../assets/real_home'         # textures cut from the home-tour video
YCB = '../assets/ycb'                    # meshes: relative to meshdir /sim/urdf; textures: to /sim/mjcf -> same place

# open-drawer cabinet relative to the robot base at the ready pose (x, y [m], yaw [rad]) — the training geometry
CAB_REL = (0.5882, -0.6896, math.radians(-91.92))
# where the cabinet stands in the house (x, y, cabinet.py yaw): two sites, training draws one per episode
#   nw     : drawio 'Drawer' — north-west corner of the kitchen, back on the warehouse wall, front facing east
#   fridge : where the robot's own images show it (2026-10-06, sim/assets/real_home/robot_cams) — drawio's
#            'Trash bin' spot, back on the fridge's north side, front facing north; the recycling tote east of it
CAB_SITES = {'nw': (-1.425, -0.82, math.pi), 'fridge': (1.25, -0.615, -math.pi / 2)}
TOTE_AT = {'nw': (1.7, -0.7), 'fridge': (1.68, -0.55)}
CAB_SITE = os.environ.get('HOUSE_CAB_SITE', 'nw')        # the live sim (house.xml / kcare ENV)
CAB_POSE = CAB_SITES[CAB_SITE]


def _robot_at(cab):
    h = cab[2] - CAB_REL[2]
    c, s_ = math.cos(h), math.sin(h)
    return (cab[0] - (c * CAB_REL[0] - s_ * CAB_REL[1]), cab[1] - (s_ * CAB_REL[0] + c * CAB_REL[1]), h)


ROBOT_AT_DRAWER = _robot_at(CAB_POSE)             # where the base must stand for the policy's view of the cabinet

WOOD_FRONT, WOOD_TOP = 0.58, 0.8                 # albedo scale of the wood texture in the house (see house_xml)
WHITE = (0.93, 0.93, 0.91, 1)
OFFWHITE = (0.96, 0.96, 0.94, 1)
CREAM = (0.90, 0.87, 0.80, 1)
BLACK = (0.08, 0.08, 0.09, 1)
CHROME = (0.78, 0.79, 0.80, 1)

# ── rooms and walls (docs/draws/floor.drawio) ─────────────────────────────────
BOUNDS = (-4.5, 5.5, -4.0, 4.0)
ROOMS = {'bedroom': (-4.5, -1.8, -0.58, 4.0), 'living room': (-1.8, 1.9, 0.6, 4.0), 'study room': (1.9, 5.5, 0.64, 4.0),
         'hall': (-1.8, 5.5, -0.58, 0.62), 'kitchen': (-1.76, 1.96, -4.0, -0.58), 'warehouse': (-3.0, -1.76, -2.32, -0.58),
         'laundry room': (-3.0, -1.76, -4.0, -2.32), 'dress room': (-4.5, -3.0, -4.0, -0.58),
         'corridor': (1.96, 2.76, -4.0, -0.58), 'restroom': (2.76, 5.5, -4.0, -0.58)}
FLOORS = {'restroom': 'mat_bath_tile', 'laundry room': 'mat_dark_tile'}   # others: laminate
# (x0, y0, x1, y1, [(door_from, door_to) along the wall]). Doors are 0.9 m (drawio 0.74-0.86): the planner
# inflates by the robot radius 0.36 m and needs a free corridor through them.
WALLS = [(-4.5, -4.0, 5.5, -4.0, []), (-4.5, 4.0, 5.5, 4.0, []),
         (-4.5, -4.0, -4.5, 4.0, []), (5.5, -4.0, 5.5, 4.0, []),
         (-1.8, 0.6, -1.8, 4.0, []),                                 # bedroom | living room
         (1.9, 0.64, 1.9, 4.0, []),                                  # living room | study
         (1.9, 0.64, 5.5, 0.64, [(1.9, 2.76)]),                      # study | hall (door)
         (-4.5, -0.58, -1.76, -0.58, [(-4.15, -3.25)]),              # bedroom | dress room (door), warehouse
         (-1.76, -4.0, -1.76, -0.58, [(-3.42, -2.52), (-1.98, -1.08)]),  # kitchen | laundry (door), warehouse (door)
         (-3.0, -4.0, -3.0, -0.58, []),                              # dress room | warehouse, laundry
         (-3.0, -2.32, -1.76, -2.32, []),                            # warehouse | laundry
         (1.96, -4.0, 1.96, -0.58, []),                              # kitchen | corridor
         (2.76, -4.0, 2.76, -0.58, []),                              # corridor | restroom
         (2.76, -0.58, 5.5, -0.58, [(3.68, 4.58)])]                  # restroom | hall (door)


def f3(v):
    return ' '.join(f'{x:.4f}' for x in v)


def quat_z(yaw):
    return f'{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}'


def box(name, pos, size, mat=None, rgba=None, extra=''):
    look = f'material="{mat}"' if mat else f'rgba="{f3(rgba or (0.7, 0.7, 0.7, 1))}"'
    return f'<geom name="{name}" type="box" pos="{f3(pos)}" size="{f3(size)}" {look} {extra}/>'


def cyl(name, pos, r, h, mat=None, rgba=None, extra=''):
    look = f'material="{mat}"' if mat else f'rgba="{f3(rgba or (0.7, 0.7, 0.7, 1))}"'
    return f'<geom name="{name}" type="cylinder" pos="{f3(pos)}" size="{r:.4f} {h / 2:.4f}" {look} {extra}/>'


def ell(name, pos, size, rgba, extra=''):
    return f'<geom name="{name}" type="ellipsoid" pos="{f3(pos)}" size="{f3(size)}" rgba="{f3(rgba)}" {extra}/>'


VIS = 'contype="0" conaffinity="0"'


def stadium(name, pos, L, r, h, rgba=None, mat=None, axis='x', extra=''):
    """Flat oval (two half-discs joined by a box): straight part L, end radius r, thickness h."""
    x, y, z = pos
    a = (L / 2, 0) if axis == 'x' else (0, L / 2)
    size = (L / 2, r, h / 2) if axis == 'x' else (r, L / 2, h / 2)
    return (box(f'{name}_mid', pos, size, mat, rgba, extra) +
            cyl(f'{name}_end0', (x - a[0], y - a[1], z), r, h, mat, rgba, extra) +
            cyl(f'{name}_end1', (x + a[0], y + a[1], z), r, h, mat, rgba, extra))


def cap(name, p0, p1, r, rgba, extra=''):
    return f'<geom name="{name}" type="capsule" fromto="{f3(p0)} {f3(p1)}" size="{r:.4f}" rgba="{f3(rgba)}" {extra}/>'


def walls_xml():
    g = []
    for i, (x0, y0, x1, y1, doors) in enumerate(WALLS):
        horiz = abs(y1 - y0) < 1e-6
        a, b = (x0, x1) if horiz else (y0, y1)
        cuts = [a] + [v for d in sorted(doors) for v in d] + [b]
        for k in range(0, len(cuts), 2):
            s, e = cuts[k], cuts[k + 1]
            if e - s < 1e-3:
                continue
            c, half = (s + e) / 2, (e - s) / 2 + (WALL_T / 2 if k in (0, len(cuts) - 2) else 0)
            pos = (c, y0, WALL_H / 2) if horiz else (x0, c, WALL_H / 2)
            size = (half, WALL_T / 2, WALL_H / 2) if horiz else (WALL_T / 2, half, WALL_H / 2)
            g.append(box(f'wall{i}_{k // 2}', pos, size, 'mat_wall', extra='group="2"'))
        for d0, d1 in doors:                                   # header above each doorway
            c = (d0 + d1) / 2
            pos = (c, y0, 2.25) if horiz else (x0, c, 2.25)
            size = ((d1 - d0) / 2, WALL_T / 2, 0.2) if horiz else (WALL_T / 2, (d1 - d0) / 2, 0.2)
            g.append(box(f'lintel{i}_{d0:.1f}', pos, size, 'mat_wall', extra='group="2"'))
    for room, (x0, x1, y0, y1) in ROOMS.items():                # per-room floor (visual; the physics floor is a plane)
        n = room.replace(' ', '_')
        g.append(box(f'floor_{n}', ((x0 + x1) / 2, (y0 + y1) / 2, -0.005), ((x1 - x0) / 2, (y1 - y0) / 2, 0.005),
                     FLOORS.get(room, 'mat_laminate'), extra=VIS))
    # ceiling with recessed downlights (hidden in the overview camera like the walls)
    bx0, bx1, by0, by1 = BOUNDS
    g.append(box('ceiling', ((bx0 + bx1) / 2, (by0 + by1) / 2, CEIL + 0.01), ((bx1 - bx0) / 2 + 0.1, (by1 - by0) / 2 + 0.1, 0.01),
                 'mat_ceiling', extra=f'group="2" {VIS}'))
    lights = []
    for (x0, x1, y0, y1) in ROOMS.values():                 # a grid of downlights ~1.6 m apart per room
        nx, ny = max(1, round((x1 - x0) / 1.6)), max(1, round((y1 - y0) / 1.6))
        lights += [(x0 + (i + 0.5) * (x1 - x0) / nx, y0 + (j + 0.5) * (y1 - y0) / ny) for i in range(nx) for j in range(ny)]
    for k, (x, y) in enumerate(lights):
        g.append(cyl(f'downlight{k}', (x, y, CEIL - 0.001), 0.06, 0.004, 'mat_downlight', extra=f'group="2" {VIS}'))
    return ''.join(g)


# ── furniture (static bodies built from boxes/cylinders) ──────────────────────
def body(name, pos, yaw, geoms):
    return f'<body name="{name}" pos="{f3(pos)}" quat="{quat_z(yaw)}">{"".join(geoms)}</body>'


def table(name, pos, yaw, L, W, H, top=None, top_rgba=WHITE, leg=BLACK, leg_r=0.018):
    g = [box(f'{name}_top', (0, 0, H - 0.015), (L / 2, W / 2, 0.015), top, rgba=top_rgba)]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        g.append(cyl(f'{name}_leg{i}', (sx * (L / 2 - 0.05), sy * (W / 2 - 0.05), (H - 0.03) / 2), leg_r, H - 0.03, rgba=leg))
    return body(name, pos, yaw, g)


def shell_chair(name, pos, yaw, rgba=OFFWHITE):
    """White moulded plastic dining chair (seat + curved back shell, thin legs)."""
    g = [box(f'{name}_seat', (0, 0, 0.45), (0.22, 0.22, 0.02), rgba=rgba),
         box(f'{name}_back', (0.2, 0, 0.66), (0.025, 0.2, 0.2), rgba=rgba),
         box(f'{name}_armL', (0.08, -0.2, 0.55), (0.14, 0.02, 0.08), rgba=rgba),
         box(f'{name}_armR', (0.08, 0.2, 0.55), (0.14, 0.02, 0.08), rgba=rgba)]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        g.append(cyl(f'{name}_leg{i}', (sx * 0.18, sy * 0.18, 0.215), 0.012, 0.43, rgba=rgba))
    return body(name, pos, yaw, g)


def wood_chair(name, pos, yaw):
    g = [box(f'{name}_seat', (0, 0, 0.45), (0.21, 0.21, 0.02), 'mat_wood_top'),
         box(f'{name}_back', (0.19, 0, 0.7), (0.02, 0.21, 0.12), 'mat_wood_top')]
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        g.append(box(f'{name}_leg{i}', (sx * 0.18, sy * 0.18, 0.215), (0.016, 0.016, 0.215), 'mat_wood_top'))
    return body(name, pos, yaw, g)


def office_chair(name, pos, yaw):
    grey = (0.42, 0.43, 0.45, 1)
    g = [box(f'{name}_seat', (0, 0, 0.48), (0.25, 0.25, 0.05), rgba=grey),
         box(f'{name}_back', (0.22, 0, 0.85), (0.03, 0.22, 0.3), rgba=grey),
         box(f'{name}_head', (0.22, 0, 1.2), (0.03, 0.13, 0.06), rgba=grey),
         cyl(f'{name}_post', (0, 0, 0.28), 0.025, 0.35, rgba=(0.2, 0.2, 0.2, 1))]
    for k in range(5):
        a = 2 * math.pi * k / 5
        g.append(box(f'{name}_foot{k}', (0.16 * math.cos(a), 0.16 * math.sin(a), 0.05), (0.15, 0.015, 0.015),
                     rgba=(0.2, 0.2, 0.2, 1)).replace('/>', f' euler="0 0 {a:.3f}"/>'))
    return body(name, pos, yaw, g)


def window(name, x, y0, y1, z0, z1, glass, horiz=False):
    """Glass pane + white frame on a wall at x (vertical wall) or at y=x (horiz)."""
    c, h = (y0 + y1) / 2, (y1 - y0) / 2
    zc, zh = (z0 + z1) / 2, (z1 - z0) / 2
    P = (lambda a, b, z: (b, a, z)) if horiz else (lambda a, b, z: (a, b, z))
    S = (lambda a, b, z: (b, a, z)) if horiz else (lambda a, b, z: (a, b, z))
    g = [box(f'{name}_glass', P(x, c, zc), S(0.056, h, zh), rgba=glass, extra=f'group="2" {VIS}')]
    for k, (cc, hh, zz, zzh) in enumerate(((c, h + 0.04, z0 - 0.02, 0.03), (c, h + 0.04, z1 + 0.02, 0.03),
                                           (y0 - 0.02, 0.03, zc, zh), (y1 + 0.02, 0.03, zc, zh), (c, 0.02, zc, zh))):
        g.append(box(f'{name}_frame{k}', P(x, cc, zz), S(0.06, hh, zzh), rgba=WHITE, extra=f'group="2" {VIS}'))
    return g


def curtain(name, x, y, w, horiz=False, rgba=CREAM):
    """Pleated curtain hanging from the ceiling: a row of thin vertical slats."""
    g = []
    n = max(3, int(w / 0.06))
    for k in range(n):
        o = -w / 2 + (k + 0.5) * w / n
        dx = 0.03 if k % 2 else 0.0
        p = (y + o, x + dx, 1.25) if horiz else (x + dx, y + o, 1.25)
        s = (w / n / 2, 0.015, 1.15) if horiz else (0.015, w / n / 2, 1.15)
        g.append(box(f'{name}_{k}', p, s, rgba=rgba, extra=VIS))
    return g


def furniture(cab_style=None, site=None):
    """(xml, spots). A spot = a surface the robot can visit: kcare ENV entry material."""
    F, spots = [], {}

    def spot(key, room, centre, yaw_front, depth, width, height, mode='left', **extra):
        """centre: surface centre (x, y); yaw_front: direction the front faces (toward where the robot stands)."""
        spots[f'{key}@{room}'] = dict(centre=[round(float(centre[0]), 3), round(float(centre[1]), 3)],
                                      front_yaw_deg=round(math.degrees(yaw_front), 1),
                                      depth=float(depth), width=float(width), height=float(height),
                                      default_mode=mode, **extra)

    # ── living room: sofa + desk on the west wall (bedroom side), window on the north wall, TV on the east wall
    F += [f'<body name="living_window">{"".join(window("win_l", 4.0, -0.9, 0.9, 0.85, 2.2, (0.78, 0.84, 0.8, 1), horiz=True))}'
          f'{"".join(curtain("curt_l0", 3.9, -1.2, 0.5, horiz=True) + curtain("curt_l1", 3.9, 1.05, 0.3, horiz=True))}</body>']
    F.append(body('aircon', (1.6, 3.72, 0), -math.pi / 2, [
        box('aircon_body', (0, 0, 0.92), (0.2, 0.26, 0.92), rgba=OFFWHITE),
        box('aircon_grille', (0.201, 0, 1.55), (0.002, 0.2, 0.2), rgba=(0.85, 0.85, 0.84, 1), extra=VIS)]))
    # cream modular sofa (photo): two-seater with one high rounded arm on the east end, an armless
    # lounge module on the west end, back cushions along the whole length, a white throw pillow
    sofa = (0.91, 0.90, 0.86, 1)
    F.append(body('sofa', (-1.32, 2.98, 0), -math.pi / 2, [
        box('sofa_base', (0.0, 0.0, 0.17), (0.95, 0.4, 0.17), rgba=sofa),
        box('sofa_seat0', (0.38, 0.05, 0.4), (0.42, 0.33, 0.07), rgba=sofa),
        box('sofa_seat1', (-0.52, 0.07, 0.38), (0.43, 0.35, 0.06), rgba=sofa),
        box('sofa_back', (0.0, -0.3, 0.6), (0.95, 0.1, 0.18), rgba=sofa),
        cap('sofa_back_top', (-0.93, -0.3, 0.78), (0.93, -0.3, 0.78), 0.1, sofa),
        box('sofa_arm', (0.86, 0.0, 0.4), (0.09, 0.4, 0.22), rgba=sofa),
        cap('sofa_arm_top', (0.86, -0.38, 0.62), (0.86, 0.38, 0.62), 0.09, sofa),
        box('sofa_pillow', (-0.75, -0.17, 0.62), (0.17, 0.07, 0.15), rgba=(0.97, 0.96, 0.93, 1)).replace('/>', ' euler="0.25 0 0.2"/>'),
        box('sofa_seam', (-0.09, 0.3, 0.42), (0.004, 0.09, 0.05), rgba=(0.8, 0.79, 0.75, 1), extra=VIS)]))
    # oval two-tier side table (photo): white rim with a blue-grey inset top, white lower tray, chrome legs;
    # long axis away from the wall
    F.append(body('side_table', (-1.38, 1.85, 0), math.pi / 2, [
        stadium('stab_top', (0, 0, 0.6), 0.3, 0.19, 0.03, rgba=OFFWHITE, axis='y'),
        stadium('stab_inset', (0, 0, 0.616), 0.3, 0.16, 0.002, rgba=(0.24, 0.31, 0.39, 1), axis='y', extra=VIS),
        stadium('stab_tray', (0, 0, 0.16), 0.24, 0.16, 0.02, rgba=OFFWHITE, axis='y')] +
        [cyl(f'stab_leg{k}', (sx * 0.13, sy * 0.22, 0.3), 0.007, 0.6, rgba=CHROME)
         for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))] +
        [box('stab_remote', (0.05, 0.05, 0.18), (0.02, 0.08, 0.008), rgba=BLACK, extra=VIS)]))
    F.append(body('tv_stand', (1.67, 2.6, 0), -math.pi / 2, [
        box('tv_stand_box', (0, 0, 0.2), (0.8, 0.18, 0.2), rgba=OFFWHITE),
        box('tv_stand_gap', (0, -0.181, 0.2), (0.78, 0.001, 0.005), rgba=(0.8, 0.8, 0.8, 1), extra=VIS),
        box('tv_screen', (0, 0.02, 0.76), (0.55, 0.02, 0.32), rgba=(0.04, 0.04, 0.05, 1)),
        box('tv_foot', (0, 0.02, 0.41), (0.12, 0.08, 0.01), rgba=(0.1, 0.1, 0.1, 1))]))
    # ── study room: desk on its west wall
    F.append(table('study_desk', (2.25, 2.5, 0), math.pi / 2, 1.0, 0.6, 0.72, top_rgba=OFFWHITE))
    F.append(body('study_monitor', (2.08, 2.5, 0.72), math.pi / 2, [
        box('smonitor_screen', (0, 0, 0.3), (0.3, 0.015, 0.18), rgba=(0.05, 0.05, 0.06, 1)),
        box('smonitor_stand', (0, 0.02, 0.06), (0.06, 0.06, 0.06), rgba=(0.15, 0.15, 0.16, 1))]))
    spot('desk', 'study room', (2.25, 2.5), 0.0, 0.6, 1.0, 0.72)

    # ── the open-drawer cabinet (style A_home = the A_real pedestal + casters), drawio 'Drawer': kitchen north-west
    #    corner, back on the warehouse wall, front facing east. ROBOT_AT_DRAWER puts the robot base where the
    #    training had it relative to the cabinet (CAB_REL, drawer_ready.build_scene nominal scene); kcare's
    #    drawer@kitchen ENV entry sends the robot exactly there (make_kcare_sim_location.py).
    site = site or CAB_SITE
    cab = CAB_SITES[site]
    rx, ry, rh = _robot_at(cab)
    cab_xy = cab[:2]
    vla_cab, _, vla_H = cabinet.cabinet_xml(dict(cab_style or cabinet.STYLES['A_home']), pos=(*cab_xy, 0), yaw=cab[2], name='cab')
    F.append(vla_cab)
    spot('drawer', 'kitchen', cab_xy, cab[2] + math.pi, 0.5, 0.4, round(vla_H, 3), site=site,
         handle='gray handle', handle_height=0.436, aliases=['drawer@living room'],
         robot_pose=dict(x=round(rx, 3), y=round(ry, 3), heading_deg=round(math.degrees(rh), 2)))

    # ── kitchen: counter, tall niche unit and wire shelf along the south wall, 4-door fridge on the east wall,
    #    dining table against the east wall
    # white handleless kitchen (photo): from the west, a drawer stack, gas hob with a pot, sink with a tall
    # gooseneck tap and a dish rack; white subway backsplash; flat white wall cabinets up to the ceiling
    flat_w = (0.95, 0.95, 0.94, 1)
    seam = (0.78, 0.78, 0.77, 1)
    F.append(body('counter', (-0.8, -3.65, 0), 0, [
        box('counter_body', (0, 0, 0.44), (0.9, 0.3, 0.44), rgba=flat_w),
        box('counter_top', (0, 0.02, 0.89), (0.91, 0.32, 0.012), rgba=(0.93, 0.93, 0.92, 1)),
        box('counter_kick', (0, 0.25, 0.05), (0.9, 0.05, 0.05), rgba=(0.82, 0.82, 0.81, 1)),
        box('backsplash', (0, -0.295, 1.175), (0.9, 0.004, 0.275), 'mat_backsplash', extra=VIS),
        box('stove', (-0.3, 0.0, 0.904), (0.25, 0.22, 0.004), rgba=(0.12, 0.12, 0.13, 1)),
        cyl('stove_burner0', (-0.42, 0.0, 0.91), 0.06, 0.006, rgba=(0.3, 0.3, 0.32, 1), extra=VIS),
        cyl('stove_burner1', (-0.18, 0.0, 0.91), 0.06, 0.006, rgba=(0.3, 0.3, 0.32, 1), extra=VIS),
        cyl('pot', (-0.18, 0.0, 0.97), 0.1, 0.11, rgba=(0.72, 0.73, 0.75, 1)),
        box('sink_basin', (0.5, 0.02, 0.9025), (0.22, 0.2, 0.001), rgba=(0.62, 0.64, 0.66, 1), extra=VIS),
        cyl('sink_tap', (0.5, -0.2, 1.08), 0.012, 0.36, rgba=CHROME),
        cap('sink_spout', (0.5, -0.2, 1.26), (0.5, -0.04, 1.22), 0.011, CHROME),
        box('dish_rack', (0.78, -0.12, 0.97), (0.1, 0.16, 0.07), rgba=(0.9, 0.9, 0.9, 1)),
        box('upper_cabinets', (0, -0.12, 1.925), (0.9, 0.18, 0.475), rgba=flat_w)] +
        [box(f'counter_door{k}', (-0.15 + 0.35 * k, 0.301, 0.46), (0.001, 0.001, 0.38), rgba=seam, extra=VIS) for k in range(3)] +
        [box(f'counter_drawer{k}', (-0.7, 0.301, z), (0.2, 0.001, 0.0015), rgba=seam, extra=VIS) for k, z in enumerate((0.3, 0.6))] +
        [box(f'upper_seam{k}', (-0.9 + 0.3 * k, 0.061, 1.925), (0.0015, 0.001, 0.47), rgba=seam, extra=VIS) for k in range(1, 6)]))
    spot('counter', 'kitchen', (-0.8, -3.63), math.pi / 2, 0.64, 1.8, 0.9)
    # tall unit: closed cabinet on top, open niches with rice cooker / microwave, cupboard below
    F.append(body('kitchen_shelf', (0.42, -3.72, 0), 0, [
        box('kshelf_side0', (-0.3, 0, 1.2), (0.012, 0.27, 1.2), rgba=flat_w),
        box('kshelf_side1', (0.3, 0, 1.2), (0.012, 0.27, 1.2), rgba=flat_w),
        box('kshelf_back', (0, -0.265, 1.2), (0.3, 0.005, 1.2), rgba=flat_w),
        box('kshelf_low', (0, 0, 0.25), (0.29, 0.26, 0.25), rgba=flat_w),
        box('kshelf_low_seam', (0, 0.261, 0.27), (0.001, 0.001, 0.22), rgba=seam, extra=VIS),
        box('kshelf_top', (0, 0, 1.97), (0.29, 0.26, 0.43), rgba=flat_w),
        box('kshelf_top_seam', (0, 0.261, 1.97), (0.001, 0.001, 0.41), rgba=seam, extra=VIS)] +
        [box(f'kshelf_board{k}', (0, 0, z), (0.29, 0.26, 0.012), rgba=flat_w) for k, z in enumerate((0.85, 1.2))] +
        [box('microwave', (0, 0.02, 1.02), (0.24, 0.18, 0.15), rgba=(0.94, 0.94, 0.94, 1)),
         box('microwave_door', (-0.04, 0.201, 1.02), (0.16, 0.002, 0.11), rgba=(0.12, 0.12, 0.13, 1), extra=VIS),
         cyl('rice_cooker', (0, 0.02, 1.34), 0.13, 0.25, rgba=(0.95, 0.95, 0.95, 1)),
         cyl('rice_cooker_lid', (0, 0.02, 1.47), 0.1, 0.01, rgba=(0.55, 0.42, 0.3, 1), extra=VIS)]))
    spot('shelf', 'kitchen', (0.42, -3.72), math.pi / 2, 0.52, 0.58, 0.512)
    # white 5-tier wire shelf: air fryer, bottles, boxes
    wire = (0.92, 0.92, 0.92, 1)
    F.append(body('wire_shelf', (0.96, -3.8, 0), 0,
        [cyl(f'wshelf_post{k}', (sx * 0.21, sy * 0.16, 0.85), 0.009, 1.7, rgba=wire)
         for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))] +
        [box(f'wshelf_tier{k}', (0, 0, z), (0.22, 0.17, 0.006), rgba=wire) for k, z in enumerate((0.06, 0.45, 0.85, 1.25, 1.65))] +
        [box('air_fryer', (0, 0.0, 0.62), (0.14, 0.15, 0.16), rgba=(0.93, 0.93, 0.92, 1)),
         cyl('wshelf_bottle0', (-0.12, 0.05, 1.37), 0.035, 0.23, rgba=(0.2, 0.45, 0.25, 1)),
         cyl('wshelf_bottle1', (-0.04, 0.05, 1.36), 0.03, 0.21, rgba=(0.85, 0.75, 0.3, 1)),
         box('wshelf_box', (0.1, 0.0, 1.0), (0.09, 0.12, 0.14), rgba=(0.85, 0.35, 0.3, 1)),
         box('wshelf_basket', (0.0, 0.0, 0.17), (0.18, 0.13, 0.1), rgba=(0.97, 0.97, 0.97, 1))]))
    # white table with one rounded end (photo), flat end on the east wall; a white shell chair
    F.append(body('dining_table', (1.18, -2.81, 0), math.pi, [
        box('dtable_mid', (-0.19, 0, 0.73), (0.54, 0.31, 0.015), rgba=OFFWHITE),
        cyl('dtable_end', (0.35, 0, 0.73), 0.31, 0.03, rgba=OFFWHITE),
        cyl('dtable_mat', (0.3, 0.05, 0.746), 0.1, 0.002, rgba=(0.6, 0.78, 0.72, 1), extra=VIS)] +
        [cyl(f'dtable_leg{k}', (sx, sy * 0.22, 0.36), 0.018, 0.71, rgba=OFFWHITE)
         for k, (sx, sy) in enumerate(((0.45, 1), (0.45, -1), (-0.65, 1), (-0.65, -1)))]))
    spot('dining table', 'kitchen', (1.18, -2.81), math.pi / 2, 0.62, 1.39, 0.745)
    F += [shell_chair('dchair0', (1.3, -2.15, 0), math.pi / 2)]
    # TCL 4-door fridge: full-height centre split, a horizontal split low down, recessed handles, display
    F.append(body('fridge', (1.55, -1.25, 0), -math.pi / 2, [
        box('fridge_body', (0, 0, 0.92), (0.36, 0.35, 0.92), rgba=(0.95, 0.95, 0.95, 1)),
        box('fridge_split_v', (0, -0.351, 0.92), (0.0025, 0.001, 0.88), rgba=(0.55, 0.55, 0.56, 1), extra=VIS),
        box('fridge_split_h', (0, -0.351, 0.8), (0.35, 0.001, 0.0025), rgba=(0.55, 0.55, 0.56, 1), extra=VIS),
        box('fridge_handle0', (-0.025, -0.352, 1.1), (0.008, 0.001, 0.22), rgba=(0.7, 0.7, 0.71, 1), extra=VIS),
        box('fridge_handle1', (0.025, -0.352, 1.1), (0.008, 0.001, 0.22), rgba=(0.7, 0.7, 0.71, 1), extra=VIS),
        box('fridge_display', (-0.12, -0.352, 1.38), (0.022, 0.001, 0.09), rgba=(0.1, 0.1, 0.12, 1), extra=VIS),
        box('fridge_label', (-0.22, -0.352, 1.45), (0.03, 0.001, 0.04), rgba=(0.9, 0.85, 0.6, 1), extra=VIS)]))
    # grey fabric recycling tote (photo + drawio 'Trash bin', north of the fridge)
    F.append(body('tote_bag', (*TOTE_AT[site], 0), -math.pi / 2, [
        box('tote_body', (0, 0, 0.24), (0.17, 0.12, 0.24), rgba=(0.36, 0.38, 0.4, 1)),
        box('tote_print', (0, -0.121, 0.25), (0.11, 0.001, 0.1), rgba=(0.75, 0.76, 0.77, 1), extra=VIS),
        cap('tote_handle', (-0.08, 0, 0.5), (0.08, 0, 0.5), 0.012, (0.3, 0.32, 0.34, 1), extra=VIS)]))

    # ── laundry room (washer on the west wall, facing the door) and warehouse (storage shelf)
    F.append(body('washer', (-2.64, -3.0, 0), -math.pi / 2, [
        box('washer_body', (0, 0, 0.43), (0.3, 0.3, 0.43), rgba=(0.94, 0.94, 0.94, 1)),
        f'<geom name="washer_door" type="cylinder" fromto="0 0.30 0.4 0 0.305 0.4" size="0.19" rgba="0.2 0.2 0.22 1" {VIS}/>']))
    spot('washer', 'laundry room', (-2.64, -3.0), 0.0, 0.6, 0.6, 0.86)
    F.append(body('laundry_shelf', (-2.55, -3.78, 0), math.pi / 2, [
        box('lshelf_side0', (0, -0.3, 0.8), (0.2, 0.01, 0.8), rgba=OFFWHITE),
        box('lshelf_side1', (0, 0.3, 0.8), (0.2, 0.01, 0.8), rgba=OFFWHITE)] +
        [box(f'lshelf_bin{k}', (0, 0, 0.25 + 0.4 * k), (0.18, 0.26, 0.14), rgba=(0.95, 0.95, 0.95, 1)) for k in range(4)]))
    F.append(body('storage_shelf', (-2.73, -1.45, 0), 0, [
        box('sshelf_side0', (0, -0.4, 0.9), (0.2, 0.01, 0.9), rgba=OFFWHITE),
        box('sshelf_side1', (0, 0.4, 0.9), (0.2, 0.01, 0.9), rgba=OFFWHITE)] +
        [box(f'sshelf_board{k}', (0, 0, 0.05 + 0.44 * k), (0.2, 0.39, 0.01), rgba=OFFWHITE) for k in range(5)] +
        [box(f'sshelf_box{k}', (0, -0.18 + 0.36 * (k % 2), 0.2 + 0.44 * (k // 2)), (0.17, 0.16, 0.14), rgba=(0.96, 0.96, 0.95, 1))
         for k in range(6)]))
    sage = (0.62, 0.68, 0.6, 1)
    F.append(body('laundry_basket', (-2.0, -3.72, 0), 0, [
        cyl('basket_low', (0, 0, 0.12), 0.17, 0.24, rgba=sage), cyl('basket_high', (0, 0, 0.34), 0.2, 0.2, rgba=sage),
        cyl('basket_rim', (0, 0, 0.445), 0.205, 0.01, rgba=(0.55, 0.6, 0.53, 1)),
        box('basket_hand0', (0.2, 0, 0.43), (0.01, 0.05, 0.02), rgba=sage), box('basket_hand1', (-0.2, 0, 0.43), (0.01, 0.05, 0.02), rgba=sage)]))

    # ── bedroom: bed with an upholstered white headboard against the north wall, window + curtains,
    #    white 3-drawer chest and a cream stool on the west wall
    # white upholstered bed (photo): thick rounded frame, tall headboard with rounded corners, beige quilted
    # duvet (texture cut from the photo), white + grey pillows
    up = (0.95, 0.95, 0.94, 1)
    F.append(body('bed', (-2.77, 2.83, 0), 0, [
        box('bed_frame', (0, -0.02, 0.15), (0.8, 1.0, 0.13), rgba=up),
        cap('bed_frame_edgeL', (-0.8, -1.0, 0.18), (-0.8, 0.98, 0.18), 0.1, up),
        cap('bed_frame_edgeR', (0.8, -1.0, 0.18), (0.8, 0.98, 0.18), 0.1, up),
        cap('bed_frame_edgeF', (-0.8, -1.02, 0.18), (0.8, -1.02, 0.18), 0.1, up),
        box('bed_mattress', (0, -0.02, 0.36), (0.76, 0.96, 0.08), rgba=(0.96, 0.96, 0.95, 1)),
        box('bed_duvet', (0, -0.25, 0.47), (0.82, 0.76, 0.04), 'mat_duvet'),
        box('bed_duvet_fall', (0, -1.0, 0.33), (0.82, 0.02, 0.15), 'mat_duvet'),
        box('bed_pillow0', (0.4, 0.74, 0.52), (0.3, 0.13, 0.07), rgba=(0.8, 0.81, 0.83, 1)),
        box('bed_pillow1', (-0.3, 0.72, 0.52), (0.3, 0.13, 0.07), rgba=(0.97, 0.97, 0.97, 1)),
        box('bed_pillow2', (0.05, 0.8, 0.66), (0.28, 0.06, 0.16), rgba=(0.98, 0.98, 0.98, 1)).replace('/>', ' euler="-0.5 0 0.15"/>'),
        box('bed_headboard', (0, 1.04, 0.68), (0.74, 0.08, 0.42), rgba=up),
        cap('bed_head_top', (-0.74, 1.04, 1.1), (0.74, 1.04, 1.1), 0.08, up),
        cap('bed_head_sideL', (-0.82, 1.04, 0.3), (-0.82, 1.04, 1.06), 0.08, up),
        cap('bed_head_sideR', (0.82, 1.04, 0.3), (0.82, 1.04, 1.06), 0.08, up),
        box('bed_head_top_fill', (0, 1.04, 1.08), (0.82, 0.08, 0.02), rgba=up)]))
    F += [f'<body name="bedroom_window">{"".join(window("win_b", 4.0, -4.33, -3.78, 0.05, 2.15, (0.12, 0.13, 0.15, 1), horiz=True))}'
          f'{"".join(curtain("curt_b0", 3.9, -4.4, 0.12, horiz=True) + curtain("curt_b1", 3.9, -3.72, 0.1, horiz=True))}</body>']
    chest_st = dict(cabinet.STYLES['W_chest'])
    chest_body, _, chest_H = cabinet.cabinet_xml(chest_st, pos=(-4.23, 0.1, 0), yaw=math.pi, name='chest')
    F.append(chest_body)
    # white desk organizer with scalloped dividers on the back right of the chest, a white board leaning on the wall
    F.append(body('chest_organizer', (-4.23, 0.1, chest_H), 0, [
        box('organizer_base', (-0.09, 0.2, 0.006), (0.1, 0.19, 0.006), rgba=OFFWHITE),
        box('organizer_back', (-0.18, 0.2, 0.08), (0.008, 0.19, 0.08), rgba=OFFWHITE)] +
        [box(f'organizer_div{k}', (-0.09, 0.02 + 0.09 * k, 0.05), (0.1, 0.005, 0.05), rgba=OFFWHITE) for k in range(5)]))
    F.append(body('leaning_board', (-4.38, 1.65, 0), 0, [
        box('board', (0, 0, 0.55), (0.012, 0.3, 0.55), rgba=(0.97, 0.97, 0.96, 1)).replace('/>', ' euler="0 -0.12 0"/>')]))
    spot('chest', 'bedroom', (-4.23, 0.1), 0.0, chest_st['depth'], chest_st['width'], round(chest_H, 3),
         handle='gold knob', handle_height=round(chest_st['plinth'] + 2.5 * 0.2, 3))
    # cream upholstered waterfall side table: rounded top slab, rounded sides, low shelf
    st_c = (0.94, 0.92, 0.88, 1)
    F.append(body('stool', (-4.25, 0.95, 0), 0, [
        box('stool_top', (0, 0, 0.42), (0.2, 0.2, 0.025), rgba=st_c),
        cap('stool_edge0', (-0.2, -0.2, 0.395), (0.2, -0.2, 0.395), 0.05, st_c),
        cap('stool_edge1', (-0.2, 0.2, 0.395), (0.2, 0.2, 0.395), 0.05, st_c),
        box('stool_side0', (0, -0.23, 0.2), (0.2, 0.02, 0.2), rgba=st_c),
        box('stool_side1', (0, 0.23, 0.2), (0.2, 0.02, 0.2), rgba=st_c),
        box('stool_shelf', (0, 0, 0.12), (0.19, 0.22, 0.02), rgba=st_c)]))
    spot('stool', 'bedroom', (-4.25, 0.95), 0.0, 0.4, 0.52, 0.445)

    # ── dress room (photo): along its west wall an open white hanging unit, a drawer unit under a hanging rail of
    #    clothes, a mirror panel, curved corner shelves with bags
    cw = (0.96, 0.96, 0.95, 1)
    F.append(body('closet_hang', (-4.18, -1.35, 0), math.pi / 2, [
        box('chang_side0', (0, -0.4, 1.05), (0.27, 0.012, 1.05), rgba=cw),
        box('chang_side1', (0, 0.4, 1.05), (0.27, 0.012, 1.05), rgba=cw),
        box('chang_top', (0, 0, 2.09), (0.27, 0.4, 0.012), rgba=cw),
        box('chang_mid', (0, 0, 1.0), (0.27, 0.4, 0.012), rgba=cw),
        box('chang_bot', (0, 0, 0.06), (0.27, 0.4, 0.06), rgba=cw),
        cyl('chang_rail0', (0, 0, 1.9), 0.012, 0.78, rgba=CHROME).replace('/>', ' euler="1.5708 0 0"/>'),
        cyl('chang_rail1', (0, 0, 0.9), 0.012, 0.78, rgba=CHROME).replace('/>', ' euler="1.5708 0 0"/>'),
        box('chang_hanger', (0.02, 0.1, 1.75), (0.005, 0.18, 0.12), rgba=(0.98, 0.98, 0.98, 1), extra=VIS)]))
    F.append(body('closet_mirror', (-4.43, -3.0, 0), math.pi / 2, [
        box('mirror_frame', (0, 0, 1.05), (0.24, 0.02, 0.9), rgba=cw),
        box('mirror_glass', (0, -0.021, 1.05), (0.21, 0.001, 0.86), rgba=(0.62, 0.66, 0.7, 1), extra=VIS)]))
    F.append(body('closet_drawers', (-4.2, -2.3, 0), math.pi / 2, [
        box('cdraw_side0', (-0.42, 0, 1.05), (0.012, 0.25, 1.05), rgba=cw),
        box('cdraw_side1', (0.42, 0, 1.05), (0.012, 0.25, 1.05), rgba=cw),
        box('cdraw_top', (0, 0, 2.09), (0.42, 0.25, 0.012), rgba=cw),
        box('cdraw_block', (0, 0, 0.4), (0.41, 0.24, 0.4), rgba=cw),
        cyl('cdraw_rail', (0, 0, 1.75), 0.012, 0.82, rgba=CHROME).replace('/>', ' euler="0 1.5708 0"/>')] +
        [box(f'cdraw_front{k}', (0, -0.241, 0.15 + 0.26 * k), (0.4, 0.001, 0.12), rgba=(0.93, 0.93, 0.92, 1), extra=VIS) for k in range(3)] +
        [box(f'cdraw_cloth{k}', (-0.3 + 0.13 * k, 0, 1.38), (0.05, 0.2, 0.3), rgba=c)
         for k, c in enumerate(((0.62, 0.5, 0.4, 1), (0.66, 0.54, 0.43, 1), (0.85, 0.82, 0.76, 1), (0.6, 0.48, 0.38, 1)))]))
    F.append(body('closet_corner', (-4.17, -3.55, 0), math.pi / 2, [
        box('ccorner_side', (-0.28, 0, 1.05), (0.012, 0.3, 1.05), rgba=cw),
        box('ccorner_back', (0, 0.29, 1.05), (0.28, 0.012, 1.05), rgba=cw),
        box('ccorner_post', (0.26, 0.2, 1.05), (0.02, 0.1, 1.05), rgba=cw)] +
        [g for k, z in enumerate((0.05, 0.45, 0.85, 1.25, 1.65, 2.08)) for g in (
            box(f'ccorner_board{k}', (-0.04, 0.05, z), (0.24, 0.24, 0.012), rgba=cw),
            cyl(f'ccorner_round{k}', (0.06, -0.12, z), 0.14, 0.024, rgba=cw))] +
        [cyl('ccorner_bottle', (-0.05, 0.1, 1.79), 0.04, 0.26, rgba=(0.94, 0.9, 0.8, 1)),
         box('ccorner_bag0', (0.0, 0.0, 1.0), (0.09, 0.06, 0.13), rgba=(0.45, 0.68, 0.85, 1)),
         box('ccorner_bag1', (0.0, 0.0, 0.58), (0.1, 0.07, 0.12), rgba=(0.8, 0.7, 0.55, 1)),
         box('ccorner_bag2', (0.0, 0.0, 0.18), (0.1, 0.07, 0.12), rgba=(0.1, 0.1, 0.11, 1))]))

    # ── restroom: grey tiles, clawfoot tub on the south wall, basin + shelf on the west wall, toilet on the east wall
    tub = (0.97, 0.97, 0.97, 1)
    F.append(body('bathtub', (4.1, -3.62, 0), 0, [
        ell('tub_outer', (0, 0, 0.44), (0.78, 0.36, 0.2), tub),
        ell('tub_inner', (0.02, 0, 0.6), (0.68, 0.28, 0.02), (0.88, 0.9, 0.92, 1), extra=VIS),
        ell('tub_back', (-0.62, 0, 0.62), (0.14, 0.33, 0.1), tub),
        box('tub_tray', (0.2, 0, 0.66), (0.2, 0.08, 0.01), 'mat_wood_top')] +
        [cyl(f'tub_foot{k}', (sx * 0.5, sy * 0.2, 0.1), 0.03, 0.2, rgba=tub)
         for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))]))
    F.append(body('bath_mat', (4.1, -2.95, 0), 0, [box('mat', (0, 0, 0.004), (0.3, 0.2, 0.004), rgba=(0.18, 0.22, 0.45, 1), extra=VIS)]))
    F.append(body('basin', (2.98, -2.2, 0), math.pi, [
        ell('basin_bowl', (-0.05, 0, 0.82), (0.22, 0.26, 0.08), rgba=(0.97, 0.97, 0.97, 1)),
        box('basin_pedestal', (0.05, 0, 0.6), (0.12, 0.14, 0.2), rgba=(0.97, 0.97, 0.97, 1)),
        cyl('basin_tap', (0.12, 0, 0.98), 0.013, 0.2, rgba=CHROME)]))
    F.append(body('bath_shelf', (2.88, -1.5, 0), math.pi, [box('bath_shelf_board', (0, 0, 0.94), (0.1, 0.3, 0.012), rgba=OFFWHITE),
                                                  box('bath_shelf_back', (0.09, 0, 0.9), (0.01, 0.3, 0.05), rgba=OFFWHITE)]))
    spot('shelf', 'restroom', (2.88, -1.5), 0.0, 0.2, 0.6, 0.952)
    F.append(body('toilet', (5.12, -1.7, 0), 0, [
        ell('toilet_bowl', (-0.05, 0, 0.3), (0.26, 0.19, 0.12), rgba=(0.97, 0.97, 0.97, 1)),
        box('toilet_base', (0.05, 0, 0.14), (0.15, 0.13, 0.14), rgba=(0.97, 0.97, 0.97, 1)),
        box('toilet_tank', (0.25, 0, 0.55), (0.08, 0.19, 0.18), rgba=(0.97, 0.97, 0.97, 1))]))
    F.append(body('towel_rack', (2.82, -1.0, 0), 0, [
        box('towel_bar', (0.02, 0, 1.45), (0.01, 0.2, 0.01), rgba=CHROME),
        box('towel', (0.03, 0, 1.2), (0.012, 0.18, 0.25), rgba=(0.55, 0.75, 0.72, 1))]))
    F.append(body('shower', (5.4, -3.3, 0), 0, [cyl('shower_pole', (0, 0, 1.6), 0.012, 1.2, rgba=CHROME),
                                               box('shower_head', (-0.05, 0, 2.2), (0.05, 0.04, 0.02), rgba=CHROME)]))
    return ''.join(F), spots


# ── small objects (YCB scans) ─────────────────────────────────────────────────
# name as kcare will ask for it, YCB id, spot it sits on, offset on the surface (along width, along depth) [m], yaw [deg]
OBJECTS = [
    ('apple', '013_apple', 'desk@study room', (-0.1, -0.12), 0),
    ('banana', '011_banana', 'desk@study room', (0.1, -0.1), 30),
    ('orange', '017_orange', 'desk@study room', (0.3, -0.12), 0),
    ('mug', '025_mug', 'desk@study room', (-0.32, -0.1), 90),
    ('cracker box', '003_cracker_box', 'shelf@kitchen', (0.0, 0.0), 90),
    ('mustard bottle', '006_mustard_bottle', 'counter@kitchen', (0.0, 0.05), 0),
    ('tomato soup can', '005_tomato_soup_can', 'counter@kitchen', (-0.27, 0.08), 0),
    ('pitcher', '019_pitcher_base', 'counter@kitchen', (-0.16, 0.05), 0),
    ('sugar box', '004_sugar_box', 'counter@kitchen', (0.77, 0.12), 20),
    ('bowl', '024_bowl', 'dining table@kitchen', (0.2, 0.0), 0),
    ('plate', '029_plate', 'dining table@kitchen', (-0.3, 0.05), 0),
    ('bleach bottle', '021_bleach_cleanser', 'washer@laundry room', (0.1, 0.0), 0),
    ('red cup', '065-a_cups', 'chest@bedroom', (-0.2, -0.08), 0),
    ('tennis ball', '056_tennis_ball', 'stool@bedroom', (0.0, 0.0), 0),
    ('spray bottle', '022_windex_bottle', 'shelf@restroom', (-0.12, 0.0), 90),
    ('sponge', '026_sponge', 'shelf@restroom', (0.15, 0.0), 0),
]


def mesh_min_z(path):
    """Lowest vertex of an OBJ (so the object rests on the surface)."""
    zs = [float(l.split()[3]) for l in open(path) if l.startswith('v ')]
    return min(zs)


def objects_xml(spots):
    assets, bodies, placed = [], [], {}
    for name, ycb, spot_key, (u, w), yaw_deg in OBJECTS:
        sp = spots[spot_key]
        cx, cy = sp['centre']
        fy = math.radians(sp['front_yaw_deg'])
        along = np.array([-math.sin(fy), math.cos(fy)])            # along the front edge
        inward = -np.array([math.cos(fy), math.sin(fy)])           # from the front edge into the surface
        p = np.array([cx, cy]) + along * u + inward * w
        obj = f'{SIM}/assets/ycb/{ycb}/google_16k/textured.obj'
        z = sp['height'] - mesh_min_z(obj) + 0.002
        tag = re.sub(r'[^a-z0-9]', '_', name)
        assets.append(f'<mesh name="m_{tag}" file="{YCB}/{ycb}/google_16k/textured.obj"/>'
                      f'<texture name="t_{tag}" type="2d" file="{YCB}/{ycb}/google_16k/texture_map.png"/>'
                      f'<material name="mt_{tag}" texture="t_{tag}"/>')
        bodies.append(f'<body name="obj_{tag}" pos="{p[0]:.4f} {p[1]:.4f} {z:.4f}" quat="{quat_z(math.radians(yaw_deg))}">'
                      f'<freejoint name="obj_{tag}_free"/>'
                      f'<geom name="obj_{tag}" type="mesh" mesh="m_{tag}" material="mt_{tag}" density="400" '
                      f'friction="1.0 0.01 0.001" condim="4"/></body>')
        placed[name] = dict(ycb=ycb, on=spot_key, pos=[round(float(v), 3) for v in (p[0], p[1], z)])
    return ''.join(assets), ''.join(bodies), placed


def robot_mobile_xml():
    """kaair_slamtec.xml with base_footprint as a mocap body (the base is driven kinematically)."""
    src = open(f'{SIM}/mjcf/kaair_slamtec.xml').read()
    out = src.replace('<body name="base_footprint" gravcomp="1">', '<body name="base_footprint" mocap="true">', 1)
    assert out != src, 'base_footprint body not found'
    out = re.sub(r'<keyframe>.*?</keyframe>', '', out, flags=re.S)
    open(f'{SIM}/mjcf/kaair_slamtec_mobile.xml', 'w').write(out)


def house_xml(robot_include='kaair_slamtec_mobile.xml', frame=None, cab_style=None, light=None, tex_tint=None, site=None):
    """The house as a full MJCF string. frame = (x, y, yaw): place the whole house through that transform (the robot
    stays at the origin: used to build open-drawer training scenes in the house). light: randomize.py's light dict
    (office ranges), mapped onto the house lamps / headlight."""
    furn, spots = furniture(cab_style, site)
    oa, ob, placed = objects_xml(spots)
    lamp, amb = (float(v) for v in os.environ.get('HOUSE_LIGHT', '0.45,0.58').split(','))   # ceiling lamps, ambient
    wood_tint = (*(tex_tint or (1.0, 1.0, 1.0)), 1.0)
    # the office wood texture is far lighter than the oak at home: on the robot's home images the drawer fronts
    # are ~0.48x and the top ~0.7x as bright as the white fridge next to them (calibrate_home.py)
    wood_front = (*(WOOD_FRONT * t for t in wood_tint[:3]), 1.0)
    wood_top = (*(WOOD_TOP * t for t in wood_tint[:3]), 1.0)
    if light:
        lamp *= light.get('diffuse', 0.55) / 0.55
        amb *= light.get('ambient', 0.32) / 0.32
    lights = ''.join(
        f'<light name="lamp_{r.replace(" ", "_")}" pos="{(x0 + x1) / 2:.2f} {(y0 + y1) / 2:.2f} 2.35" dir="0 0 -1" '
        f'diffuse="{lamp:.3f} {lamp:.3f} {lamp * 0.98:.3f}" attenuation="1 0.02 0.005" castshadow="false"/>'
        for r, (x0, x1, y0, y1) in ROOMS.items())
    body = f'{lights}{walls_xml()}{furn}{ob}'
    if frame is not None:
        fx, fy, fa = frame
        body = f'<frame pos="{fx:.5f} {fy:.5f} 0" quat="{quat_z(fa)}">{body}</frame>'
    xml = f'''<mujoco model="kcare_house">
  <include file="{robot_include}"/>
  <visual><headlight ambient="{amb:.3f} {amb:.3f} {amb:.3f}" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/><map znear="0.002" zfar="60"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".93 .94 .95" rgb2=".80 .82 .84" width="256" height="256"/>
    <texture name="tex_laminate" type="2d" file="{HOME_TEX}/floor_laminate.png"/>
    <texture name="tex_backsplash" type="2d" file="{HOME_TEX}/backsplash.png"/>
    <texture name="tex_duvet" type="2d" file="{HOME_TEX}/duvet_photo.png"/>
    <texture name="tex_home_wood" type="2d" file="{HOME_TEX}/home_wood.png"/>
    <texture name="tex_home_wood_v" type="2d" file="{HOME_TEX}/home_wood_v.png"/>
    <texture name="tex_bath_tile" type="2d" file="{HOME_TEX}/bath_tile.png"/>
    <texture name="tex_wood_top" type="2d" file="{TEX}/wood_top.png"/>
    <texture name="tex_gray_side" type="2d" file="{TEX}/gray_side.png"/>
    <material name="mat_laminate" texture="tex_laminate" texuniform="true" texrepeat="0.9 0.9"/>
    <material name="mat_bath_tile" texture="tex_bath_tile" texuniform="true" texrepeat="1.67 1.67"/>
    <material name="mat_dark_tile" texture="tex_bath_tile" texuniform="true" texrepeat="3.3 3.3" rgba="0.8 0.8 0.8 1"/>
    <material name="mat_backsplash" texture="tex_backsplash" texuniform="true" texrepeat="2.5 2.5"/>
    <material name="mat_duvet" texture="tex_duvet" texuniform="true" texrepeat="1.0 1.0"/>
    <material name="mat_home_wood" texture="tex_home_wood" texuniform="true" texrepeat="1.39 1.39" rgba="{f3(wood_tint)}" specular="0.15"/>
    <material name="mat_home_wood_v" texture="tex_home_wood_v" texuniform="true" texrepeat="1.39 1.39" rgba="{f3(wood_tint)}" specular="0.15"/>
    <material name="mat_wall" rgba="0.98 0.98 0.97 1"/>
    <material name="mat_ceiling" rgba="0.96 0.96 0.95 1"/>
    <material name="mat_downlight" rgba="1 1 0.97 1" emission="1"/>
    <material name="mat_wood_front" texture="tex_wood_top" texuniform="true" texrepeat="3 3" rgba="{f3(wood_front)}"/>
    <material name="mat_wood_top" texture="tex_wood_top" texuniform="true" texrepeat="3 3" rgba="{f3(wood_top)}"/>
    <material name="mat_gray_side" texture="tex_gray_side" texuniform="true" texrepeat="6 6"/>
    {oa}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.5 0.5 0.5 1" group="3"/>
    {body}
  </worldbody>
</mujoco>'''
    return xml, spots, placed


def drawer_scene_xml(cab_style, centre, yaw, rand):
    """Open-drawer training scene in the house: the robot base fixed at the origin (kaair_slamtec.xml, as in the office
    scenes) and the house placed so its cabinet (at rand['site'], see CAB_SITES) lands at (centre, yaw) — the pose drawer_ready.build_scene computed
    from the ready TCP and the episode's placement jitter. Used by build_scene when task['scene'] == 'house'."""
    site = rand.get('site', CAB_SITE)
    hx, hy, hyaw = CAB_SITES[site]                    # cabinet in the house
    fa = yaw - hyaw                                   # frame: house cabinet pose -> requested pose
    cf, sf = math.cos(fa), math.sin(fa)
    fx, fy = centre[0] - (cf * hx - sf * hy), centre[1] - (sf * hx + cf * hy)
    xml, _, _ = house_xml('kaair_slamtec.xml', frame=(fx, fy, fa), cab_style=cab_style, light=rand.get('light'),
                          tex_tint=rand.get('tex_tint'), site=site)
    return xml


def build():
    robot_mobile_xml()
    xml, spots, placed = house_xml()
    open(f'{SIM}/mjcf/house.xml', 'w').write(xml)
    layout = dict(rooms={k: dict(x=[v[0], v[1]], y=[v[2], v[3]]) for k, v in ROOMS.items()},
                  walls=[dict(p0=[w[0], w[1]], p1=[w[2], w[3]], doors=[list(d) for d in w[4]]) for w in WALLS],
                  spots=spots, objects=placed, start=dict(x=0.0, y=0.0, yaw_deg=0.0),
                  bounds=dict(x=[BOUNDS[0], BOUNDS[1]], y=[BOUNDS[2], BOUNDS[3]]), cab_site=CAB_SITE,
                  source='docs/draws/floor.drawio (owner, 2026-10-07); looks from the owner photos + home_tour.mp4')
    yaml.safe_dump(layout, open(f'{SIM}/config/house_layout.yaml', 'w'), sort_keys=False, allow_unicode=True)
    print(f'-> {SIM}/mjcf/house.xml   {len(spots)} spots, {len(placed)} objects')
    return layout


def _cam(eye, tgt):
    import mujoco
    v = np.subtract(tgt, eye); dist = float(np.linalg.norm(v))
    c = mujoco.MjvCamera(); c.lookat[:] = tgt; c.distance = dist
    c.azimuth = math.degrees(math.atan2(-v[1], -v[0])) + 180
    c.elevation = math.degrees(math.asin(v[2] / dist))
    return c


# video frame time [s] -> (eye, target) of a person-height camera at roughly the same spot
REAL_VIEWS = {9.9: ((1.3, 0.6, 1.5), (-1.6, 3.6, 0.8)), 18.8: ((1.0, -0.9, 1.45), (-0.3, -4.0, 0.7)),
              26.7: ((-2.0, 0.1, 1.5), (-2.7, 4.0, 0.5)), 44.5: ((4.1, -0.8, 1.5), (3.6, -4.0, 0.5))}


# the owner's photos (sim/assets/real_home/photos, 2026-10-06) -> (eye, target, vertical fov) at about the same spot
PHOTO_VIEWS = {'living': ((0.9, 0.9, 1.35), (-1.6, 3.2, 0.55), 52),
               'kitchen': ((-1.3, -0.4, 1.45), (0.8, -3.9, 0.95), 55),
               'bedroom': ((-1.95, 0.0, 1.45), (-3.9, 3.0, 0.35), 52),
               'closet': ((-3.3, -0.5, 1.45), (-4.3, -2.6, 1.15), 90),
               'pedestal': ((CAB_POSE[0] - 1.4 * math.cos(CAB_POSE[2]), CAB_POSE[1] - 1.4 * math.sin(CAB_POSE[2]), 1.05),
                            (CAB_POSE[0], CAB_POSE[1], 0.3), 85)}


def render(layout):
    import cv2, mujoco
    m = mujoco.MjModel.from_xml_path(f'{SIM}/mjcf/house.xml'); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(300):                                    # let the objects settle
        mujoco.mj_step(m, d)
    out = f'{SIM}/out/house'; os.makedirs(out, exist_ok=True)
    r = mujoco.Renderer(m, 1080, 1920)
    no_ceiling = mujoco.MjvOption(); no_ceiling.geomgroup[2] = 0          # birds-eye views: hide walls + ceiling
    top = _cam((0.5, 0.0, 15.0), (0.5, 0.01, 0.0))
    r.update_scene(d, top, scene_option=no_ceiling)
    cv2.imwrite(f'{out}/top.png', cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    for name, (eye, tgt) in {'living': ((1.5, 0.2, 4.0), (-0.6, 2.6, 0.3)), 'kitchen': ((0.8, 0.2, 4.0), (0.0, -2.8, 0.5)),
                             'bedroom': ((-1.6, -0.2, 4.0), (-3.2, 2.2, 0.4)), 'restroom': ((4.0, 0.2, 4.0), (4.0, -2.6, 0.4))}.items():
        r.update_scene(d, _cam(eye, tgt), scene_option=no_ceiling)
        cv2.imwrite(f'{out}/{name}.png', cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    # video frame | sim at about the same viewpoint (phone: ~65 deg vertical FOV on 16:9)
    video = f'{SIM}/assets/real_home/home_tour.mp4'
    if os.path.exists(video):
        import av
        m.vis.global_.fovy = 62
        r2 = mujoco.Renderer(m, 540, 960)
        want = {int(t * 29.97): t for t in REAL_VIEWS}
        frames = {}
        for i, f in enumerate(av.open(video).decode(video=0)):
            if i in want:
                frames[want[i]] = cv2.resize(cv2.cvtColor(f.to_ndarray(format='rgb24'), cv2.COLOR_RGB2BGR), (960, 540))
            if len(frames) == len(want):
                break
        rows = []
        for t, (eye, tgt) in REAL_VIEWS.items():
            r2.update_scene(d, _cam(eye, tgt))
            rows.append(np.concatenate([frames[t], cv2.cvtColor(r2.render(), cv2.COLOR_RGB2BGR)], 1))
        cv2.imwrite(f'{out}/compare_real.jpg', np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
        r2.close()
    # the owner's photos (2026-10-06) | sim from about the same spot
    pdir = f'{SIM}/assets/real_home/photos'
    if os.path.isdir(pdir):
        tiles = []
        for name, (eye, tgt, fovy) in PHOTO_VIEWS.items():
            ph = cv2.imread(f'{pdir}/{name}.jpg')
            h, w = ph.shape[:2]
            H = 540 if w > h else 900
            W = int(round(w * H / h))
            m.vis.global_.fovy = fovy
            rr = mujoco.Renderer(m, H, W)
            rr.update_scene(d, _cam(eye, tgt))
            sim = cv2.cvtColor(rr.render(), cv2.COLOR_RGB2BGR); rr.close()
            pair = np.concatenate([cv2.resize(ph, (W, H), interpolation=cv2.INTER_AREA), sim], 1)
            cv2.putText(pair, f'{name}: photo | sim', (10, 30), 0, 0.8, (0, 0, 255), 2)
            cv2.imwrite(f'{out}/photo_{name}.jpg', pair, [cv2.IMWRITE_JPEG_QUALITY, 85])
    r.close()
    print('->', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--render', action='store_true')
    a = ap.parse_args()
    lay = build()
    if a.render:
        render(lay)
