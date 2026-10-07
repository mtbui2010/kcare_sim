"""Parametric drawer cabinets for the "open drawer" task.

A cabinet is a static carcass (sides, top, bottom, back, dividers) with one
body per drawer on a slide joint that pulls toward the robot.

    body_xml, target_drawer, height = cabinet_xml(STYLES['A'], pos=(0.78, 0, 0), yaw=0)

Local frame: the cabinet faces local -x, so a drawer opens along local -x.
`pos` is the centre of the carcass footprint on the floor, `yaw` [rad] turns it
about z (yaw=-pi/2 makes it face world +y, i.e. a cabinet on the robot's -y
side, kcare 'left' mode).

Style A is the real kcare cabinet (tasks.py ENV 'drawer@living room'): top
surface 0.565 m, gray handle at 0.415 m.
"""

T = 0.018          # panel thickness
GAP = 0.004        # clearance around a drawer front

# name -> geometry. drawers: list of front heights, bottom -> top.
STYLES = {
    'A': dict(label='A  nightstand 3 drawers, bar handle (kcare drawer@living room)',
              width=0.46, depth=0.42, plinth=0.05, drawers=[0.166, 0.166, 0.166], target=2,
              top_extra=0.0, handle='bar', handle_w=0.14, handle_dz=-0.05,   # handle at 0.415 m
              body_rgba=(0.93, 0.92, 0.88, 1), front_rgba=(0.95, 0.94, 0.90, 1)),
    # The real cabinet at drawer@living room as seen on 2026-09-28 (wrist depth + intrinsics; re-measure
    # with a tape): mobile pedestal, deep file drawer at the bottom, middle drawer = the one
    # approach_drawer aims at (handle ~0.436 m), shallow top drawer with a lock. Wood-textured
    # front/top, dark grey sides, knurled aluminium bar handles on dark grey posts.
    'A_real': dict(label='A_real  kcare pedestal, middle drawer',
              width=0.40, depth=0.50, plinth=0.04, drawers=[0.34, 0.12, 0.065], target=1,
              top_extra=0.0, handle='bar', handle_w=0.135, handle_r=0.0075,
              handle_dz=[0.106, -0.004, 0.0025], lock_on=2,
              body_rgba=(0.24, 0.25, 0.27, 1), front_rgba=(0.80, 0.72, 0.62, 1),
              handle_rgba=(0.78, 0.79, 0.80, 1), post_rgba=(0.22, 0.23, 0.25, 1),
              front_material='mat_wood_front', top_material='mat_wood_top', body_material='mat_gray_side'),
    'B': dict(label='B  chest 4 drawers (3rd from bottom), knob handle',
              width=0.62, depth=0.45, plinth=0.08, drawers=[0.21, 0.21, 0.21, 0.21], target=2,
              top_extra=0.03, handle='knob', handle_w=0.0,
              body_rgba=(0.62, 0.45, 0.30, 1), front_rgba=(0.70, 0.52, 0.35, 1)),
    'C': dict(label='C  metal file cabinet 3 drawers, flat pull',
              width=0.42, depth=0.52, plinth=0.03, drawers=[0.21, 0.21, 0.21], target=2,
              top_extra=0.01, handle='flat', handle_w=0.16,
              body_rgba=(0.55, 0.57, 0.60, 1), front_rgba=(0.62, 0.64, 0.67, 1)),
    'D': dict(label='D  kitchen base cabinet, top drawer, long bar',
              width=0.80, depth=0.55, plinth=0.12, drawers=[0.40, 0.16], target=1,
              top_extra=0.0, countertop=0.035, handle='bar', handle_w=0.36, fixed=[0],
              body_rgba=(0.30, 0.34, 0.38, 1), front_rgba=(0.33, 0.37, 0.42, 1)),
    # The white 3-drawer chest in the KETI demo apartment's bedroom (home tour video 2026-09-29):
    # white fronts, two small gold knobs per drawer, short gold legs (approximated by the plinth).
    'W_chest': dict(label='W_chest  white bedroom chest 3 drawers, gold knobs',
              width=0.80, depth=0.42, plinth=0.10, drawers=[0.20, 0.20, 0.20], target=2,
              top_extra=0.01, handle='knob', handle_w=0.0,
              body_rgba=(0.93, 0.93, 0.91, 1), front_rgba=(0.96, 0.96, 0.94, 1),
              handle_rgba=(0.80, 0.66, 0.38, 1),
              feet=dict(kind='leg', inset=0.05, r=0.022, r_tip=0.012, rgba=(0.88, 0.80, 0.66, 1),
                        tip_rgba=(0.80, 0.66, 0.38, 1))),
    # The A_real pedestal as it stands in the KETI apartment. The robot's own cameras there (2026-10-06,
    # sim/assets/real_home/robot_cams/*_drawer.jpg) show the office look: warm wood front + top, DARK GREY sides
    # (the owner's phone photo had suggested oak sides — wrong). Adds what A_real models as a plinth: black
    # twin-wheel casters under a recessed base, top slab overhanging 1.5 cm.
    'A_home': dict(label='A_home  kcare pedestal at home (A_real look + casters)',
              width=0.40, depth=0.50, plinth=0.04, drawers=[0.34, 0.12, 0.065], target=1,
              top_extra=0.0, handle='bar', handle_w=0.135, handle_r=0.0075,
              handle_dz=[0.106, -0.004, 0.0025], lock_on=2, top_overhang=0.015,
              body_rgba=(0.24, 0.25, 0.27, 1), front_rgba=(0.80, 0.72, 0.62, 1),
              handle_rgba=(0.78, 0.79, 0.80, 1), post_rgba=(0.72, 0.73, 0.74, 1),
              front_material='mat_wood_front', top_material='mat_wood_top', body_material='mat_gray_side',
              feet=dict(kind='caster', inset=0.045, r=0.028, rgba=(0.06, 0.06, 0.07, 1))),
}
HANDLE_RGBA = (0.55, 0.56, 0.58, 1)   # "gray handle" (kcare vla_drawer_open prompt)


def _box(name, pos, size, rgba, extra='', material=None):
    p = ' '.join(f'{v:.4f}' for v in pos)
    s = ' '.join(f'{v:.4f}' for v in size)
    look = f'material="{material}"' if material else 'rgba="' + ' '.join(f'{v:.3f}' for v in rgba) + '"'
    return f'<geom name="{name}" type="box" pos="{p}" size="{s}" {look} {extra}/>'


def _handle(prefix, x, z, kind, w, radius=0.008, rgba=None, post_rgba=None):
    """Handle geoms in the drawer frame; x = front face (drawer local, -x out)."""
    g, r = [], rgba or HANDLE_RGBA
    pr = post_rgba or r
    fr = 'friction="1.2 0.01 0.0002" condim="4"'
    if kind == 'bar':
        off = 0.035
        for i, y in enumerate((-w / 2 + 0.01, w / 2 - 0.01)):
            g.append(_box(f'{prefix}_post{i}', (x - off / 2, y, z), (off / 2, 0.006, 0.006), pr, fr))
        g.append(f'<geom name="{prefix}_handle" type="capsule" fromto="{x - off:.4f} {-w / 2:.4f} {z:.4f} '
                 f'{x - off:.4f} {w / 2:.4f} {z:.4f}" size="{radius}" rgba="{" ".join(f"{v:.3f}" for v in r)}" {fr}/>')
    elif kind == 'knob':
        for i, y in enumerate((-0.12, 0.12)):
            g.append(f'<geom name="{prefix}_stem{i}" type="cylinder" fromto="{x:.4f} {y} {z:.4f} {x - 0.025:.4f} {y} {z:.4f}" '
                     f'size="0.007" rgba="{" ".join(map(str, r))}" {fr}/>')
            g.append(f'<geom name="{prefix}_knob{i}" type="sphere" pos="{x - 0.03:.4f} {y} {z:.4f}" size="0.017" '
                     f'rgba="{" ".join(map(str, r))}" {fr}/>')
    elif kind == 'flat':
        # pull plate standing off the front with a downward lip to hook fingers behind
        g.append(_box(f'{prefix}_plate', (x - 0.012, 0, z + 0.012), (0.012, w / 2, 0.003), r, fr))
        g.append(_box(f'{prefix}_lip', (x - 0.024, 0, z), (0.002, w / 2, 0.015), r, fr))
        for i, y in enumerate((-w / 2 + 0.004, w / 2 - 0.004)):
            g.append(_box(f'{prefix}_end{i}', (x - 0.012, y, z + 0.004), (0.012, 0.003, 0.010), r, fr))
    return g


def cabinet_xml(st, pos=(0.78, 0.0, 0.0), yaw=0.0, name='cab'):
    import math
    W, D = st['width'], st['depth']
    fronts = st['drawers']
    H = st['plinth'] + sum(fronts) + st.get('top_extra', 0.0) + T
    cx, cy, _ = pos
    br, fr = st['body_rgba'], st['front_rgba']
    xf = -D / 2                              # carcass front plane (local x)
    g = []
    # carcass (static, local frame at footprint centre)
    bm, fm, tm = st.get('body_material'), st.get('front_material'), st.get('top_material')
    # sides + back stop under the top slab (a shared top face z-fights: grey blotches on the wood top)
    Hs = H - T
    g.append(_box(f'{name}_left', (0, -W / 2 + T / 2, Hs / 2), (D / 2, T / 2, Hs / 2), br, material=bm))
    g.append(_box(f'{name}_right', (0, W / 2 - T / 2, Hs / 2), (D / 2, T / 2, Hs / 2), br, material=bm))
    g.append(_box(f'{name}_back', (D / 2 - T / 2, 0, Hs / 2), (T / 2, W / 2 - T, Hs / 2), br, material=bm))
    ov = st.get('top_overhang', 0.0)
    g.append(_box(f'{name}_top', (0, 0, H - T / 2), (D / 2 + ov, W / 2 + ov, T / 2), fr if tm else br, material=tm))
    ft, P = st.get('feet'), st['plinth']
    if not ft:
        g.append(_box(f'{name}_plinth', (0.02, 0, P / 2), (D / 2 - 0.02, W / 2 - T, P / 2), br, material=bm))
    else:
        # raised carcass: sides stop at the base panel, the feet carry it
        g[0] = _box(f'{name}_left', (0, -W / 2 + T / 2, (Hs + P) / 2), (D / 2, T / 2, (Hs - P) / 2), br, material=bm)
        g[1] = _box(f'{name}_right', (0, W / 2 - T / 2, (Hs + P) / 2), (D / 2, T / 2, (Hs - P) / 2), br, material=bm)
        g[2] = _box(f'{name}_back', (D / 2 - T / 2, 0, (Hs + P) / 2), (T / 2, W / 2 - T, (Hs - P) / 2), br, material=bm)
        g.append(_box(f'{name}_base', (0, 0, P + T / 4), (D / 2, W / 2, T / 4), br, material=bm))
        k = ft['inset']
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            x, y = sx * (D / 2 - k), sy * (W / 2 - k)
            c = ' '.join(f'{v:.3f}' for v in ft['rgba'])
            if ft['kind'] == 'caster':       # twin-wheel caster: a black wheel pair under a small plate
                r = min(ft['r'], P / 2)
                g.append(f'<geom name="{name}_caster{i}" type="cylinder" fromto="{x:.4f} {y - 0.022:.4f} {r:.4f} '
                         f'{x:.4f} {y + 0.022:.4f} {r:.4f}" size="{r:.4f}" rgba="{c}"/>')
                g.append(_box(f'{name}_caster{i}_plate', (x, y, P - 0.004), (0.022, 0.026, 0.004), ft['rgba']))
            else:                            # tapered leg with a metal tip
                tip = ' '.join(f'{v:.3f}' for v in ft['tip_rgba'])
                g.append(f'<geom name="{name}_leg{i}" type="capsule" fromto="{x:.4f} {y:.4f} {0.03:.4f} {x:.4f} {y:.4f} {P:.4f}" '
                         f'size="{ft["r"]:.4f}" rgba="{c}"/>')
                g.append(f'<geom name="{name}_tip{i}" type="cylinder" fromto="{x:.4f} {y:.4f} 0 {x:.4f} {y:.4f} 0.03" '
                         f'size="{ft["r_tip"]:.4f}" rgba="{tip}"/>')
    if st.get('countertop'):
        ct = st['countertop']
        g.append(_box(f'{name}_counter', (-0.02, 0, H + ct / 2), (D / 2 + 0.03, W / 2 + 0.02, ct / 2), (0.85, 0.83, 0.78, 1)))
    drawers, z = [], st['plinth']
    for i, h in enumerate(fronts):
        g.append(_box(f'{name}_shelf{i}', (0, 0, z + T / 4), (D / 2 - T, W / 2 - T, T / 4), br))
        zc = z + h / 2
        if i in st.get('fixed', []):         # a door panel, not a drawer
            g.append(_box(f'{name}_door{i}', (xf - T / 2, 0, zc), (T / 2, W / 2 - GAP, h / 2 - GAP), fr))
        else:
            drawers.append((i, zc, h))
        z += h

    bodies = []
    for i, zc, h in drawers:
        dn = f'{name}_drawer{i}'
        inner_w, inner_d = W / 2 - T - GAP, D / 2 - T - 0.01
        box_h = max(h / 2 - GAP - 0.02, 0.012)       # shallow drawers (6.5 cm pencil drawer) keep a real box
        wall = max(box_h - 0.02, 0.006)                 # side/rear walls stand on the drawer bottom (-box_h)
        dg = [
            _box(f'{dn}_front', (xf - T / 2, 0, 0), (T / 2, W / 2 - GAP, h / 2 - GAP), fr, material=fm),
            _box(f'{dn}_bottom', (xf + inner_d, 0, -box_h), (inner_d, inner_w, 0.004), br),
            _box(f'{dn}_sideL', (xf + inner_d, -inner_w + 0.005, wall - box_h), (inner_d, 0.005, wall), br),
            _box(f'{dn}_sideR', (xf + inner_d, inner_w - 0.005, wall - box_h), (inner_d, 0.005, wall), br),
            _box(f'{dn}_rear', (xf + 2 * inner_d - 0.005, 0, wall - box_h), (0.005, inner_w, wall), br),
        ]
        hz = h / 2 - 0.04 if st['handle'] == 'flat' else 0.0   # flat pull sits under the top edge
        dz = st.get('handle_dz', 0.0)
        hz += dz[i] if isinstance(dz, (list, tuple)) else dz        # per-drawer handle height offsets
        dg += _handle(dn, xf - T, hz, st['handle'], st['handle_w'], st.get('handle_r', 0.008),
                      st.get('handle_rgba'), st.get('post_rgba'))
        if st.get('lock_on') == i:                                    # key lock, left of the handle
            dg.append(f'<geom name="{dn}_lock" type="cylinder" fromto="{xf - T:.4f} {-W * 0.34:.4f} {hz:.4f} '
                      f'{xf - T - 0.004:.4f} {-W * 0.34:.4f} {hz:.4f}" size="0.009" rgba="0.75 0.75 0.76 1"/>')
        travel = 0.8 * D
        bodies.append(
            f'<body name="{dn}" pos="0 0 {zc:.4f}">'
            f'<joint name="{dn}_slide" type="slide" axis="-1 0 0" range="0 {travel:.3f}" damping="5" frictionloss="2"/>'
            f'<inertial pos="{xf + inner_d:.3f} 0 0" mass="1.5" diaginertia="0.02 0.02 0.02"/>'
            + ''.join(dg) + '</body>')

    quat = f'{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}'
    body = (f'<body name="{name}" pos="{cx:.4f} {cy:.4f} 0" quat="{quat}">' + ''.join(g) + ''.join(bodies) + '</body>')
    target = f'{name}_drawer{st["target"]}'
    return body, target, H
