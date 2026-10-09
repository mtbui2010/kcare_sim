#!/usr/bin/env python3
"""The owner's test-table objects as simulator objects (real products, real sizes).

    .venv/bin/python scripts/make_real_objects.py      -> assets/ycb/r##_<id>/google_16k/{textured.obj, texture_map.png}

Which objects: the owner's top-down photos of the robot's white test table (Coca-Cola 250 ml slim can, Chilsung
Cider 250 ml can, Maxim T.O.P the Black 275 ml bottle-can, a 500 ml water bottle, Minute Maid orange 350 ml, Haitai
Ace 121 g box, Lotte Lotsand pineapple box, GNC Mega Men bottle, a green supplement jar, a kraft snack pouch,
Galaxy S5, a TV remote, a black cap, brown / beige towels, a striped toy ball, a wooden-handle spoon and fork) and
the earlier side photos (Cass Fresh can, Barista Rules cup, white milk bottle, small water bottle, grey / yellow
cups). Apple, orange and banana stay the YCB scans.

Textures: product photos where a clean front view exists (assets/real_home/objects/tex/*.png; sources and
licences in tex/sources.json -- retailer photos are for this simulation only, not for the public repo), else
procedural (drawn here from the colours in the owner's photos). Meshes are procedural at the product's real size:
cylinders, lathed bottles (the water bottle with its ridges), open cups, boxes (label on the face that is up when
the item lies on a table), a cap, a sphere, spoon and fork. Atlas: left 80 % = label wrapped twice round a body
of revolution, right strip = top / bottom colour patches. props.json: name, material overrides (transparent
bottles). Collision = MuJoCo's convex hull, as for the YCB objects.
"""
import json, math, os

import cv2
import numpy as np

SIM = os.environ.get('SIM_ROOT', '/sim')
TEX = f'{SIM}/assets/real_home/objects/tex'
DST = f'{SIM}/assets/ycb'
N = 48
SIDE_U = 0.8
TOP_UV, BOT_UV = (0.9, 0.75), (0.9, 0.25)
rng = np.random.default_rng(7)


# ── procedural textures (BGR; the atlas resizes them) ────────────────────────
def noise(h, w, base, amp=10, blur=1.0):
    img = np.clip(np.array(base[::-1], float) + rng.normal(0, amp, (h, w, 1)), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), blur) if blur else img


def text_lines(img, x0, y0, x1, y1, n, colour, thick=2):
    for i in range(n):
        y = int(y0 + (y1 - y0) * (i + 0.5) / n)
        cv2.line(img, (x0, y), (int(x0 + (x1 - x0) * rng.uniform(0.5, 1.0)), y), colour[::-1], thick)


def tex_green_jar():
    img = noise(400, 300, (112, 132, 70), 6)
    cv2.rectangle(img, (0, 110), (300, 330), (238, 238, 232)[::-1], -1)          # white label
    text_lines(img, 20, 130, 280, 220, 5, (40, 60, 30), 3)
    for i in range(30):                                                          # barcode
        if rng.random() < 0.6:
            cv2.line(img, (190 + i * 3, 245), (190 + i * 3, 310), (20, 20, 20), 2)
    return img


def tex_kraft_pouch():
    img = noise(500, 360, (196, 168, 128), 9, 1.5)
    text_lines(img, 60, 60, 300, 150, 3, (70, 55, 40), 7)
    cv2.rectangle(img, (90, 190), (270, 300), (235, 225, 210)[::-1], -1)        # the little photo panel
    cv2.ellipse(img, (180, 245), (60, 30), 0, 0, 360, (200, 150, 90)[::-1], -1)
    text_lines(img, 60, 330, 300, 430, 4, (90, 70, 50), 3)
    cv2.rectangle(img, (250, 450), (330, 480), (60, 50, 40)[::-1], -1)
    return img


def tex_towel(colour):
    img = noise(300, 420, colour, 14, 0.8)
    for y in range(0, 300, 6):                                                   # terry rows
        img[y:y + 1] = (img[y:y + 1] * 0.85).astype(np.uint8)
    cv2.line(img, (0, 150), (420, 160), tuple(int(c * 0.7) for c in colour[::-1]), 3)   # the fold
    return img


def tex_striped_ball():
    img = noise(256, 512, (236, 122, 34), 6)
    for k in range(4):
        pts = np.array([(x, int(128 + 70 * math.sin(x / 512 * 4 * math.pi + k * 1.6))) for x in range(0, 520, 8)], np.int32)
        cv2.polylines(img, [pts], False, (15, 20, 25), 7)
    return img


def tex_plain(colour, amp=4):
    return noise(256, 256, colour, amp)


def tex_milk():
    img = noise(500, 300, (240, 240, 235), 3)
    cv2.rectangle(img, (0, 150), (300, 400), (40, 110, 70)[::-1], -1)
    text_lines(img, 30, 190, 270, 360, 5, (240, 240, 235), 5)
    return img


def tex_water_small():
    img = noise(400, 300, (235, 240, 245), 3)
    cv2.rectangle(img, (0, 170), (300, 250), (230, 120, 170)[::-1], -1)
    return img


def tex_steel():
    img = noise(256, 512, (185, 188, 192), 3, 2)
    for x in range(0, 512):                                                      # brushed highlights
        img[:, x] = np.clip(img[:, x].astype(int) + int(35 * math.cos(x / 512 * 4 * math.pi)), 0, 255)
    return img


def tex_check():
    img = np.full((512, 400, 3), 235, np.uint8)
    for k in range(0, 512, 40):
        img[k:k + 12] = (60, 50, 160)                                             # red-ish (BGR) bands
    for k in range(0, 400, 40):
        img[:, k:k + 12] = (img[:, k:k + 12] * 0.45).astype(np.uint8)
    return noise_overlay(img)


def noise_overlay(img, amp=6):
    return np.clip(img.astype(int) + rng.normal(0, amp, img.shape[:2] + (1,)), 0, 255).astype(np.uint8)


def tex_person():
    img = noise(256, 512, (70, 90, 120), 6)                                       # navy clothes
    img[:, 440:] = noise(256, 72, (225, 185, 160), 4)                             # skin (head end)
    return img


RIDGES = [(0, .033)] + [(0.02 + 0.015 * i, .033 if i % 2 == 0 else .0305) for i in range(9)] + \
         [(.15, .033), (.178, .022), (.188, .014), (.19, .016), (.205, .016)]

# id: (name the user says, kind, size, texture (file in TEX or callable), extra)
#   cyl (d, h) | lathe [(z, r)] | cup (d_bottom, d_top, h) | box (L x, D y, H z) | capdome (R, H, brim) | sphere d |
#   spoon / fork (length)
OBJECTS = {
    'r01_coke_can':     ('coke can', 'cyl', (0.055, 0.133), 'coke_can.png', {'top': (200, 200, 205)}),
    'r02_cider_can':    ('green can', 'cyl', (0.055, 0.133), 'cider_can.png', {'top': (200, 200, 205)}),
    'r03_green_jar':    ('green jar', 'lathe', [(0, .031), (.085, .031), (.088, .028), (.108, .028)], tex_green_jar,
                         {'top': (95, 115, 60)}),
    'r04_gnc_bottle':   ('black bottle', 'lathe', [(0, .037), (.105, .037), (.11, .032), (.135, .032)], 'gnc_bottle.png',
                         {'top': (20, 20, 22)}),
    'r05_snack_pouch':  ('snack bag', 'box', (0.12, 0.17, 0.025), tex_kraft_pouch, {'face': 'top'}),
    'r06_beige_towel':  ('beige towel', 'box', (0.15, 0.11, 0.03), lambda: tex_towel((200, 190, 172)), {'face': 'top'}),
    'r07_grey_cup':     ('grey cup', 'cup', (0.062, 0.078, 0.092), lambda: tex_plain((150, 152, 150)), {}),
    'r08_juice_bottle': ('orange juice', 'lathe', [(0, .031), (.12, .031), (.14, .022), (.155, .015), (.157, .016), (.18, .016)],
                         'juice_bottle.png', {'top': (40, 125, 55)}),
    'r09_water_bottle': ('water bottle', 'lathe', RIDGES, 'water_bottle.png', {'top': (60, 110, 200), 'rgba': [1, 1, 1, 0.4]}),
    'r10_brown_towel':  ('brown towel', 'box', (0.16, 0.12, 0.03), lambda: tex_towel((112, 96, 80)), {'face': 'top'}),
    'r11_yellow_cup':   ('yellow cup', 'cup', (0.065, 0.082, 0.10), lambda: tex_plain((240, 200, 90)), {}),
    'r12_ace_box':      ('cracker box', 'box', (0.21, 0.075, 0.04), 'ace_box.png', {'face': 'top', 'side': (200, 30, 40)}),
    'r13_phone':        ('phone', 'box', (0.142, 0.0725, 0.0081), 'phone.png', {'face': 'top', 'side': (25, 30, 45)}),
    'r14_remote':       ('remote control', 'box', (0.19, 0.05, 0.022), 'remote.png', {'face': 'top', 'side': (25, 25, 27)}),
    'r15_cass_can':     ('beer can', 'cyl', (0.066, 0.122), 'cass_can.png', {'top': (200, 200, 205)}),
    'r16_top_coffee':   ('coffee can', 'lathe', [(0, .033), (.092, .033), (.112, .022), (.12, .0195), (.121, .02), (.134, .02)],
                         'top_coffee.png', {'top': (185, 150, 80)}),
    'r17_barista_cup':  ('coffee cup', 'lathe', [(0, .030), (.118, .039), (.119, .041), (.126, .040), (.128, .022)],
                         'barista_cup.png', {'top': (245, 245, 245)}),
    'r18_milk_bottle':  ('milk bottle', 'lathe', [(0, .035), (.12, .035), (.155, .024), (.165, .016), (.168, .018), (.19, .018)],
                         tex_milk, {'top': (235, 235, 230)}),
    'r19_small_water':  ('small water bottle', 'lathe', [(0, .029), (.11, .029), (.135, .019), (.145, .013), (.147, .015), (.165, .015)],
                         tex_water_small, {'top': (230, 150, 190), 'rgba': [1, 1, 1, 0.4]}),
    'r20_pine_box':     ('green snack box', 'box', (0.26, 0.075, 0.05), 'pine_box.png',
                         {'face': 'top', 'side': (60, 150, 60), 'extend': True}),
    'r21_black_cap':    ('black cap', 'capdome', (0.095, 0.085, 0.07), lambda: tex_plain((28, 28, 30), 3), {'top': (28, 28, 30)}),
    'r22_striped_ball': ('striped ball', 'sphere', 0.085, tex_striped_ball, {}),
    'r23_wood_spoon':   ('spoon', 'spoon', 0.19, None, {'top': (150, 98, 58), 'bottom': (195, 195, 200)}),
    'r24_wood_fork':    ('fork', 'fork', 0.19, None, {'top': (150, 98, 58), 'bottom': (195, 195, 200)}),
    # ── the visually-impaired demo (scenario sheet): kitchen, dining, dress room, entrance ──
    'd01_shin_ramyun':  ('shin ramyun', 'box', (0.17, 0.125, 0.035), 'shin_ramyun.png', {'face': 'top', 'side': (200, 30, 30)}),
    'd02_ansung':       ('ansungtangmyun', 'box', (0.17, 0.125, 0.035), 'ansungtangmyun.png', {'face': 'top', 'side': (230, 120, 40)}),
    'd03_chapagetti':   ('chapagetti', 'box', (0.17, 0.125, 0.035), 'chapagetti.png', {'face': 'top', 'side': (150, 175, 90)}),
    'd04_tumbler':      ('tumbler', 'lathe', [(0, .034), (.19, .036), (.192, .034), (.205, .034)], tex_steel, {'top': (60, 60, 64)}),
    'd05_wine_glass':   ('wine glass', 'lathe', [(0, .036), (.004, .036), (.006, .005), (.1, .004), (.115, .03), (.15, .042),
                                                 (.2, .036)], lambda: tex_plain((235, 238, 242), 2), {'rgba': [1, 1, 1, 0.3]}),
    'd06_pot':          ('pot', 'pot', (0.2, 0.11), tex_steel, {'handle': (40, 80, 200)}),
    'd07_beige_bag':    ('beige bag', 'bag', (0.30, 0.13, 0.24), 'beige_bag.png', {'side': (180, 165, 140), 'handle': (40, 40, 45)}),
    'd08_blue_bag':     ('blue bag', 'bag', (0.24, 0.12, 0.22), 'blue_bag.png', {'side': (30, 70, 100), 'handle': (30, 70, 100)}),
    'd09_black_bag':    ('black bag', 'bag', (0.28, 0.13, 0.22), 'black_bag.png', {'side': (25, 25, 27), 'handle': (25, 25, 27)}),
    'd10_shoe_left':    ('black shoe', 'shoe', (0.27, 0.10, 0.10), lambda: tex_plain((30, 30, 32), 5), {'sole': (235, 235, 235)}),
    'd11_shoe_right':   ('black shoe', 'shoe', (0.27, 0.10, 0.10), lambda: tex_plain((30, 30, 32), 5), {'sole': (235, 235, 235)}),
    'd12_white_shirt':  ('white shirt', 'shirt', (0.40, 0.62), lambda: tex_plain((240, 240, 238), 5), {'static': True}),
    'd13_black_shirt':  ('black shirt', 'shirt', (0.40, 0.62), lambda: tex_plain((30, 30, 34), 5), {'static': True}),
    'd14_check_shirt':  ('checked shirt', 'shirt', (0.40, 0.62), tex_check, {'static': True}),
    'd15_person':       ('person', 'person', 1.65, tex_person, {'static': True}),
}


class Mesh:
    def __init__(self):
        self.v, self.vt, self.f = [], [], []

    def vert(self, p, uv):
        self.v.append(tuple(float(c) for c in p)); self.vt.append(uv)
        return len(self.v)

    def quad(self, a, b, c, d):
        self.f += [(a, b, c), (a, c, d)]

    def boxpart(self, c, half, uv):
        """Axis-aligned box (centre, half sizes) in one colour patch."""
        (cx, cy, cz), (hx, hy, hz) = c, half
        V = [self.vert((cx + sx * hx, cy + sy * hy, cz + sz * hz), uv) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for a, b, cc, d in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self.quad(V[a], V[b], V[cc], V[d])

    def write(self, path):
        with open(path, 'w') as fh:
            fh.write('mtllib textured.mtl\nusemtl material_0\n')
            fh.writelines(f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in self.v)
            fh.writelines(f'vt {u:.5f} {v:.5f}\n' for u, v in self.vt)
            fh.writelines(f'f {a}/{a} {b}/{b} {c}/{c}\n' for a, b, c in self.f)


def atlas(label, top, bottom, size=1024, extend=False, aspect=None):
    img = np.zeros((size, size, 3), np.uint8)
    W = int(SIDE_U * size)
    if extend and aspect:            # a short label on a long face: keep its proportions, repeat its right part
        h = label.shape[0]; target = int(h * aspect)
        if label.shape[1] < target:
            tail = label[:, int(label.shape[1] * 0.6):]
            label = np.hstack([label] + [tail] * int(math.ceil((target - label.shape[1]) / tail.shape[1])))[:, :target]
    img[:, :W] = cv2.resize(label, (W, size), interpolation=cv2.INTER_CUBIC)
    img[:size // 2, W:] = top[::-1]
    img[size // 2:, W:] = bottom[::-1]
    return img


def revolve(m, prof, open_top=False, inner=None):
    h = prof[-1][0] or 1.0
    rings = []
    for z, r in prof:
        rings.append([m.vert((r * math.cos(2 * math.pi * k / N), r * math.sin(2 * math.pi * k / N), z),
                             (SIDE_U * ((2 * k / N) % 1.0 if k < N else 1.0), z / h)) for k in range(N + 1)])
    for r0, r1 in zip(rings, rings[1:]):
        for k in range(N):
            m.quad(r0[k], r0[k + 1], r1[k + 1], r1[k])

    def cap(z, r, uv, up):           # own vertices: shared ring vertices carry the side UVs (a swirl on the lid)
        c = m.vert((0, 0, z), uv)
        ring = [m.vert((r * math.cos(2 * math.pi * k / N), r * math.sin(2 * math.pi * k / N), z), uv) for k in range(N + 1)]
        for k in range(N):
            m.f.append((c, ring[k], ring[k + 1]) if up else (c, ring[k + 1], ring[k]))
    cap(prof[0][0], prof[0][1], BOT_UV, False)
    if open_top and inner:
        inn = [[m.vert((r * math.cos(2 * math.pi * k / N), r * math.sin(2 * math.pi * k / N), z), TOP_UV) for k in range(N + 1)]
               for z, r in inner]
        for k in range(N):
            m.quad(rings[-1][k], rings[-1][k + 1], inn[0][k + 1], inn[0][k])
        for r0, r1 in zip(inn, inn[1:]):
            for k in range(N):
                m.quad(r0[k], r0[k + 1], r1[k + 1], r1[k])
        c2 = m.vert((0, 0, inner[-1][0]), TOP_UV)
        for k in range(N):
            m.f.append((c2, inn[-1][k], inn[-1][k + 1]))
    else:
        cap(prof[-1][0], prof[-1][1], TOP_UV, True)


def box(m, L, D, H, face):
    x, y = L / 2, D / 2
    P = lambda i, j, k: ((-x, x)[i], (-y, y)[j], (0, H)[k])
    faces = {'top': [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], 'bottom': [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)],
             'front': [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], 'back': [(1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],
             'left': [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)], 'right': [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]}
    for name, idx in faces.items():
        uvs = [(0, 0), (SIDE_U, 0), (SIDE_U, 1), (0, 1)] if name == face else [BOT_UV] * 4
        m.quad(*[m.vert(P(*q), uv) for q, uv in zip(idx, uvs)])


def sphere(m, d):
    r, nl = d / 2, 24
    rows = [[m.vert((r * math.cos(t) * math.cos(2 * math.pi * k / N), r * math.cos(t) * math.sin(2 * math.pi * k / N),
                     r + r * math.sin(t)), (SIDE_U * k / N, i / nl)) for k in range(N + 1)]
            for i, t in enumerate(math.pi * i / nl - math.pi / 2 for i in range(nl + 1))]
    for r0, r1 in zip(rows, rows[1:]):
        for k in range(N):
            m.quad(r0[k], r0[k + 1], r1[k + 1], r1[k])


def capdome(m, R, Hd, brim):
    rings = [[m.vert((R * math.cos(t) * math.cos(2 * math.pi * k / N), R * math.cos(t) * math.sin(2 * math.pi * k / N),
                      Hd * math.sin(t)), (SIDE_U * k / N, i / 12)) for k in range(N + 1)]
             for i, t in enumerate((math.pi / 2) * i / 12 for i in range(13))]
    for r0, r1 in zip(rings, rings[1:]):
        for k in range(N):
            m.quad(r0[k], r0[k + 1], r1[k + 1], r1[k])
    c = m.vert((0, 0, 0), TOP_UV)
    for k in range(N):
        m.f.append((c, rings[0][k + 1], rings[0][k]))
    pts = [(R * math.cos(a), R * math.sin(a)) for a in np.linspace(-1.2, 1.2, 13)]
    out = [(x + brim * math.cos(math.atan2(y, x)) * (1 - abs(math.atan2(y, x)) / 1.6), y) for x, y in pts]
    for (x0, y0), (x1, y1), (u0, w0), (u1, w1) in zip(pts, pts[1:], out, out[1:]):
        a = m.vert((x0, y0, 0.004), TOP_UV); b = m.vert((x1, y1, 0.004), TOP_UV)
        cc = m.vert((u1, w1, 0.004), TOP_UV); dd = m.vert((u0, w0, 0.004), TOP_UV)
        m.quad(a, b, cc, dd); m.quad(dd, cc, b, a)


def cutlery(m, L, fork):
    """Wooden handle (top patch) along -x, steel neck and head (bottom patch) along +x, lying on z=0."""
    hl = L * 0.55
    m.boxpart((-L / 2 + hl / 2, 0, 0.006), (hl / 2, 0.0095, 0.006), TOP_UV)
    m.boxpart((-L / 2 + hl + 0.015, 0, 0.004), (0.015, 0.004, 0.0015), BOT_UV)
    xh = -L / 2 + hl + 0.03
    if fork:
        m.boxpart((xh + 0.012, 0, 0.004), (0.012, 0.012, 0.0015), BOT_UV)
        for j in range(4):
            m.boxpart((xh + 0.046, -0.009 + j * 0.006, 0.004), (0.022, 0.0012, 0.0012), BOT_UV)
    else:
        nb = 16
        c = m.vert((xh + 0.027, 0, 0.006), BOT_UV)
        ring = [m.vert((xh + 0.027 + 0.027 * math.cos(2 * math.pi * k / nb), 0.02 * math.sin(2 * math.pi * k / nb), 0.003), BOT_UV)
                for k in range(nb + 1)]
        for k in range(nb):
            m.f.append((c, ring[k], ring[k + 1])); m.f.append((c, ring[k + 1], ring[k]))


def pot(m, d, h, handle_uv=BOT_UV):
    """Steel pot (open top) with two side handles in the bottom patch colour."""
    r = d / 2
    revolve(m, [(0, r * 0.95), (0.01, r), (h, r)], open_top=True, inner=[(h, r - 0.003), (0.008, r - 0.003)])
    for sx in (1, -1):
        m.boxpart((sx * (r + 0.03), 0, h - 0.02), (0.03, 0.035, 0.008), handle_uv)


def bag(m, L, D, H):
    """Tote bag: tapered body (front / back = photo) and a handle arch on top (bottom patch colour)."""
    x, y, xt = L / 2, D / 2, L / 2 * 0.85
    pts = {'fbl': (-x, -y, 0), 'fbr': (x, -y, 0), 'ftr': (xt, -y * 0.8, H), 'ftl': (-xt, -y * 0.8, H),
           'bbl': (-x, y, 0), 'bbr': (x, y, 0), 'btr': (xt, y * 0.8, H), 'btl': (-xt, y * 0.8, H)}
    tex = [(0, 0), (SIDE_U, 0), (SIDE_U, 1), (0, 1)]
    def face(keys, uvs):
        m.quad(*[m.vert(pts[k], uv) for k, uv in zip(keys, uvs)])
    face(('fbl', 'fbr', 'ftr', 'ftl'), tex)
    face(('bbr', 'bbl', 'btl', 'btr'), tex)
    for keys in (('bbl', 'fbl', 'ftl', 'btl'), ('fbr', 'bbr', 'btr', 'ftr'), ('ftl', 'ftr', 'btr', 'btl'), ('bbl', 'bbr', 'fbr', 'fbl')):
        face(keys, [TOP_UV] * 4)
    for k in range(9):                                                            # handle arch
        a0, a1 = math.pi * k / 9, math.pi * (k + 1) / 9
        p0 = (-0.08 * math.cos(a0), 0, H + 0.1 * math.sin(a0)); p1 = (-0.08 * math.cos(a1), 0, H + 0.1 * math.sin(a1))
        c = ((p0[0] + p1[0]) / 2, 0, (p0[2] + p1[2]) / 2)
        m.boxpart(c, (max(abs(p1[0] - p0[0]) / 2, 0.006), 0.006, max(abs(p1[2] - p0[2]) / 2, 0.006)), BOT_UV)


def shoe(m, L, W, H):
    """Sneaker: white sole slab (bottom patch) + a half-ellipsoid upper (texture), toe along +x."""
    m.boxpart((0, 0, 0.012), (L / 2, W / 2, 0.012), BOT_UV)
    rows = []
    for i in range(9):
        t = (math.pi / 2) * i / 8
        rows.append([m.vert((L / 2 * 0.97 * math.cos(t) * math.cos(2 * math.pi * k / N),
                             W / 2 * 0.95 * math.cos(t) * math.sin(2 * math.pi * k / N),
                             0.024 + (H - 0.024) * math.sin(t) * (0.75 + 0.25 * math.cos(2 * math.pi * k / N + math.pi))),
                            (SIDE_U * k / N, i / 8)) for k in range(N + 1)])
    for r0, r1 in zip(rows, rows[1:]):
        for k in range(N):
            m.quad(r0[k], r0[k + 1], r1[k + 1], r1[k])


def shirt(m, W, H):
    """A shirt on a hanger, seen flat: textured body panel (front and back) + sleeves, 2 cm thick, hanging from
    z=0 down; a hook on top."""
    bw, t = W / 2 * 0.7, 0.01
    for y, order in ((-t, (0, 1, 2, 3)), (t, (1, 0, 3, 2))):
        P = [(-bw, y, -H), (bw, y, -H), (bw, y, 0), (-bw, y, 0)]
        UV = [(0, 0), (SIDE_U, 0), (SIDE_U, 1), (0, 1)]
        m.quad(*[m.vert(P[i], UV[i]) for i in order])
    for sx in (1, -1):                                                            # sleeves, 1/4 down the shirt
        m.boxpart((sx * (bw + W / 2 * 0.15), 0, -0.13), (W / 2 * 0.15, t, 0.1), (SIDE_U * (0.5 + 0.4 * sx), 0.75))
    m.boxpart((0, 0, 0.02), (0.004, 0.004, 0.03), TOP_UV)


def person(m, L):
    """A person lying on the floor on their back, head at +x (static prop for the fall scenario)."""
    s = L / 1.65
    parts = [((0.0, 0, 0.11), (0.28, 0.17, 0.1)), ((0.62, 0, 0.1), (0.1, 0.08, 0.09)),           # torso, head
             ((-0.55, 0.09, 0.07), (0.32, 0.065, 0.065)), ((-0.55, -0.09, 0.07), (0.32, 0.065, 0.065)),   # legs
             ((0.05, 0.27, 0.06), (0.28, 0.05, 0.05)), ((0.05, -0.27, 0.06), (0.28, 0.05, 0.05))]     # arms
    for (cx, cy, cz), (hx, hy, hz) in parts:
        u = 0.6 if cx < 0.5 else 0.79
        m.boxpart((cx * s, cy * s, cz * s), (hx * s, hy * s, hz * s), (u, 0.5))


def main():
    made = {}
    for oid, (name, kind, size, tex, ex) in OBJECTS.items():
        if isinstance(tex, str):
            label = cv2.imread(f'{TEX}/{tex}')
            if label is None:
                print('missing texture', tex); continue
        else:
            label = tex() if callable(tex) else np.zeros((8, 8, 3), np.uint8)
        mean = tuple(int(v) for v in cv2.mean(label)[:3][::-1])
        top = ex.get('top', ex.get('side', mean))
        bottom = ex.get('bottom', ex.get('side', top if kind in ('cyl', 'box', 'spoon', 'fork') else mean))   # bottle bottoms: body colour
        m = Mesh()
        if kind == 'cyl':
            d, h = size; revolve(m, [(0, d / 2), (h, d / 2)])
        elif kind == 'lathe':
            revolve(m, size)
        elif kind == 'cup':
            db, dt, h = size
            revolve(m, [(0, db / 2), (h, dt / 2)], open_top=True, inner=[(h, dt / 2 - 0.002), (0.006, db / 2 - 0.002)])
            top = bottom = tuple(int(v * 0.85) for v in mean)
        elif kind == 'box':
            box(m, *size, ex.get('face', 'top'))
        elif kind == 'capdome':
            capdome(m, *size)
        elif kind == 'sphere':
            sphere(m, size)
        elif kind in ('spoon', 'fork'):
            cutlery(m, size, kind == 'fork')
        elif kind == 'pot':
            pot(m, *size); bottom = ex['handle']
        elif kind == 'bag':
            bag(m, *size); bottom = ex['handle']
        elif kind == 'shoe':
            shoe(m, *size); bottom = ex['sole']
        elif kind == 'shirt':
            shirt(m, *size); top = (230, 230, 230)
        elif kind == 'person':
            person(m, size)
        d = f'{DST}/{oid}/google_16k'
        os.makedirs(d, exist_ok=True)
        m.write(f'{d}/textured.obj')
        open(f'{d}/textured.mtl', 'w').write('newmtl material_0\nmap_Kd texture_map.png\n')
        aspect = size[0] / size[1] if kind == 'box' else None
        cv2.imwrite(f'{d}/texture_map.png', atlas(label, top, bottom, extend=ex.get('extend', False), aspect=aspect))
        props = {'name': name, 'texture': tex if isinstance(tex, str) else 'procedural'}
        if 'rgba' in ex:
            props['rgba'] = ex['rgba']
        if ex.get('static'):
            props['static'] = True
        json.dump(props, open(f'{DST}/{oid}/props.json', 'w'))
        made[oid] = name
        print(f'{oid:20s} {name:20s} {kind:8s} {len(m.v):5d} verts  texture {props["texture"]}')
    json.dump(made, open(f'{os.path.dirname(TEX)}/objects.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
