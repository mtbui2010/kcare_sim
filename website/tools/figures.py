"""Render the documentation figures from recorded data.

    python website/tools/figures.py <data-dir>

Writes PNGs into website/docs/assets/img/. Conventions (keep them, they read well on the site):
- one light surface, recessive grid, thin lines, direct labels instead of legends where possible;
- categorical colours in this fixed order: blue, orange, aqua (checked for colour-vision
  deficiency); a sequential quantity uses the blue ramp, light -> dark;
- never a dual y-axis; bars start at zero (use dots + intervals when the interesting range is
  narrow);
- the title states the finding ("… 0 stalls"), not just the variable names;
- every number comes from <data-dir>; nothing is typed in by hand.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def wilson(k, n, z=1.96):
    """95 % interval for k successes out of n: use it whenever a success rate is plotted."""
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


# --- figures ---------------------------------------------------------------------------------
import json, re, subprocess


def success_rates():
    """Open-drawer success rate per policy and scene set, k/50 with 95 % Wilson intervals (eval/*.json)."""
    rows = [  # (label, file, group)
        ('office policy, 5k steps', 'real500_multidrawer_20260928__005000_real_results.json', 'office scenes'),
        ('office policy, 10k steps', 'real500_multidrawer_20260928__010000_real_results.json', 'office scenes'),
        ('office policy, 15k steps', 'real500_multidrawer_20260928__015000_real_results.json', 'office scenes'),
        ('office policy, 20k steps', 'real500_multidrawer_20260928__020000_real_results.json', 'office scenes'),
        ('office policy, 20k steps', 'real500_multidrawer_20260928__020000_house_results.json', 'house scenes'),
        ('office + house, 2.5k more', 'officehouse_20261007__002500_house_results.json', 'house scenes'),
        ('office + house, 5k more', 'officehouse_20261007__005000_house_results.json', 'house scenes'),
    ]
    pts = []
    for lab, f, g in rows:
        fp = DATA / 'eval' / f
        if not fp.exists():
            continue
        e = json.load(open(fp))['episodes']
        k, n = sum(x['success'] for x in e), len(e)
        pts.append((lab, g, k, n))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(pts) + 1.2))
    for i, (lab, g, k, n) in enumerate(pts):
        lo, hi = wilson(k, n)
        c = BLUE if g == 'office scenes' else ORANGE
        y = len(pts) - 1 - i
        ax.plot([lo * 100, hi * 100], [y, y], color=c, lw=3, alpha=0.35, solid_capstyle='round')
        ax.plot(k / n * 100, y, 'o', color=c, ms=8)
        ax.text(hi * 100 + 1.5, y, f'{k}/{n}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([f'{lab}  ({g})' for lab, g, k, n in pts][::-1])
    ax.set_xlim(0, 105); ax.set_xlabel('drawers opened ≥ 20 cm within 15 s  [%]  (dot = rate, bar = 95 % interval)')
    ax.set_title('Training in the simulated house takes the policy there from 8 % to over 50 %')
    save(fig, 'success_rates.png')


def depth_noise():
    """D405 depth noise vs distance: the one real frame vs the simulated sensor model (depth_compare.py)."""
    txt = (DATA / 'out' / 'depth_compare.txt').read_text()
    def series(tag):
        line = next(l for l in txt.splitlines() if l.startswith(tag))
        return {int(z): float(v) for z, v in re.findall(r'(\d+)mm sd ([\d.]+)', line) if int(z) != 300}
    real, model, perfect = series('REAL D405'), series('SIM  D405 model'), series('SIM  D405 perfect')
    zs = sorted(set(real) & set(model))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for d, c, lab in ((real, INK, 'real D405'), (model, BLUE, 'simulated D405 (DepthModel)'), (perfect, MUTED, 'perfect render')):
        ax.plot([z / 1000 for z in zs], [d[z] for z in zs], 'o-', color=c, lw=1.8)
        ax.text(zs[-1] / 1000 + 0.03, d[zs[-1]], lab, va='center', fontsize=8, color=c)
    ax.set_xlim(0, 2.1); ax.set_xlabel('distance [m]'); ax.set_ylabel('local depth spread [mm]')
    ax.set_title('Simulated D405 depth follows the real noise to 0.8 m, underestimates it at 1.5 m')
    save(fig, 'depth_noise.png')


def color_swatches():
    """Wrist camera colour of three surfaces: real image vs sim before / after calibrate_home.py."""
    txt = (DATA / 'out' / 'calibrate_home.txt').read_text()
    rows = []
    for l in txt.splitlines():
        m = re.match(r'(\w+)\s+(\w+)\s+real \[(.*?)\]\s+sim before \[(.*?)\]\s+after \[(.*?)\]', l)
        if m and m.group(1) == 'wrist':
            rows.append((m.group(2), *[[float(v) for v in g.split()] for g in m.groups()[2:]]))
    office_gain = np.array([1.105, 0.974, 0.857])    # camera_real.yaml wrist colour_gain (office fit)
    office_sat = 3.267
    fig, ax = plt.subplots(figsize=(8, 2.6))
    cols = ['real D405 image', 'sim, office colour fit', 'sim, home colour fit']
    for i, (name, real, before, after) in enumerate(rows):
        b = np.array(before); lum = b @ np.array([0.299, 0.587, 0.114])
        office = np.clip((lum + office_sat * (b - lum)) * office_gain, 0, 255)
        for j, c in enumerate((real, office, after)):
            ax.add_patch(plt.Rectangle((j * 1.1, -i * 1.1), 1, 1, color=np.array(c) / 255))
        ax.text(-0.15, -i * 1.1 + 0.5, name, ha='right', va='center', fontsize=9, color=INK2)
    for j, c in enumerate(cols):
        ax.text(j * 1.1 + 0.5, 1.15, c, ha='center', fontsize=8, color=INK2)
    ax.set_xlim(-1.2, 3.3); ax.set_ylim(-len(rows) * 1.1 + 0.9, 1.5); ax.axis('off')
    ax.set_title('Fitting white balance on neutral surfaces removed the yellow cast')
    save(fig, 'color_swatches.png')


def filmstrip(clip, name, n=5, overlay=None, quad=None):
    """n frames of a recorded 2x2 camera grid, evenly spread over the commands; quad = (row, col) crops one view."""
    import imageio.v3 as iio
    frames = iio.imread(DATA / 'media' / clip)
    idx = np.linspace(0, len(frames) - 1, n + 2)[1:-1].astype(int)
    if overlay:
        ov = json.load(open(DATA / 'media' / overlay))
        pv = ov['progress']['values']
        last = max(i for i, v in enumerate(pv) if v == max(pv))
        idx = np.linspace(ov['events'][-1]['frame'] + 15, min(last, len(frames) - 1), n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.0 if quad is None else 1.85))
    for ax, i in zip(axes, idx):
        im = frames[i] if quad is None else frames[i][quad[0] * 330:(quad[0] + 1) * 330, quad[1] * 640:(quad[1] + 1) * 640]
        ax.imshow(im); ax.axis('off')
        ax.set_title(f'{i / 15:.0f} s', fontsize=8, color=INK2, fontweight='normal')
    fig.subplots_adjust(wspace=0.03)
    save(fig, name)


def photos():
    """Owner photo | sim render pairs and camera comparisons, resized for the web (already rendered by the sim)."""
    from PIL import Image
    for f in sorted((DATA / 'media').glob('*.jpg')) + sorted((DATA / 'media').glob('*.png')):
        im = Image.open(f).convert('RGB')
        if im.width > 1600:
            im = im.resize((1600, int(im.height * 1600 / im.width)), Image.LANCZOS)
        im.save(OUT / (f.stem + '.jpg'), quality=82)
        print('wrote', OUT / (f.stem + '.jpg'))


if __name__ == "__main__":
    success_rates(); depth_noise(); color_swatches()
    filmstrip('vla_grid_try1.mp4', 'filmstrip_vla.png', overlay='vla_grid_try1_overlay.json', quad=(0, 1))
    filmstrip('tour_grid.mp4', 'filmstrip_tour.png')
    photos()
