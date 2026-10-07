#!/usr/bin/env python3
"""Which scene factors go with policy failures?

    python3 scripts/analyze_eval.py /nas/checkpoints/<run>/eval/005000_results.json [more results.json ...]
      -> /sim/out/analysis/<run>_<step>_factors.{md,json,png}
         /sim/out/analysis/<run>_<step>_failures.png   (first head|wrist frame of every failed scene)

eval_policy.py builds each scene from randomize.sample(default_rng(seed)), so
the exact randomization of every evaluated episode is recovered from its seed.
For each factor: mean over successes vs failures, point-biserial correlation
with success, and success rate in the lower / upper half of the factor. With
only a few failures these are hints, not proof — look at the failure mosaic.
"""
import json, math, os, sys

import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import randomize  # noqa: E402


def lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def factors(rand):
    front, handle = lum(rand['front_rgba']), lum(rand['handle_rgba'])
    return {
        'cabinet yaw [deg]': math.degrees(rand['yaw']),
        'lateral offset [cm]': rand['lateral'] * 100,
        'depth offset [cm]': rand['depth'] * 100,
        'drawer pre-open [cm]': rand['drawer0'] * 100,
        'head pan jitter [deg]': math.degrees(rand['q_extra']['head_joint1']),
        'head tilt jitter [deg]': math.degrees(rand['q_extra']['head_joint2']),
        'cabinet brightness': lum(rand['body_rgba']),
        'handle vs front contrast': abs(handle - front),
        'floor brightness': lum(rand['floor_rgb'][0]),
        'light diffuse': rand['light']['diffuse'],
        'light ambient': rand['light']['ambient'],
    }


def analyze(path, rcfg):
    r = json.load(open(path))
    eps = r['episodes']
    rows = [dict(seed=e['seed'], success=e['success'], opened_cm=e['opened'] * 100,
                 **factors(randomize.sample(np.random.default_rng(e['seed']), rcfg))) for e in eps]
    ok = np.array([x['success'] for x in rows])
    names = [k for k in rows[0] if k not in ('seed', 'success', 'opened_cm')]
    stats = []
    for n in names:
        v = np.array([x[n] for x in rows])
        corr = float(np.corrcoef(v, ok)[0, 1]) if 0 < ok.sum() < len(ok) else float('nan')
        med = np.median(v)
        stats.append(dict(factor=n, mean_success=float(v[ok].mean()) if ok.any() else float('nan'),
                          mean_fail=float(v[~ok].mean()) if (~ok).any() else float('nan'), corr=corr,
                          rate_low=float(ok[v <= med].mean()), rate_high=float(ok[v > med].mean())))
    stats.sort(key=lambda s: -abs(s['corr']) if not math.isnan(s['corr']) else 0)
    return r, rows, stats


def render_failures(rows, task, cfg, out_png, max_n=12):
    import cv2, mujoco
    import check_sim as cs, drawer_ready as dr
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    rcfg = yaml.safe_load(open(f'{SIM}/config/randomization.yaml'))
    tiles = []
    for x in [x for x in rows if not x['success']][:max_n]:
        rand = randomize.sample(np.random.default_rng(x['seed']), rcfg)
        q = dr.ready_q(task, task['mode'], cfg['home'])
        xml, *_ = dr.build_scene(task, ref.fk(q)['tool_tcp_link'], rand=rand)
        path = f'{SIM}/mjcf/_analysis_scene.xml'
        open(path, 'w').write(xml)
        m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
        for j, dv in rand['q_extra'].items():
            q[j] += dv
        cs.set_q(m, d, q, cfg); mujoco.mj_forward(m, d)
        rh = mujoco.Renderer(m, 720, 1280); rh.update_scene(d, 'head_rgb')
        head = cv2.resize(rh.render(), (424, 240), interpolation=cv2.INTER_AREA); rh.close()
        rw = mujoco.Renderer(m, 480, 848); rw.update_scene(d, 'wrist_rgb')
        wrist = cv2.resize(rw.render(), (424, 240), interpolation=cv2.INTER_AREA); rw.close()
        tile = np.concatenate([head, wrist], 1).copy()
        cv2.putText(tile, f"seed ..{x['seed'] % 1000:03d}  opened {x['opened_cm']:.1f} cm", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        tiles.append(tile)
    os.remove(f'{SIM}/mjcf/_analysis_scene.xml') if tiles else None
    if not tiles:
        return False
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    import imageio
    imageio.imwrite(out_png, grid)
    return True


def main():
    rcfg = yaml.safe_load(open(f'{SIM}/config/randomization.yaml'))
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
    out_dir = f'{SIM}/out/analysis'
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for path in sys.argv[1:]:
        r, rows, stats = analyze(path, rcfg)
        run = path.split('/checkpoints/')[1].split('/')[0]
        step = os.path.basename(path).split('_')[0]
        tag = f'{out_dir}/{run}_{step}'
        n_fail = sum(not x['success'] for x in rows)
        lines = [f"# {run} step {step}: {r['success_rate'] * 100:.0f}% success, {n_fail} failures / {len(rows)}\n",
                 '| factor | mean (success) | mean (fail) | corr with success | success rate low half | high half |',
                 '|---|---|---|---|---|---|']
        for s in stats:
            lines.append(f"| {s['factor']} | {s['mean_success']:.2f} | {s['mean_fail']:.2f} | {s['corr']:+.2f} | "
                         f"{s['rate_low'] * 100:.0f}% | {s['rate_high'] * 100:.0f}% |")
        open(tag + '_factors.md', 'w').write('\n'.join(lines) + '\n')
        json.dump(dict(rows=rows, stats=stats), open(tag + '_factors.json', 'w'), indent=1)
        print('\n'.join(lines))

        top = [s['factor'] for s in stats[:4]]
        fig, axs = plt.subplots(1, len(top), figsize=(4 * len(top), 3.2), sharey=True)
        for ax, n in zip(np.atleast_1d(axs), top):
            v = np.array([x[n] for x in rows]); o = np.array([x['opened_cm'] for x in rows])
            okm = np.array([x['success'] for x in rows])
            ax.scatter(v[okm], o[okm], s=18, c='#2a7fbf', label='thành công')
            ax.scatter(v[~okm], o[~okm], s=26, c='#d9534f', marker='x', label='thất bại')
            ax.axhline(20, color='gray', lw=0.8, ls='--')
            ax.set_xlabel(n); ax.grid(alpha=0.3)
        np.atleast_1d(axs)[0].set_ylabel('ngăn mở [cm]'); np.atleast_1d(axs)[0].legend(fontsize=8)
        fig.suptitle(f"{run} — bước {int(step)}: {r['success_rate'] * 100:.0f}% thành công", fontsize=10)
        fig.tight_layout(); fig.savefig(tag + '_factors.png', dpi=130); plt.close(fig)
        if render_failures(rows, task, cfg, tag + '_failures.png'):
            print('->', tag + '_failures.png')
        print('->', tag + '_factors.{md,json,png}\n')


if __name__ == '__main__':
    main()
