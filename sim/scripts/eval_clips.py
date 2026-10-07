#!/usr/bin/env python3
"""Labelled clips of a checkpoint: a few successes and failures, prompt + outcome on every frame.

    .venv/bin/python scripts/eval_clips.py --ckpt <run>/checkpoints/005000/pretrained_model \
        --ok 3 --fail 3 --max 24 [--rand-config randomization_real.yaml --camera-model]
      -> /sim/out/eval_clips/<run>_<step>_{ok,fail}_<drawer>_seed<seed>.mp4

Runs the eval scenes (same seeds as eval_policy.py) in order and keeps the first
--ok successes (preferring a different drawer each time) and --fail failures.
The policy samples noise, so an outcome can differ from the one in <step>_results.json.
"""
import argparse, os, sys

import cv2
import imageio
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import check_sim as cs, drawer_ready as dr, expert_open_drawer as ex, randomize  # noqa: E402
from eval_policy import load_policy, run  # noqa: E402

NAMES = {0: 'bottom', 1: 'middle', 2: 'top'}


def stamp(frames, lines, colour):
    out = []
    for f in frames:
        f = np.ascontiguousarray(f).copy()
        for i, t in enumerate(lines):
            y = 26 + 26 * i
            cv2.putText(f, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(f, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour if i else (255, 255, 0), 2)
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ok', type=int, default=3)
    ap.add_argument('--fail', type=int, default=3)
    ap.add_argument('--max', type=int, default=24, help='scenes to try at most')
    ap.add_argument('--seed0', type=int, default=10 ** 9)
    ap.add_argument('--task', default='task_open_drawer.yaml')
    ap.add_argument('--rand-config', default='randomization_real.yaml')
    ap.add_argument('--camera-model', action='store_true')
    a = ap.parse_args()

    ckpt = os.path.abspath(a.ckpt)
    step = os.path.basename(os.path.dirname(ckpt))
    run_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt))))
    out_dir = f'{SIM}/out/eval_clips'
    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/{a.task}'))
    rcfg = yaml.safe_load(open(f'{SIM}/config/{a.rand_config}'))
    cams = None
    if a.camera_model:
        import camera_model
        cams = camera_model.load_camera_models()
    tcp_T = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf').fk(dr.ready_q(task, task['mode'], cfg['home']))['tool_tcp_link']
    policy, pre, post = load_policy(ckpt, 'cuda')
    scene_path = f'{SIM}/mjcf/_clips_scene_{os.getpid()}.xml'
    kept = {'ok': [], 'fail': []}
    try:
        for k in range(a.max):
            if len(kept['ok']) >= a.ok and len(kept['fail']) >= a.fail:
                break
            seed = a.seed0 + k
            rng = np.random.default_rng(seed)
            rand = randomize.sample(rng, rcfg)
            if cams is not None:
                for c in cams.values():
                    c.episode_jitter(rng)
            # a success on a drawer we already have is only kept once every drawer has one
            target = rand.get('target')
            xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
            open(scene_path, 'w').write(xml)
            m = mujoco.MjModel.from_xml_path(scene_path); d = mujoco.MjData(m)
            frames = []
            r = run(m, d, task, cfg, rand, policy, pre, post, 15.0, 0.20, frames, cams, rng)
            kind = 'ok' if r['success'] else 'fail'
            why = (f"opened {r['opened'] * 100:.1f} cm" +
                   (f", WRONG drawer {r['wrong_drawer_m'] * 100:.0f} cm" if r['wrong_drawer_m'] >= 0.03 else ''))
            print(f"seed {seed} {NAMES.get(target)} '{r['prompt']}': {kind.upper()} {why}", flush=True)
            have = [x[1] for x in kept[kind]]
            want = a.ok if kind == 'ok' else a.fail
            if len(kept[kind]) >= want or (kind == 'ok' and target in have and len(set(have)) < 3
                                            and k < a.max - (a.ok - len(kept['ok']))):
                continue
            label = [f'"{r["prompt"]}"  ({NAMES.get(target)} drawer)',
                     f"{'SUCCESS' if r['success'] else 'FAIL'}: {why}  [{run_name[:24]} step {int(step)}]"]
            clip = stamp(frames, label, (80, 255, 80) if r['success'] else (255, 80, 80))
            path = f'{out_dir}/{run_name}_{step}_{kind}_{NAMES.get(target)}_seed{seed}.mp4'
            imageio.mimwrite(path, clip, fps=ex.HZ, quality=6, macro_block_size=1)
            kept[kind].append((path, target))
            print('  ->', path, flush=True)
    finally:
        if os.path.exists(scene_path):
            os.remove(scene_path)
    print('kept', {k: len(v) for k, v in kept.items()})


if __name__ == '__main__':
    main()
