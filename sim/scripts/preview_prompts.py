#!/usr/bin/env python3
"""One row per drawer target: the prompt, and head | wrist at the start / grasp / end.

    .venv/bin/python scripts/preview_prompts.py kaair_sim_open_drawer_real500
      -> /sim/out/dataset_preview/prompts.png
         prints the prompt counts per drawer (the balance check)
"""
import collections, json, os, sys

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SIM = os.environ.get('SIM_ROOT', '/sim')
DATA = os.environ.get('KAAIR_DATA', '/nas/datasets')
NAMES = {0: 'bottom', 1: 'middle', 2: 'top'}


def label(img, text, scale=0.7):
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 2)
    return img


def to_img(t):
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    repo = sys.argv[1]
    ds = LeRobotDataset(repo, root=f'{DATA}/{repo}', video_backend='pyav')
    # episode index -> task string (first frame of each episode)
    ep_task = {}
    for ep in range(ds.meta.total_episodes):
        i0 = int(ds.meta.episodes[ep]['dataset_from_index'])
        ep_task[ep] = ds[i0]['task']
    # prompt -> drawer, from the rand config the data was made with
    import yaml
    dt = yaml.safe_load(open(f'{SIM}/config/randomization_real.yaml'))['drawer_targets']['prompts']
    drawer_of = {p: t for t, ps in dt.items() for p in ps}
    count = collections.Counter((drawer_of.get(p), p) for p in ep_task.values())
    for (t, p), n in sorted(count.items(), key=lambda x: (-1 if x[0][0] is None else x[0][0], x[0][1])):
        print(f'{NAMES.get(t, "?"):7s} {p:28s} {n}')
    per = collections.Counter(drawer_of.get(p) for p in ep_task.values())
    print('per drawer:', {NAMES.get(k, k): v for k, v in sorted(per.items())})

    rows = []
    for t in (2, 1, 0):
        eps = [e for e, p in ep_task.items() if drawer_of.get(p) == t][:2]
        for ep in eps:
            a, b = int(ds.meta.episodes[ep]['dataset_from_index']), int(ds.meta.episodes[ep]['dataset_to_index'])
            tiles = []
            for k, frac in enumerate((0.0, 0.45, 0.999)):
                f = ds[a + int((b - a) * frac)]
                h = cv2.resize(to_img(f['observation.images.head']), (424, 240), interpolation=cv2.INTER_AREA)
                w = cv2.resize(to_img(f['observation.images.wrist']), (424, 240), interpolation=cv2.INTER_AREA)
                if k == 0:
                    h = label(h, f'ep{ep}: "{ep_task[ep]}"', 0.55)
                tiles += [h, w]
            rows.append(np.concatenate(tiles, 1))
    grid = np.concatenate(rows, 0)
    out = f'{SIM}/out/dataset_preview/prompts.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print('->', out)


if __name__ == '__main__':
    main()
