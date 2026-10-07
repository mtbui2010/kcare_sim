#!/usr/bin/env python3
"""Read a generated dataset back through LeRobot and summarise it.

    /sim/.venv/bin/python scripts/inspect_dataset.py [--repo-id kaair_sim_open_drawer]
      -> prints episodes / frames / per-dim action+state ranges
         /sim/out/dataset_preview/mosaic.png  first head+wrist frame of 12 episodes
         /sim/out/dataset_preview/episodes_grid.mp4  4 episodes played side by side

Decoding goes through LeRobot's own loader (pyav backend), i.e. exactly what
a training run would see.
"""
import argparse, os
import imageio, numpy as np
from PIL import Image, ImageDraw, ImageFont
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SIM = os.environ.get('SIM_ROOT', '/sim')
DATA = os.environ.get('KAAIR_DATA', '/nas/datasets')


def to_img(t):  # CHW float [0,1] -> HWC uint8
    return (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-id', default='kaair_sim_open_drawer')
    a = ap.parse_args()
    root = f'{DATA}/{a.repo_id}'
    ds = LeRobotDataset(a.repo_id, root=root, video_backend='pyav')
    out = f'{SIM}/out/dataset_preview'
    os.makedirs(out, exist_ok=True)

    names = ds.features['action']['names']
    hf = ds.hf_dataset.with_format('numpy')
    act, st = np.stack(hf['action']), np.stack(hf['observation.state'])
    ep_idx = np.asarray(hf['episode_index'])
    lengths = np.bincount(ep_idx)
    print(f'{a.repo_id}: {ds.num_episodes} episodes, {ds.num_frames} frames @ {ds.fps} fps '
          f'({ds.num_frames / ds.fps / 60:.1f} min), episode length {lengths.min()}-{lengths.max()} frames, '
          f'robot_type {ds.meta.robot_type}, tasks {list(ds.meta.tasks.index)}')
    print(f'{"":14s}{"action min":>11s}{"max":>9s}{"state min":>11s}{"max":>9s}   |action-state| mean')
    for i, n in enumerate(names):
        print(f'{n:14s}{act[:, i].min():11.3f}{act[:, i].max():9.3f}{st[:, i].min():11.3f}{st[:, i].max():9.3f}'
              f'   {np.abs(act[:, i] - st[:, i]).mean():.4f}')

    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    tiles = []
    for e in range(min(12, ds.num_episodes)):
        x = ds[int(starts[e])]
        pair = np.concatenate([to_img(x['observation.images.head']), to_img(x['observation.images.wrist'])], 1)
        im = Image.fromarray(pair).resize((848, 240))
        ImageDraw.Draw(im).text((6, 4), f'ep {e}  head | wrist', fill=(255, 255, 0), font=font,
                                stroke_width=2, stroke_fill=(0, 0, 0))
        tiles.append(im)
    cols = 2
    mosaic = Image.new('RGB', (848 * cols, 240 * ((len(tiles) + cols - 1) // cols)))
    for k, t in enumerate(tiles):
        mosaic.paste(t, ((k % cols) * 848, (k // cols) * 240))
    mosaic.save(f'{out}/mosaic.png')

    eps = list(range(min(4, ds.num_episodes)))
    T = int(max(lengths[eps]))
    frames = []
    for t in range(0, T):
        row = []
        for e in eps:
            x = ds[int(starts[e] + min(t, lengths[e] - 1))]
            pair = np.concatenate([to_img(x['observation.images.head']), to_img(x['observation.images.wrist'])], 1)
            row.append(np.asarray(Image.fromarray(pair).resize((848, 240))))
        frames.append(np.concatenate([np.concatenate(row[:2], 1), np.concatenate(row[2:4], 1)], 0)
                      if len(row) == 4 else np.concatenate(row, 0))
    imageio.mimwrite(f'{out}/episodes_grid.mp4', frames, fps=ds.fps, quality=6, macro_block_size=1)
    print('->', f'{out}/mosaic.png', f'{out}/episodes_grid.mp4')


if __name__ == '__main__':
    main()
