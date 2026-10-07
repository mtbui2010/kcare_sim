#!/usr/bin/env python3
"""Fit the sim camera model to real captures (capture_scene.py output on the NAS).

    python3 scripts/estimate_camera.py /nas/real/20260928_scene1
      -> /sim/config/camera_real.yaml

Per camera (head = Femto Bolt colour, wrist = D405 colour):
  * intrinsics + plumb-bob distortion straight from camera_info;
  * JPEG quality: the image's own luminance quantization table vs the IJG
    standard table scaled by quality q (the publisher's encoder setting);
  * sensor noise: from a burst of consecutive frames of a static scene, the
    temporal std of every pixel against its mean — fitted as
    sigma^2(I) = a*I + b per channel (shot + read noise). Measured after the
    camera's own JPEG, so it is the noise the dataset actually contains;
  * colour gain of the wrist camera relative to the head camera, from the same
    drawer front seen by both (the sim textures are taken from the head view).
"""
import glob, json, os, sys

import cv2
import numpy as np
import yaml
from PIL import Image

SIM = os.environ.get('SIM_ROOT', '/sim')
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)


def jpeg_quality(path):
    q = Image.open(path).quantization
    if not q:
        return None
    luma = np.array(q[0], float)
    best, best_err = None, 1e18
    for quality in range(1, 101):
        s = 5000 / quality if quality < 50 else 200 - 2 * quality
        table = np.clip(np.floor((STD_LUMA * s + 50) / 100), 1, 255)
        err = np.abs(np.sort(table) - np.sort(luma)).sum()     # zigzag order does not matter for the fit
        if err < best_err:
            best, best_err = quality, err
    return best


def noise_fit(frames):
    stack = np.stack([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) for f in frames])
    mean, var = stack.mean(0), stack.var(0)
    fit = []
    for c in range(3):
        m, v = mean[..., c].ravel(), var[..., c].ravel()
        keep = (m > 10) & (m < 245)                               # avoid clipped pixels
        bins = np.linspace(10, 245, 24)
        idx = np.digitize(m[keep], bins)
        xs = np.array([m[keep][idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() > 200])
        ys = np.array([np.median(v[keep][idx == i]) for i in range(1, len(bins)) if (idx == i).sum() > 200])
        a, b = np.polyfit(xs, ys, 1)
        fit.append((max(float(a), 0.0), max(float(b), 0.0)))
    return fit, float(np.sqrt(var).mean())


def main():
    src = sys.argv[1]
    out = {}
    for cam in ('head', 'wrist'):
        info = json.load(open(f'{src}/{cam}_camera_info.json'))
        K = info['K']
        frames = sorted(glob.glob(f'{src}/burst_{cam}/*.jpg'))
        fit, mean_std = noise_fit(frames)
        out[cam] = dict(
            width=info['width'], height=info['height'],
            fx=K[0], fy=K[4], cx=K[2], cy=K[5], distortion=info['D'][:5],
            jpeg_quality=jpeg_quality(frames[0]),
            noise_var_a=[round(a, 5) for a, _ in fit], noise_var_b=[round(b, 4) for _, b in fit],
            measured_mean_std=round(mean_std, 3),
        )
    # same drawer front in both views (pixel boxes picked on the 2026-09-28 capture)
    h = cv2.cvtColor(cv2.imread(f'{src}/head.jpg'), cv2.COLOR_BGR2RGB).astype(float)
    w = cv2.cvtColor(cv2.imread(f'{src}/wrist.jpg'), cv2.COLOR_BGR2RGB).astype(float)
    front_h = h[605:625, 120:160].reshape(-1, 3).mean(0)
    front_w = w[238:338, 472:578].reshape(-1, 3).mean(0)
    out['head']['color_gain'] = [1.0, 1.0, 1.0]
    out['wrist']['color_gain'] = [round(float(x), 3) for x in front_w / front_h]
    out['_source'] = src
    yaml.safe_dump(out, open(f'{SIM}/config/camera_real.yaml', 'w'), sort_keys=False)
    print(yaml.safe_dump(out, sort_keys=False))


if __name__ == '__main__':
    main()
