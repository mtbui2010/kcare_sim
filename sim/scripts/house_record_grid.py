#!/usr/bin/env python3
"""Record the four sim cameras as a 2x2 grid while kcare_robot runs a plan — the dashboard's camera view.

    source /workspace/ros2/ros2_ws/install/setup.bash; export ROS_DOMAIN_ID=77
    python3 scripts/house_record_grid.py --plan "move:desk@study room;find:orange@desk@study room" \
        --speed 3 --out tour_grid        -> /sim/out/house/tour_grid.mp4 + tour_grid_overlay.json

Grid (1280x720): head camera (Femto) | wrist camera (D405) / follow camera | fixed top-down camera, each
labelled, plus the kcare command being run. Commands go through kcare's HTTP API exactly as the dashboard
sends them (POST /skill/<name>). The overlay JSON (narrated-tutorial-video format) holds one event per command
and the largest drawer opening per frame (/sim/drawer_states, sim ground truth) for a progress bar.
"""
import argparse, json, os, subprocess, threading, time, urllib.request

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, JointState

SIM = os.environ.get('SIM_ROOT', '/sim')
OUT = f'{SIM}/out/house'
VIEWS = [('head', 'head camera (Femto)'), ('wrist', 'wrist camera (D405)'),
         ('over', 'follow camera'), ('top', 'top-down camera')]


class Rec(Node):
    def __init__(self):
        super().__init__('house_grid_recorder')
        self.img, self.drawers, self.drawer0 = {}, {}, None
        for k, t in (('wrist', '/hand/camera/color/image_raw/compressed'), ('over', '/sim/overview/image_raw/compressed'),
                     ('top', '/sim/topdown/image_raw/compressed')):
            self.create_subscription(CompressedImage, t, self._jpg(k), 1)
        self.create_subscription(Image, '/femto/color/image_raw', self._raw, 1)
        self.create_subscription(JointState, '/sim/drawer_states', self._dr, 10)

    def _jpg(self, k):
        def f(m):
            self.img[k] = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        return f

    def _raw(self, m):
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        self.img['head'] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    def _dr(self, m):
        self.drawers = {n: p for n, p in zip(m.name, m.position) if n.startswith('cab_')}

    def opened_cm(self):
        if self.drawer0 is None or not self.drawers:
            return 0.0
        return 100 * max(self.drawers.get(n, 0.0) - self.drawer0.get(n, 0.0) for n in self.drawers)


def label(img, text, org, scale=0.55, col=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def fit(img, w, h):
    """Letterbox img into w x h."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 24, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def call(api, skill, inputs):
    req = urllib.request.Request(f'{api}/skill/{skill}', data=json.dumps({'inputs': inputs}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=400) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='http://127.0.0.1:8001')
    ap.add_argument('--plan', required=True, help='"skill:inputs;skill:inputs" sent to kcare in order')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed-up (frames are sampled in sim time)')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--out', default='grid')
    a = ap.parse_args()
    plan = [tuple(x.split(':', 1)) for x in a.plan.split(';')]
    rclpy.init()
    node = Rec()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    state = {'cmd': '', 'result': '', 'done': False, 'events': [], 'success': None}

    def run():
        try:
            time.sleep(1.5)
            for skill, inp in plan:
                state['cmd'], state['result'] = f'{skill}::{inp}', ''
                state['events'].append((len(frames), f'{skill}::{inp}'))
                if skill.startswith('vla'):
                    node.drawer0 = dict(node.drawers)
                r = call(a.api, skill, inp)
                if skill == 'find':
                    ins = r.get('ins') or {}
                    state['result'] = f"found {', '.join(ins)}" if r.get('isdone') and ins else 'not found'
                elif skill.startswith('vla'):
                    state['success'] = bool(r.get('isdone'))
                    state['result'] = (r.get('msg') or '')[:60]
                else:
                    state['result'] = 'done' if r.get('isdone') else f"failed: {r.get('msg', '')[:50]}"
                time.sleep(2.0)
        except Exception as e:                    # a dead command thread must end the recording
            state['result'] = f'error: {e}'[:70]
            print('command thread failed:', e, flush=True)
        state['done'] = True

    frames, progress, badge = [], [], None
    tmp = f'{OUT}/_{a.out}_raw.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (1280, 720))
    while len(node.img) < 4:
        time.sleep(0.1)
    threading.Thread(target=run, daemon=True).start()
    period, t_next = a.speed / a.fps, time.time()
    while not state['done']:
        f = np.zeros((720, 1280, 3), np.uint8)
        for i, (k, name) in enumerate(VIEWS):
            x, y = (i % 2) * 640, (i // 2) * 330
            f[y:y + 330, x:x + 640] = fit(node.img[k], 640, 330)
            label(f, name, (x + 10, y + 22))
        f[660:720] = 18
        label(f, f"kcare_robot  {state['cmd']}", (12, 684), 0.6, (0, 220, 255))
        if state['result']:
            label(f, state['result'], (12, 710), 0.5, (140, 255, 140) if 'fail' not in state['result'] and 'not' not in state['result'] else (90, 90, 255))
        if a.speed != 1:
            label(f, f'x{a.speed:g}', (1225, 684), 0.6)
        vw.write(f)
        frames.append(1)
        progress.append(round(node.opened_cm(), 2))
        if state['success'] and badge is None:
            badge = len(frames)
        t_next += period
        time.sleep(max(0.0, t_next - time.time()))
    vw.release()
    ff = subprocess.run([f'{SIM}/.venv/bin/python', '-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
                        capture_output=True, text=True).stdout.strip()
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
                    '-movflags', '+faststart', f'{OUT}/{a.out}.mp4'], check=True)
    os.remove(tmp)
    ov = {'events': [{'frame': fr, 'text': t} for fr, t in state['events']],
          'progress': {'values': progress, 'label': 'drawer opened  {v:4.1f} cm', 'max': 28, 'goal': 20,
                       'goal_label': '20 cm = success'}}
    if badge is not None:
        ov['badge'] = {'from_frame': badge, 'text': 'success'}
    json.dump(ov, open(f'{OUT}/{a.out}_overlay.json', 'w'))
    print('->', f'{OUT}/{a.out}.mp4', len(frames), 'frames', 'max opened', max(progress) if progress else 0)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
