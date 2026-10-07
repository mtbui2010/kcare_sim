#!/usr/bin/env python3
"""Record a video of the robot doing a short errand tour through kcare_robot's HTTP API.

    source /workspace/ros2/ros2_ws/install/setup.bash
    python3 scripts/house_record.py [--api http://localhost:8002] [--speed 2]
      -> /sim/out/house/robot_tour.mp4

Each frame: follow camera (big), floor plan with the robot's pose and trail, head and wrist
cameras, and the command being executed. Commands go through kcare exactly as from the
dashboard: POST /skill/move, POST /skill/find.
"""
import argparse, json, math, os, subprocess, threading, time, urllib.request

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

SIM = os.environ.get('SIM_ROOT', '/sim')
OUT = f'{SIM}/out/house'
PLAN = [('move', 'desk@study room'), ('find', 'orange@desk@study room'),
        ('move', 'counter@kitchen'), ('find', 'mustard bottle@counter@kitchen'),
        ('move', 'chest@bedroom'), ('find', 'red cup@chest@bedroom'),
        ('move', 'shelf@restroom'), ('move', 'drawer@kitchen'), ('vla_drawer_open', 'open drawer'), ('move', 'home')]
RES = 0.05                                # house_bridge.Planner grid: house bounds + 0.5 m margin


class Rec(Node):
    def __init__(self):
        super().__init__('house_recorder')
        self.img, self.pose, self.trail = {}, None, []
        for k, t in (('over', '/sim/overview/image_raw/compressed'), ('wrist', '/hand/camera/color/image_raw/compressed')):
            self.create_subscription(CompressedImage, t, self._jpg(k), 1)
        self.create_subscription(Image, '/femto/color/image_raw', self._raw, 1)
        self.create_subscription(PoseStamped, '/robot_pose/mobile_pose', self._pose, 1)

    def _jpg(self, k):
        def f(m):
            self.img[k] = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        return f

    def _raw(self, m):
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        self.img['head'] = cv2.cvtColor(cv2.resize(im, (424, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)

    def _pose(self, m):
        q = m.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (m.pose.position.x, m.pose.position.y, yaw)
        if not self.trail or math.dist(self.trail[-1], self.pose[:2]) > 0.05:
            self.trail.append(self.pose[:2])


def floor_plan(lay, size=(424, 360)):
    """Occupancy grid from the bridge as a clean floor plan + room names; returns (image, world->pixel)."""
    occ = cv2.imread(f'{OUT}/occupancy.png', 0)[::-1]             # rows = y upward again
    b = lay.get('bounds', dict(x=[-1.5, 9.5], y=[-4.5, 4.5]))
    X0, Y0 = b['x'][0] - 0.5, b['y'][0] - 0.5
    xs, ys = (b['x'][0] - 0.1, b['x'][1] + 0.1), (b['y'][0] - 0.1, b['y'][1] + 0.1)
    crop = occ[int((ys[0] - Y0) / RES):int((ys[1] - Y0) / RES), int((xs[0] - X0) / RES):int((xs[1] - X0) / RES)]
    plan = np.full(crop.shape + (3,), 245, np.uint8)
    plan[crop < 150] = (190, 190, 190)                            # inflated band (light)
    plan[crop < 60] = (60, 60, 60)                                # obstacles / walls
    plan = cv2.resize(plan[::-1], size, interpolation=cv2.INTER_NEAREST)
    sx, sy = size[0] / (xs[1] - xs[0]), size[1] / (ys[1] - ys[0])
    to_px = lambda x, y: (int((x - xs[0]) * sx), int((ys[1] - y) * sy))
    for room, r in lay['rooms'].items():
        if room in ('hall', 'entrance', 'corridor', 'warehouse'):
            continue
        p = to_px(r['x'][0] + 0.15, (r['y'][0] + r['y'][1]) / 2)
        cv2.putText(plan, room, p, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 90, 40), 1, cv2.LINE_AA)
    return plan, to_px


def label(img, text, org, scale=0.6, col=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


def call(api, skill, inputs):
    req = urllib.request.Request(f'{api}/skill/{skill}', data=json.dumps({'inputs': inputs}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='http://127.0.0.1:8001')
    ap.add_argument('--speed', type=float, default=2.0, help='playback speed-up')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--plan', default=None, help='"skill:inputs;skill:inputs" (default: the errand tour)')
    ap.add_argument('--out', default='robot_tour.mp4')
    a = ap.parse_args()
    plan = [tuple(x.split(':', 1)) for x in a.plan.split(';')] if a.plan else PLAN
    lay = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))
    fplan, to_px = floor_plan(lay)
    rclpy.init()
    node = Rec()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    state = {'cmd': 'start', 'result': '', 'done': False}

    def run():
        try:
            _run()
        except Exception as e:                   # a dead command thread must end the recording, not hang it
            state['result'] = f'error: {e}'[:70]
            print('command thread failed:', e, flush=True)
            time.sleep(2)
        state['done'] = True

    def _run():
        time.sleep(2)
        for skill, inp in plan:
            state['cmd'], state['result'] = f'{skill}::{inp}', ''
            r = call(a.api, skill, inp)
            if skill == 'find':
                ins = r.get('ins') or {}
                state['result'] = (f"found: {', '.join(ins)} (score {max(v.get('score', 0) for v in ins.values()):.2f})"
                                   if r.get('isdone') and ins else 'not found')
            elif skill.startswith('vla'):
                state['result'] = (r.get('msg') or '')[:70]
            else:
                state['result'] = 'arrived' if r.get('isdone') else f"failed: {r.get('msg', '')[:60]}"
            time.sleep(2.5)                                       # hold the result on screen
    threading.Thread(target=run, daemon=True).start()

    tmp = f'{OUT}/_robot_tour_raw.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (1280, 720))
    period = a.speed / a.fps
    t_next = time.time()
    while not state['done']:
        if len(node.img) < 3:
            time.sleep(0.1); continue
        f = np.full((720, 1280, 3), 30, np.uint8)
        f[0:600, 0:848] = cv2.resize(node.img['over'], (848, 600))
        pl = fplan.copy()
        if len(node.trail) > 1:
            cv2.polylines(pl, [np.array([to_px(*p) for p in node.trail], np.int32)], False, (40, 160, 255), 2)
        if node.pose:
            x, y, yaw = node.pose
            c = to_px(x, y)
            cv2.circle(pl, c, 7, (0, 90, 230), -1)
            cv2.line(pl, c, (int(c[0] + 14 * math.cos(yaw)), int(c[1] - 14 * math.sin(yaw))), (0, 0, 0), 2)
        f[0:360, 856:1280] = pl
        f[360:600, 856:1280] = node.img['head']
        f[600:720, 856:1068] = cv2.resize(node.img['wrist'], (212, 120))
        label(f, 'follow camera (sim only)', (12, 30), 0.6)
        label(f, 'floor plan', (866, 22), 0.5)
        label(f, 'head camera', (866, 385), 0.5)
        label(f, 'wrist', (1072, 640), 0.5)
        label(f, f"kcare_robot  {state['cmd']}", (12, 645), 0.75, (0, 255, 255))
        if state['result']:
            label(f, state['result'], (12, 690), 0.7, (120, 255, 120) if 'fail' not in state['result'] and 'not' not in state['result'] else (80, 80, 255))
        label(f, f'x{a.speed:g}', (800, 30), 0.6)
        vw.write(f)
        t_next += period
        time.sleep(max(0.0, t_next - time.time()))
    vw.release()
    ff = subprocess.run([f'{SIM}/.venv/bin/python', '-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
                        capture_output=True, text=True).stdout.strip()
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26',
                    '-movflags', '+faststart', f'{OUT}/{a.out}'], check=True)
    os.remove(tmp)
    print('->', f'{OUT}/{a.out}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
