"""Record a task run as one video: the sim cameras + a dialogue / event panel (task_runner --video DIR).

Frames are grabbed in wall time every 1/`grab_hz` s and written at `fps`, so the video plays `fps/grab_hz` times
faster than real time; a clock in the corner shows the real elapsed time. Layout (1280x720):

    +--------------------------+----------------+
    | follow camera (big)      |  dialogue /    |
    +------------+-------------+  events        |
    | head cam   | wrist cam   |                |
    +------------+-------------+----------------+
"""
import os, subprocess, textwrap, threading, time

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

W, H, PANEL = 1280, 720, 440
COL = {'user': (90, 200, 255), 'robot': (140, 255, 140), 'plan': (220, 220, 220), 'ok': (120, 230, 120),
       'fail': (90, 90, 255), 'info': (200, 200, 200), 'title': (255, 255, 255), 'recover': (60, 200, 255),
       'fault': (255, 120, 255)}


def fit(img, w, h):
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


class TaskVideo:
    def __init__(self, bench, path, title, grab_hz=5.0, fps=15):
        self.b, self.path, self.title = bench, path, title
        self.grab_hz, self.fps = grab_hz, fps
        self.img = {}
        for k, t in (('over', '/sim/overview/image_raw/compressed'), ('wrist', '/hand/camera/color/image_raw/compressed')):
            bench.create_subscription(CompressedImage, t, self._jpg(k), 1)
        for _ in range(60):                         # wait for the first frames (else the video opens black)
            if 'over' in self.img and 'wrist' in self.img and bench.img.get('head'):
                break
            time.sleep(0.1)
        for _ in range(60):                         # wait for the first frames (else the video opens black)
            if 'over' in self.img and 'wrist' in self.img and bench.img.get('head'):
                break
            time.sleep(0.1)
        self.lines, self.step = [], ''
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.t0 = time.time()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.tmp = path + '.tmp.mp4'
        self.vw = cv2.VideoWriter(self.tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _jpg(self, k):
        def f(m):
            self.img[k] = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        return f

    def _head(self):
        v = self.b.img.get('head')
        if not v:
            return None
        m = v[0]
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    def add(self, kind, text):
        with self.lock:
            self.lines.append((time.time() - self.t0, kind, text))

    def set_step(self, text):
        with self.lock:
            self.step = text

    def _panel(self):
        p = np.full((H, PANEL, 3), 28, np.uint8)
        cv2.putText(p, self.title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL['title'], 2, cv2.LINE_AA)
        el = time.time() - self.t0
        cv2.putText(p, f'real time {int(el // 60):02d}:{int(el % 60):02d}   (video x{self.fps / self.grab_hz:.0f})', (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        with self.lock:
            lines, step = list(self.lines), self.step
        rows = []
        for t, kind, text in lines:
            prefix = {'user': 'USER: ', 'robot': 'ROBOT: ', 'fault': '[TEST] '}.get(kind, '')
            for k, chunk in enumerate(textwrap.wrap(prefix + text, 50) or ['']):
                rows.append((kind, (f'{int(t // 60):02d}:{int(t % 60):02d} ' if k == 0 else '      ') + chunk))
        y, maxrows = 84, (H - 130) // 19
        for kind, s in rows[-maxrows:]:
            cv2.putText(p, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COL.get(kind, COL['info']), 1, cv2.LINE_AA)
            y += 19
        cv2.rectangle(p, (0, H - 40), (PANEL, H), (55, 55, 55), -1)
        cv2.putText(p, step[:52], (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return p

    def _frame(self):
        L = W - PANEL
        top = fit(self.img.get('over'), L, 460)
        bl = fit(self._head(), L // 2, H - 460)
        br = fit(self.img.get('wrist'), L // 2, H - 460)
        for im, s in ((top, 'follow camera'), (bl, 'head camera'), (br, 'wrist camera')):
            cv2.putText(im, s, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        left = np.vstack([top, np.hstack([bl, br])])
        return np.hstack([left, self._panel()])

    def _loop(self):
        nxt = time.time()
        while not self.stop_ev.is_set():
            try:
                self.vw.write(self._frame())
            except Exception:
                pass
            nxt += 1.0 / self.grab_hz
            time.sleep(max(0.0, nxt - time.time()))

    def close(self, tail_sec=3.0):
        time.sleep(tail_sec)
        self.stop_ev.set(); self.th.join(timeout=5)
        for _ in range(int(self.fps * 2)):            # hold the last frame 2 s
            self.vw.write(self._frame())
        self.vw.release()
        sim = os.environ.get('SIM_ROOT', '/sim')
        try:
            ff = subprocess.run([f'{sim}/.venv/bin/python', '-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'],
                                capture_output=True, text=True).stdout.strip()
            subprocess.run([ff, '-y', '-loglevel', 'error', '-i', self.tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-crf', '26', '-preset', 'veryfast', self.path], check=True)
            os.remove(self.tmp)
        except Exception:
            os.replace(self.tmp, self.path)
        return self.path
