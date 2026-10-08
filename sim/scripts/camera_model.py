"""Make a MuJoCo render look like the real camera's published image.

    cams = load_camera_models()                       # from config/camera_real.yaml
    img = cams['head'].apply(render_rgb, rng)         # uint8 HxWx3 -> uint8 HxWx3

MuJoCo renders an ideal pinhole image with the real intrinsics (fx, fy, cx,
cy are set on the MJCF camera, see urdf_to_mjcf.py). This adds what the real
sensor + driver do on top, in the order they happen on the robot:
  1. lens distortion (plumb-bob k1 k2 p1 p2 k3 from camera_info): the ROS topic
     is image_raw, i.e. NOT rectified;
  2. auto-exposure: scale the image so its mean luminance hits the camera's
     target (measured on the real image) — a bright window behind the cabinet
     darkens the foreground, as on the robot;
  3. colour: saturation (the D405 renders the same wood far more saturated than
     the Femto) and white-balance gain, fitted by calibrate_color.py; small
     per-episode AE/AWB jitter;
  4. sensor noise, sigma^2 = a*I + b per channel (fitted on a static burst);
  5. JPEG at the publisher's quality (Femto 75, D405 95) — the recorder stores
     the compressed topics.
  0. (optional, before 1) motion: pass the camera pose + time to `apply` and the
     image is smeared along the image motion during the exposure (`exposure_ms`),
     plus a row-dependent shift for a rolling-shutter sensor (`readout_ms`: the
     Femto's colour sensor; the D405 is global shutter).
Jitter is drawn once per episode (`episode_jitter`) — real AE/AWB drift slowly,
not per frame.

`DepthModel` (cfg `<cam>.depth`) does the same for the depth image — see its
docstring: Femto Bolt = time-of-flight registered to colour, D405 = passive
stereo, fitted to the one real D405 frame we have (/nas/real/20260928_scene1).
"""
import math
import os

import cv2
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')


class CameraModel:
    def __init__(self, cfg, wb_jitter=0.06, exposure_jitter=0.12, jpeg_jitter=5):
        self.cfg = cfg
        w, h = cfg['width'], cfg['height']
        self.K = np.array([[cfg['fx'], 0, cfg['cx']], [0, cfg['fy'], cfg['cy']], [0, 0, 1]], np.float64)
        self.D = np.array(cfg['distortion'], np.float64)
        # distortion as a remap: for every pixel of the DISTORTED output image, where it
        # comes from in the ideal pinhole render (undistortPoints inverts the lens model)
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([uu.ravel(), vv.ravel()], 1)[:, None, :]
        und = cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(h, w, 2)
        self.map_x, self.map_y = und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)
        self.gain = np.array(cfg.get('color_gain', [1, 1, 1]), np.float32)      # white balance
        self.saturation = float(cfg.get('saturation', 1.0))
        self.ae_target = cfg.get('ae_target')                                   # mean luminance, None = off
        # white-referenced exposure + auto white balance (what a camera's AE/AWB do in a bright room): the
        # brightest low-chroma pixels are taken as "white", pulled to `awb_white` chromaticity with strength
        # `awb`, and exposed so their mean luminance is `white_level`. Off unless set (office fit unchanged).
        self.white_level = cfg.get('white_level')
        self.awb = float(cfg.get('awb', 0.0))
        self.awb_white = np.array(cfg.get('awb_white', [1.0, 1.0, 1.0]), np.float32)
        self.a = np.array(cfg.get('noise_var_a', [0, 0, 0]), np.float32)
        self.b = np.array(cfg.get('noise_var_b', [0, 0, 0]), np.float32)
        self.quality = int(cfg.get('jpeg_quality') or 90)
        self.wb_jitter, self.exposure_jitter, self.jpeg_jitter = wb_jitter, exposure_jitter, jpeg_jitter
        self.exposure_s = float(cfg.get('exposure_ms', 0.0)) / 1000      # shutter time, for motion blur
        self.readout_s = float(cfg.get('readout_ms', 0.0)) / 1000        # rolling-shutter top->bottom readout
        self.motion_depth = float(cfg.get('motion_depth_m', 1.0))         # typical scene distance (translation flow)
        # color_gain was fitted under the office lights (the D405 comes out warm there); under other lights the
        # camera's AWB lands elsewhere: per episode, a gain between the fitted one and neutral (wb_neutral_mix)
        self.wb_mix = cfg.get('wb_neutral_mix')
        self._rows = (np.arange(h, dtype=np.float32) / h - 0.5)[:, None]
        self.episode_jitter(None)

    def episode_jitter(self, rng):
        """Draw this episode's AE/AWB state (None = no jitter)."""
        self._prev = None                                   # last (pose, time) for the motion model
        self.ep_shutter = self.exposure_s
        if rng is None:
            self.ep_gain, self.ep_quality, self.ep_exposure = self.gain, self.quality, 1.0
            return
        wb = 1 + rng.uniform(-self.wb_jitter, self.wb_jitter, 3)
        self.ep_exposure = 1 + rng.uniform(-self.exposure_jitter, self.exposure_jitter)
        g = self.gain
        if self.wb_mix:                                    # 0 = fitted gain, 1 = neutral
            k = float(rng.uniform(*self.wb_mix))
            g = g * (1 - k) + k * float(np.mean(g))
        self.ep_gain = (g * wb).astype(np.float32)
        self.ep_quality = int(np.clip(self.quality + rng.integers(-self.jpeg_jitter, self.jpeg_jitter + 1), 30, 100))
        self.ep_shutter = self.exposure_s * float(rng.uniform(0.6, 1.4))   # AE picks the shutter: varies with the room

    def image_flow(self, cam_T, t):
        """Image motion [px/s] of the scene centre from the camera pose change since the last call.
        cam_T: 4x4 camera pose in a fixed frame (MuJoCo camera axes: x right, y up, looking along -z)."""
        prev, self._prev = self._prev, (cam_T, t)
        if prev is None or t - prev[1] <= 1e-6 or t - prev[1] > 0.5:
            return 0.0, 0.0
        dt = t - prev[1]
        R = prev[0][:3, :3].T @ cam_T[:3, :3]                       # rotation in the previous camera frame
        w = cv2.Rodrigues(R)[0].ravel() / dt                         # angular velocity [rad/s]
        v = prev[0][:3, :3].T @ (cam_T[:3, 3] - prev[0][:3, 3]) / dt
        fx, fy = self.K[0, 0], self.K[1, 1]
        # turning about the camera's up axis moves the image sideways, about its x axis up/down; translation
        # moves it by f * v / Z at the typical scene distance (image v grows downward, camera y is up)
        du = -fx * w[1] - fx * v[0] / self.motion_depth
        dv = -fy * w[0] + fy * v[1] / self.motion_depth
        return float(du), float(dv)

    def _motion(self, x, flow):
        du, dv = flow
        if self.readout_s > 0 and abs(du) + abs(dv) > 1:            # rolling shutter: later rows see a later scene
            mx = self.map_x + self._rows * (du * self.readout_s)
            my = self.map_y + self._rows * (dv * self.readout_s)
        else:
            mx, my = self.map_x, self.map_y
        x = cv2.remap(x, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        L = math.hypot(du, dv) * self.ep_shutter                     # blur length [px]
        if L > 0.75:
            n = int(min(math.ceil(L), 61)) | 1
            k = np.zeros((n, n), np.float32)
            c = n // 2
            a = math.atan2(dv, du)
            cv2.line(k, (int(round(c - math.cos(a) * L / 2)), int(round(c - math.sin(a) * L / 2))),
                     (int(round(c + math.cos(a) * L / 2)), int(round(c + math.sin(a) * L / 2))), 1.0, 1)
            x = cv2.filter2D(x, -1, k / max(k.sum(), 1e-6), borderType=cv2.BORDER_REPLICATE)
        return x

    def _white_ref(self, x):
        Y = np.array([0.299, 0.587, 0.114], np.float32)
        s = x[::8, ::8].reshape(-1, 3)
        lum = s @ Y
        chroma = s.max(1) - s.min(1)
        w = (lum >= np.percentile(lum, 70)) & (chroma < 0.2 * lum + 8)
        if w.sum() < 30:                                    # no white in view: plain mean-luminance AE
            if self.ae_target:
                x = x * float(np.clip(self.ae_target * self.ep_exposure / max(float(lum.mean()), 1.0), 0.3, 3.0))
            return x
        m = s[w].mean(0) + 1e-3
        if self.awb > 0:                                    # pull "white" to awb_white, keep its luminance
            g = (self.awb_white / (self.awb_white @ Y)) / (m / (m @ Y))
            g = (g ** self.awb).astype(np.float32)
            g = g / float((m * g) @ Y / (m @ Y))
            x = x * g; m = m * g
        if self.white_level:
            x = x * float(np.clip(self.white_level * self.ep_exposure / float(m @ Y), 0.4, 3.0))
        return x

    def apply(self, img, rng=None, stages=('ae', 'color'), cam_T=None, t=None):
        """cam_T, t (optional): camera pose + time of this frame -> motion blur / rolling shutter."""
        flow = self.image_flow(cam_T, t) if cam_T is not None and t is not None else (0.0, 0.0)
        x = self._motion(img, flow).astype(np.float32)
        if 'ae' in stages and (self.white_level or self.awb > 0):
            x = self._white_ref(x)
        elif 'ae' in stages and self.ae_target:
            lum = float((x @ np.array([0.299, 0.587, 0.114], np.float32)).mean())
            x = x * float(np.clip(self.ae_target * self.ep_exposure / max(lum, 1.0), 0.3, 3.0))
        if 'color' in stages:
            lum = (x @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
            x = (lum + self.saturation * (x - lum)) * self.ep_gain
        if rng is not None:
            sigma = np.sqrt(np.maximum(self.a * x + self.b, 0))
            x = x + rng.standard_normal(x.shape).astype(np.float32) * sigma
        x = np.clip(x, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.ep_quality])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _load(path=None, sensor=None, profile=None):
    """camera_real.yaml (+ camera_<profile>.yaml colour overrides, e.g. 'home' from calibrate_home.py)
    + camera_sensor.yaml (motion / depth)."""
    cfg = yaml.safe_load(open(path or f'{SIM}/config/camera_real.yaml'))
    if profile:
        for cam, extra in yaml.safe_load(open(f'{SIM}/config/camera_{profile}.yaml')).items():
            if not cam.startswith('_'):
                cfg[cam].update(extra)
    sp = sensor or f'{SIM}/config/camera_sensor.yaml'
    if os.path.exists(sp):                       # motion / depth behaviour, kept out of the calibrated file
        for cam, extra in (yaml.safe_load(open(sp)) or {}).items():
            cfg.setdefault(cam, {}).update(extra)
    return cfg


def load_camera_models(path=None, sensor=None, profile=None):
    cfg = _load(path, sensor, profile)
    return {cam: CameraModel(cfg[cam]) for cam in ('head', 'wrist')}


def cam_pose(m, d, name):
    """4x4 world pose of a MuJoCo camera (for CameraModel.apply's motion model)."""
    import mujoco
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    T = np.eye(4); T[:3, :3] = d.cam_xmat[i].reshape(3, 3); T[:3, 3] = d.cam_xpos[i]
    return T


def load_depth_models(path=None, sensor=None, profile=None):
    cfg = _load(path, sensor, profile)
    return {cam: DepthModel(cfg[cam]) for cam in ('head', 'wrist') if 'depth' in cfg[cam]}


# holes are drawn by thresholding normal fields at a quantile, so the bridge's NoiseBank (normals only) works
def _ppf(p):
    """Standard-normal quantile (Acklam's rational approximation, |err| < 1.2e-9)."""
    p = np.clip(np.asarray(p, np.float64), 1e-9, 1 - 1e-9)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    q = np.where(p < 0.5, p, 1 - p)
    r = np.sqrt(-2 * np.log(q))
    tail = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    tail = np.where(p < 0.5, tail, -tail)
    u = p - 0.5; r2 = u * u
    mid = (((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * u / \
          (((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1)
    return np.where(np.abs(u) <= 0.47575, mid, tail)


class DepthModel:
    """Make MuJoCo's perfect depth look like the robot's depth topics.

        dm = load_depth_models()['wrist']
        z = dm.apply(z_m, lum, rng)       # z [m] float32 HxW (0 = no depth), lum = luminance of the ideal render

    Femto Bolt (`type: tof`, /femto/depth/aligned = ToF depth registered to the 1280x720 colour image):
      - depth only inside the depth camera's FOV (NFOV 75 x 65 deg vs colour 81 x 52): columns at the sides empty;
      - native 640x576 upsampled -> 2x2 blocks;
      - valid range (NFOV unbinned 0.5 .. 3.86 m); random error sigma = a + b*z;
      - flying pixels on depth edges (mixed returns): dropped or set between foreground and background;
      - no return from very dark surfaces (low IR reflectivity);
      - parallax shadow: background next to a foreground edge that the depth camera (baseline from the colour
        camera) cannot see has no depth after registration.
    D405 (`type: stereo`, /hand/camera/depth/image_rect_raw, passive stereo, no projector):
      - sigma_z = z^2 * sigma_disp / (f * B), spatially correlated (the SDK's spatial filter turns it into blobs);
        fitted: 1.9 mm at 0.5 m, 7 mm at 0.8 m on the real frame;
      - holes on texture-less surfaces (white walls / fronts) and more holes far away, as blobs;
      - occlusion band left of foreground edges (right imager cannot see it), foreground fattened by a few px;
      - nothing closer than min-z (7 cm: the gripper fingers), an invalid band at the left image edge.
    The constants live in config/camera_real.yaml (<cam>.depth); the Femto ones are from the datasheet (no real
    Femto depth frame recorded yet), the D405 ones from /nas/real/20260928_scene1/wrist_depth.png.
    """

    def __init__(self, cfg):
        self.c = cfg['depth']
        self.kind = self.c['type']
        self.W, self.H = cfg['width'], cfg['height']
        self.fx, self.fy, self.cx, self.cy = cfg['fx'], cfg['fy'], cfg['cx'], cfg['cy']
        self.u = np.arange(self.W, dtype=np.float32)[None, :]
        self.fov_mask = None
        if self.c.get('fov_deg'):                       # depth camera FOV inside the colour image
            hx, hy = (math.tan(math.radians(a / 2)) for a in self.c['fov_deg'])
            uu, vv = np.meshgrid(np.arange(self.W), np.arange(self.H))
            self.fov_mask = (np.abs(uu - self.cx) / self.fx <= hx) & (np.abs(vv - self.cy) / self.fy <= hy)

    def _field(self, rng, cell):
        """Smooth standard-normal-ish random field (blobs of ~cell px)."""
        h, w = max(self.H // cell, 2), max(self.W // cell, 2)
        f = rng.standard_normal((h, w)).astype(np.float32)
        return cv2.resize(f, (self.W, self.H), interpolation=cv2.INTER_CUBIC) * 1.6     # keep ~unit variance

    def _shadow(self, z, baseline, side):
        """Pixels the second camera (baseline [m] along image x, side +1 = right) cannot see: a background
        pixel is hidden if a pixel on the second camera's side lands on or past it in the second view."""
        zz = np.where(z > 0, z, np.inf)
        w = self.u - side * self.fx * baseline / zz                       # column in the second camera
        if side > 0:                                                       # look right: min over pixels to the right
            m = np.minimum.accumulate(w[:, ::-1], axis=1)[:, ::-1]
            m = np.concatenate([m[:, 1:], np.full((self.H, 1), np.inf, np.float32)], 1)
            return (m < w - 0.5) & (z > 0)
        m = np.maximum.accumulate(w, axis=1)
        m = np.concatenate([np.full((self.H, 1), -np.inf, np.float32), m[:, :-1]], 1)
        return (m > w + 0.5) & (z > 0)

    def apply(self, z, lum, rng):
        c = self.c
        z = z.astype(np.float32).copy()
        z[z <= 0] = 0
        valid0 = z > 0
        k3 = np.ones((3, 3), np.uint8)
        zmax_ = cv2.dilate(z, k3)
        zmin_ = cv2.erode(np.where(valid0, z, 1e3).astype(np.float32), k3)
        edge = valid0 & ((zmax_ - zmin_) > c.get('edge_jump_m', 0.04) + 0.03 * z)
        if self.kind == 'tof':
            b = c.get('block', 2)
            if b > 1:                                                        # native resolution
                z = cv2.resize(cv2.resize(z, (self.W // b, self.H // b), interpolation=cv2.INTER_NEAREST),
                               (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            a0, a1 = c['sigma_mm']
            n = cv2.resize(rng.standard_normal((self.H // b, self.W // b)).astype(np.float32), (self.W, self.H),
                           interpolation=cv2.INTER_NEAREST)
            z = z + n * (a0 + a1 * z * 1000) / 1000
            # flying pixels: a fraction dropped, the rest somewhere between the two surfaces
            fl = rng.standard_normal((self.H, self.W)).astype(np.float32)
            drop = edge & (fl < _ppf(c.get('flying_drop', 0.5)))
            mix = edge & ~drop
            tt = np.clip(0.5 + 0.3 * fl, 0, 1)
            z = np.where(mix, zmin_ + tt * (np.minimum(zmax_, zmin_ + 1.0) - zmin_), z)
            z[drop] = 0
            z[lum < c.get('dark_lum', 0)] = 0                                # no IR return from black surfaces
            if self.fov_mask is not None:
                z[~self.fov_mask] = 0
        else:
            f, B = self.fx, c['baseline_m']
            sig = np.sqrt((c.get('sigma_floor_mm', 0.8) / 1000) ** 2 + (z * z * c['sigma_disp_px'] / (f * B)) ** 2)
            corr = self._field(rng, c.get('blob_px', 24))
            fine = rng.standard_normal((self.H, self.W)).astype(np.float32)
            z = z + sig * (0.85 * corr + 0.5 * fine)
            # holes: texture-less (local std of the ideal render) and far, as blobs
            g = lum.astype(np.float32)
            mu = cv2.blur(g, (15, 15))
            sd = np.sqrt(np.maximum(cv2.blur(g * g, (15, 15)) - mu * mu, 0))
            p = c.get('lowtex_p', 0.4) * np.clip(1 - sd / c.get('lowtex_std', 3.0), 0, 1)
            z0, z1 = c.get('far_hole_z', [1.5, 6.0])
            p = np.maximum(p, c.get('far_hole_p', 0.6) * np.clip((z - z0) / (z1 - z0), 0, 1))
            hp = c.get('hole_px', 20)                                       # irregular blobs: two scales
            hf = 0.6 * self._field(rng, hp) + 0.8 * self._field(rng, 3 * hp)
            holes = hf < _ppf(np.clip(p, 1e-4, 0.999)).astype(np.float32)
            # foreground fattening at edges (the spatial filter / census window bleeds the near surface)
            fp = c.get('edge_fatten_px', 2)
            if fp > 0:
                band = cv2.dilate(edge.astype(np.uint8), np.ones((2 * fp + 1, 2 * fp + 1), np.uint8)) > 0
                zf = cv2.erode(np.where(z > 0, z, 1e3).astype(np.float32), np.ones((2 * fp + 1, 2 * fp + 1), np.uint8))
                z = np.where(band & (zf < 1e3), zf, z)
            z[holes & valid0] = 0
            if c.get('left_band_px'):
                z[:, :c['left_band_px']] = 0
        if c.get('baseline_shadow_m'):                                        # parallax / occlusion band
            z[self._shadow(np.where(valid0, z, 0), c['baseline_shadow_m'], c.get('shadow_side', 1))] = 0
        lo, hi = c['range_m']
        z[(z < lo) | (z > hi)] = 0
        z[~valid0] = 0
        return z


def mjcf_intrinsics(cam_cfg):
    """MJCF camera attributes reproducing the real pinhole (fx, fy, cx, cy).
    MuJoCo's principalpixel is (W/2 - cx, H/2 - cy) — checked by projecting a known point."""
    w, h = cam_cfg['width'], cam_cfg['height']
    return dict(resolution=f'{w} {h}', sensorsize=f'{w / 1000:.3f} {h / 1000:.3f}',
                focalpixel=f"{cam_cfg['fx']:.3f} {cam_cfg['fy']:.3f}",
                principalpixel=f"{w / 2 - cam_cfg['cx']:.3f} {h / 2 - cam_cfg['cy']:.3f}")
