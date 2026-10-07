#!/usr/bin/env python3
"""Scripted expert for "open drawer", from the kcare ready pose.

    python3 expert_open_drawer.py            # one episode -> out/open_drawer/episode_sample.{mp4,npz}

Phases (TCP targets in world frame, d = approach direction into the cabinet):
  hold      keep the ready pose
  pregrasp  turn the wrist so the fingers close vertically on the horizontal
            bar, pre-close the gripper, stop `pre_d` in front of the bar
  approach  slide in until the bar sits between the finger pads
  grasp     close the gripper
  pull      pull straight back along -d by `pull` m
  release   open the gripper
  retreat   back off along -d

The expert runs at the dataset rate (30 Hz). Each control step it solves IK for
the next TCP waypoint (seeded from the last solution, so joints stay on one
branch) and commands joint positions + gripper width — the same 8-D action the
real robot records (joint1..7 + gripper_width [m]). Lift and head stay at the
ready pose.
"""
import math, os, sys
import imageio, mujoco, numpy as np, yaml
from scipy.optimize import least_squares
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
import cabinet, check_sim as cs, drawer_ready as dr

SIM = os.environ.get('SIM_ROOT', '/sim')
HZ = 30
ARM = [f'joint{i}' for i in range(1, 8)]

EXPERT = dict(pre_d=0.08, grasp_depth=0.02, pull=0.25, retreat=0.08,
              pre_width=0.04, closed_width=0.0, finger_axis_z=1,
              durations=dict(hold=0.5, pregrasp=2.5, approach=1.5, grasp=1.5, pull=3.0, release=1.0, retreat=1.2))
# grasp / release command the gripper as a step, like kcare's grip::close/open;
# the Koras closes at ~0.05 m/s, so the phase must outlast the travel. A ramped
# command let the pull start before the fingers had reached the bar.
STEP_GRIPPER = ('grasp', 'release')


class ArmIK:
    def __init__(self, ref, q_fixed):
        self.ref, self.q_fixed = ref, dict(q_fixed)
        self.lo = np.array([ref.joints[j]['limit'][0] for j in ARM])
        self.hi = np.array([ref.joints[j]['limit'][1] for j in ARM])

    def tcp(self, x):
        return self.ref.fk(dict(self.q_fixed, **dict(zip(ARM, x))))['tool_tcp_link']

    def solve(self, pos, z_axis, y_axis, x0):
        def res(x):
            T = self.tcp(x)
            return np.concatenate([(T[:3, 3] - pos) * 20, (T[:3, 2] - z_axis) * 2, (T[:3, 1] - y_axis) * 2,
                                   (x - x0) * 0.02])      # stay close to the previous solution
        sol = least_squares(res, np.clip(x0, self.lo, self.hi), bounds=(self.lo, self.hi), xtol=1e-10, ftol=1e-10)
        T = self.tcp(sol.x)
        return sol.x, np.linalg.norm(T[:3, 3] - pos)


def smooth(s):  # 0..1 -> 0..1, zero velocity at both ends
    return s * s * (3 - 2 * s)


def run_episode(m, d, task, cfg, mode='left', render=None, log=print, q_extra=None, drawer0=0.0, target=None):
    """q_extra: joint offsets on top of the ready pose (e.g. head jitter);
    drawer0: initial opening of the target drawer [m];
    target: drawer index to open (0 = bottom ... top); None = the style's default."""
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    q_ready = dr.ready_q(task, mode, cfg['home'])
    for j, dv in (q_extra or {}).items():
        q_ready[j] += dv
    ik = ArmIK(ref, q_ready)
    n_sub = max(1, round(1.0 / HZ / m.opt.timestep))
    m.opt.timestep = 1.0 / HZ / n_sub

    cs.set_q(m, d, q_ready, cfg)
    d.ctrl[:] = cs.ctrl_from(m, q_ready, cfg)
    mujoco.mj_forward(m, d)

    drawer = f"cab_drawer{cabinet.STYLES[task['cabinet']]['target'] if target is None else target}"
    slide = m.joint(f'{drawer}_slide')
    slide_adr = m.jnt_qposadr[slide.id]
    d.qpos[slide_adr] = drawer0
    mujoco.mj_forward(m, d)

    T0 = ref.fk(q_ready)['tool_tcp_link']
    # approach = into the cabinet = against the drawer's pull axis (the cabinet may be yawed)
    pull_axis = d.xmat[m.jnt_bodyid[slide.id]].reshape(3, 3) @ m.jnt_axis[slide.id]
    d_app = -pull_axis; d_app[2] = 0; d_app /= np.linalg.norm(d_app)
    handle = d.geom_xpos[m.geom(f'{drawer}_handle').id].copy()

    # grasp orientation: tool z into the cabinet, fingers vertical (closing on a
    # horizontal bar). The finger axis sign is FIXED, not "whichever wrist turn is
    # shorter": that choice flipped with the cabinet yaw and gave the dataset a
    # +90 deg and a -90 deg wrist mode for the same scene.
    z_goal = d_app
    x_ready = np.array([q_ready[j] for j in ARM])
    y_goal = np.array([0, 0, float(EXPERT['finger_axis_z'])])

    grasp_pos = handle + d_app * EXPERT['grasp_depth']
    pre_pos = handle - d_app * EXPERT['pre_d']
    pull_pos = grasp_pos - d_app * EXPERT['pull']
    retreat_pos = pull_pos - d_app * EXPERT['retreat']
    open_w = task['ready_pose']['gripper']
    # (phase, TCP from, TCP to, unused, gripper width at the end of the phase)
    phases = [
        ('hold', T0[:3, 3], T0[:3, 3], 0.0, open_w),
        ('pregrasp', T0[:3, 3], pre_pos, 1.0, EXPERT['pre_width']),
        ('approach', pre_pos, grasp_pos, 1.0, EXPERT['pre_width']),
        ('grasp', grasp_pos, grasp_pos, 1.0, EXPERT['closed_width']),
        ('pull', grasp_pos, pull_pos, 1.0, EXPERT['closed_width']),
        ('release', pull_pos, pull_pos, 1.0, EXPERT['pre_width']),
        ('retreat', pull_pos, retreat_pos, 1.0, EXPERT['pre_width']),
    ]

    rec = dict(action=[], state=[], drawer=[], phase=[])
    x = x_ready.copy()
    width_cmd = open_w
    for name, p_from, p_to, _, w_end in phases:
        steps = int(round(EXPERT['durations'][name] * HZ))
        w_start = width_cmd
        ik_err = 0.0
        for k in range(steps):
            s = smooth((k + 1) / steps)
            pos = p_from + (p_to - p_from) * s
            if name == 'hold':
                x_tgt = x_ready
            elif name == 'pregrasp':
                # blend orientation from the ready pose to the grasp orientation
                z = T0[:3, 2] * (1 - s) + z_goal * s; z /= np.linalg.norm(z)
                y = T0[:3, 1] * (1 - s) + y_goal * s; y -= z * np.dot(y, z); y /= np.linalg.norm(y)
                x_tgt, err = ik.solve(pos, z, y, x)
                ik_err = max(ik_err, err)
            else:
                x_tgt, err = ik.solve(pos, z_goal, y_goal, x)
                ik_err = max(ik_err, err)
            x = x_tgt
            width_cmd = w_end if name in STEP_GRIPPER else w_start + (w_end - w_start) * s

            q_cmd = dict(q_ready, **dict(zip(ARM, x)), virtual_gripper_joint=width_cmd)
            d.ctrl[:] = cs.ctrl_from(m, q_cmd, cfg)
            state = [cs.get_q(m, d, j) for j in ARM] + [cs.get_q(m, d, 'virtual_gripper_joint')]
            rec['state'].append(state)
            rec['action'].append(list(x) + [width_cmd])
            rec['phase'].append(name)
            if render:   # observation at this step + the action about to be applied
                render(m, d, name, float(d.qpos[slide_adr]), state=state, action=rec['action'][-1])
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            rec['drawer'].append(float(d.qpos[slide_adr]))
        tcp_now = d.body('tool_tcp_link').xpos
        log(f'  {name:9s} done: drawer open {d.qpos[slide_adr] * 100:5.1f} cm, '
            f'gripper {cs.get_q(m, d, "virtual_gripper_joint") * 1000:5.1f} mm, '
            f'IK err {ik_err * 1000:4.1f} mm, TCP tracking err {np.linalg.norm(tcp_now - p_to) * 1000:5.1f} mm, '
            f'contacts {sorted({m.body(m.geom_bodyid[c.geom1]).name + "|" + m.body(m.geom_bodyid[c.geom2]).name for c in d.contact[:d.ncon]})}')

    rec = {k: np.array(v) for k, v in rec.items()}
    rec['success'] = bool(rec['drawer'][-1] - drawer0 > 0.8 * EXPERT['pull'])
    return rec


class VideoComposer:
    """Overview (top) + head / wrist cameras (bottom) at the dataset resolutions, scaled into one frame."""

    def __init__(self, m, cam):
        self.cam = cam
        self.r_over = mujoco.Renderer(m, 540, 960)
        self.r_head = mujoco.Renderer(m, 720, 1280)
        self.r_wrist = mujoco.Renderer(m, 480, 848)
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
        self.frames = []

    def __call__(self, m, d, phase, drawer, **_):
        self.r_over.update_scene(d, self.cam); over = Image.fromarray(self.r_over.render())
        self.r_head.update_scene(d, 'head_rgb'); head = Image.fromarray(self.r_head.render()).resize((640, 360))
        self.r_wrist.update_scene(d, 'wrist_rgb'); wrist = Image.fromarray(self.r_wrist.render()).resize((636, 360))
        canvas = Image.new('RGB', (1280, 900), (20, 20, 20))
        canvas.paste(over.resize((1280, 540)), (0, 0))
        canvas.paste(head, (0, 540)); canvas.paste(wrist, (644, 540))
        dr_ = ImageDraw.Draw(canvas)
        kw = dict(font=self.font, stroke_width=2, stroke_fill=(0, 0, 0))
        dr_.text((14, 10), f'open drawer  |  phase: {phase:9s} |  drawer {drawer * 100:4.1f} cm  |  t = {len(self.frames) / HZ:4.1f} s',
                 fill=(255, 255, 255), **kw)
        dr_.text((10, 546), 'head_rgb 1280x720', fill=(255, 255, 0), **kw)
        dr_.text((654, 546), 'wrist_rgb 848x480', fill=(255, 255, 0), **kw)
        self.frames.append(np.asarray(canvas))


def main():
    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
    mode = task['mode']
    path = f'{SIM}/mjcf/scene_open_drawer_{mode}.xml'
    if not os.path.exists(path):
        raise SystemExit('run drawer_ready.py first')
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # same framing as drawer_ready.py: from the robot's side, looking along the approach
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.4, -0.35, 0.5]
    cam.distance, cam.azimuth, cam.elevation = 2.0, -141.7, -22
    video = VideoComposer(m, cam)
    rec = run_episode(m, d, task, cfg, mode, render=video)
    out = f'{SIM}/out/open_drawer'
    imageio.mimwrite(f'{out}/episode_sample.mp4', video.frames, fps=HZ, quality=7, macro_block_size=1)
    np.savez(f'{out}/episode_sample.npz', **{k: v for k, v in rec.items() if k != 'success'})
    print(f"success={rec['success']}  frames={len(video.frames)}  drawer final {rec['drawer'][-1] * 100:.1f} cm  "
          f"action {rec['action'].shape} state {rec['state'].shape}")
    print('->', f'{out}/episode_sample.mp4')


if __name__ == '__main__':
    main()
