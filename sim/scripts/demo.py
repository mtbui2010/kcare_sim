#!/usr/bin/env python3
"""Short, labelled videos for the step-by-step demo (docs/demo/kaair_demo.ipynb).

    python3 scripts/demo.py expert --drawer top --seed 7
        the scripted expert opens the chosen drawer        -> /sim/out/demo/expert_<drawer>_seed<seed>.mp4
    .venv/bin/python scripts/demo.py run --ckpt <.../pretrained_model> --prompt "open the top drawer" --seed 7
        the trained VLA gets only images + the sentence     -> /sim/out/demo/run_<slug>_seed<seed>.mp4

Every frame: an overview camera (not seen by the robot) above the head | wrist
images the policy actually receives (camera model applied, as in the dataset),
plus the sentence and how far each drawer is open. Scenes are the real-like
ones (task_open_drawer.yaml + randomization_real.yaml); --seed picks one.
"""
import argparse, math, os, re, sys

import cv2
import imageio
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import camera_model, check_sim as cs, drawer_ready as dr, expert_open_drawer as ex, randomize  # noqa: E402

NAMES = {0: 'bottom', 1: 'middle', 2: 'top'}
IDS = {v: k for k, v in NAMES.items()}
OUT = f'{SIM}/out/demo'
HEAD_WH, WRIST_WH = (1280, 720), (848, 480)


def drawer_of(prompt, rcfg):
    """Which drawer a sentence asks for: the training phrasings first, then keywords (None = unknown)."""
    for t, ps in rcfg['drawer_targets']['prompts'].items():
        if prompt.strip().lower() in ps:
            return int(t)
    p = prompt.lower()
    for t, keys in ((2, ('top', 'first', '1st', 'upper')), (0, ('bottom', 'third', '3rd', 'lowest', 'last')),
                    (1, ('middle', 'second', '2nd'))):
        if any(re.search(rf'\b{k}\b', p) for k in keys):
            return t
    return None


class Scene:
    def __init__(self, seed, target=None, prompt=None):
        self.cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
        self.task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
        self.rcfg = yaml.safe_load(open(f'{SIM}/config/randomization_real.yaml'))
        self.cams = camera_model.load_camera_models()
        self.rng = np.random.default_rng(seed)
        self.rand = randomize.sample(self.rng, self.rcfg)
        for c in self.cams.values():
            c.episode_jitter(self.rng)
        if target is not None:
            self.rand['target'] = target
        if prompt is not None:
            self.rand['prompt'] = prompt
        q_nom = dr.ready_q(self.task, self.task['mode'], self.cfg['home'])
        tcp_T = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf').fk(q_nom)['tool_tcp_link']
        xml, *_ = dr.build_scene(self.task, tcp_T, rand=self.rand)
        path = f'{SIM}/mjcf/_demo_scene_{os.getpid()}.xml'
        open(path, 'w').write(xml)
        self.m = mujoco.MjModel.from_xml_path(path); self.d = mujoco.MjData(self.m)
        os.remove(path)
        n = sum(1 for i in range(self.m.njnt) if re.fullmatch(r'cab_drawer\d_slide', self.m.joint(i).name))
        self.slides = [self.m.jnt_qposadr[self.m.joint(f'cab_drawer{i}_slide').id] for i in range(n)]
        # overview: behind and beside the robot, looking at the cabinet
        cab = self.d.body('cab').xpos if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'cab') >= 0 else [0.6, -0.4, 0.4]
        self.over = mujoco.MjvCamera()
        self.over.lookat[:] = [cab[0] * 0.6, cab[1] * 0.6, 0.55]
        self.over.distance, self.over.azimuth, self.over.elevation = 2.2, math.degrees(math.atan2(cab[1], cab[0])) + 140, -20
        self.r_over = mujoco.Renderer(self.m, 480, 848)
        self.r_head = mujoco.Renderer(self.m, HEAD_WH[1], HEAD_WH[0])
        self.r_wrist = mujoco.Renderer(self.m, WRIST_WH[1], WRIST_WH[0])
        self.frames = []

    def opened(self):
        return [float(self.d.qpos[a]) for a in self.slides]

    def observe(self):
        """head (848x480 after the recorder's resize) and wrist, through the camera model."""
        self.r_head.update_scene(self.d, 'head_rgb'); head = self.r_head.render()
        self.r_wrist.update_scene(self.d, 'wrist_rgb'); wrist = self.r_wrist.render().copy()
        head = self.cams['head'].apply(head, self.rng); wrist = self.cams['wrist'].apply(wrist, self.rng)
        return cv2.resize(head, WRIST_WH, interpolation=cv2.INTER_AREA), wrist

    def frame(self, head, wrist, title, status):
        self.r_over.update_scene(self.d, self.over); over = self.r_over.render()
        bottom = np.concatenate([cv2.resize(head, (424, 240)), cv2.resize(wrist, (424, 240))], 1)
        f = np.concatenate([over, bottom], 0).copy()
        op = self.opened()
        lines = [(title, (255, 255, 0)), (status, (255, 255, 255)),
                 ('   '.join(f'{NAMES[i]} {op[i] * 100:4.1f} cm' for i in range(len(op))[::-1]), (180, 230, 255))]
        for i, (t, c) in enumerate(lines):
            cv2.putText(f, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(f, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        for x, t in ((10, 'robot head camera'), (434, 'robot wrist camera')):
            cv2.putText(f, t, (x, 480 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(f, t, (x, 480 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        self.frames.append(f)

    def save(self, name, tail_s=1.0):
        self.frames += [self.frames[-1]] * int(tail_s * ex.HZ)
        os.makedirs(OUT, exist_ok=True)
        path = f'{OUT}/{name}.mp4'
        imageio.mimwrite(path, self.frames, fps=ex.HZ, quality=6, macro_block_size=1)
        for r in (self.r_over, self.r_head, self.r_wrist):
            r.close()
        return path


def cmd_expert(a):
    target = IDS[a.drawer]
    sc = Scene(a.seed, target=target)
    prompt = sc.rand.get('prompt')

    def on_step(m, d, phase, drawer, **_):
        head, wrist = sc.observe()
        sc.frame(head, wrist, f'EXPERT (scripted, knows the handle position)  "{prompt}"', f'phase: {phase}')

    rec = ex.run_episode(sc.m, sc.d, sc.task, sc.cfg, sc.task['mode'], render=on_step, log=lambda *x: None,
                         q_extra=sc.rand['q_extra'], drawer0=sc.rand['drawer0'], target=target)
    path = sc.save(f'expert_{a.drawer}_seed{a.seed}')
    print(f"expert, {a.drawer} drawer: {'SUCCESS' if rec['success'] else 'FAIL'}, "
          f"opened {(rec['drawer'][-1] - sc.rand['drawer0']) * 100:.1f} cm, {len(rec['action'])} frames")
    print('->', path)


def cmd_run(a):
    import torch
    from eval_policy import ARM, load_policy, to_chw
    rcfg = yaml.safe_load(open(f'{SIM}/config/randomization_real.yaml'))
    target = drawer_of(a.prompt, rcfg)
    sc = Scene(a.seed, target=target if target is not None else None, prompt=a.prompt)
    policy, pre, post = load_policy(os.path.abspath(a.ckpt), 'cuda')
    policy.reset()
    m, d = sc.m, sc.d
    q_ready = dr.ready_q(sc.task, sc.task['mode'], sc.cfg['home'])
    for j, dv in sc.rand['q_extra'].items():
        q_ready[j] += dv
    n_sub = max(1, round(1.0 / ex.HZ / m.opt.timestep)); m.opt.timestep = 1.0 / ex.HZ / n_sub
    cs.set_q(m, d, q_ready, sc.cfg)
    if target is not None:
        d.qpos[sc.slides[target]] = sc.rand['drawer0']
    d.ctrl[:] = cs.ctrl_from(m, q_ready, sc.cfg)
    mujoco.mj_forward(m, d)
    start = sc.opened()
    ckpt_name = os.path.basename(os.path.dirname(os.path.realpath(a.ckpt)))   # 'last' -> 020000
    title = f'VLA (images + sentence only, step {ckpt_name})  "{a.prompt}"'
    t_done = None
    for step in range(int(a.timeout * ex.HZ)):
        head, wrist = sc.observe()
        state = [cs.get_q(m, d, j) for j in ARM] + [cs.get_q(m, d, 'virtual_gripper_joint')]
        obs = {'observation.state': torch.tensor(state, dtype=torch.float32),
               'observation.images.head': to_chw(head), 'observation.images.wrist': to_chw(wrist), 'task': a.prompt}
        with torch.inference_mode():
            act = post(policy.select_action(pre(obs))).squeeze(0).float().cpu().numpy()
        sc.frame(head, wrist, title, f't = {step / ex.HZ:4.1f} s' + ('   drawer open!' if t_done else ''))
        q_cmd = dict(q_ready, **dict(zip(ARM, act[:7])), virtual_gripper_joint=float(np.clip(act[7], 0.0, 0.1)))
        d.ctrl[:] = cs.ctrl_from(m, q_cmd, sc.cfg)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        gain = [o - s for o, s in zip(sc.opened(), start)]
        if t_done is None and max(gain) > 0.20:
            t_done = step
        if t_done is not None and step > t_done + ex.HZ:
            break
    gain = [o - s for o, s in zip(sc.opened(), start)]
    best = int(np.argmax(gain))
    if max(gain) < 0.20:
        verdict = 'FAIL: no drawer opened 20 cm'
    elif target is None:
        verdict = f'opened the {NAMES[best]} drawer (sentence did not name a drawer I know)'
    elif best == target and sorted(gain)[-2] < 0.03:
        verdict = f'SUCCESS: opened the {NAMES[best]} drawer as asked'
    else:
        verdict = f'WRONG: asked for {NAMES[target]}, opened {NAMES[best]}'
    for _ in range(int(1.5 * ex.HZ)):              # final card
        sc.frame(*sc.observe(), title, verdict)
    slug = re.sub(r'[^a-z0-9]+', '_', a.prompt.lower()).strip('_')[:40]
    path = sc.save(f'run_{slug}_seed{a.seed}', tail_s=0)
    print(verdict + '  |  ' + ', '.join(f'{NAMES[i]} {g * 100:+.1f} cm' for i, g in enumerate(gain)))
    print('->', path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('expert'); e.add_argument('--drawer', choices=list(IDS), default='middle')
    e.add_argument('--seed', type=int, default=7)
    r = sub.add_parser('run'); r.add_argument('--ckpt', required=True); r.add_argument('--prompt', required=True)
    r.add_argument('--seed', type=int, default=7); r.add_argument('--timeout', type=float, default=15.0)
    a = ap.parse_args()
    {'expert': cmd_expert, 'run': cmd_run}[a.cmd](a)


if __name__ == '__main__':
    main()
