#!/usr/bin/env python3
"""Closed-loop evaluation of a trained policy on the MuJoCo "open drawer" task.

    CUDA_VISIBLE_DEVICES=3 /sim/.venv/bin/python scripts/eval_policy.py \
        --ckpt /nas/checkpoints/<run>/checkpoints/last/pretrained_model --n 50
      -> /nas/checkpoints/<run>/eval/<step>_results.json   (per-episode results + success rate)
         /sim/out/eval/<run>_<step>_ep<k>.mp4             (first --videos rollouts, head | wrist)

Each episode: a randomized scene (randomize.py) from seeds the generator never
uses (10**9 + k), the robot at the kcare ready pose, then the policy drives the
arm at 30 Hz with exactly the dataset's observation (head 1280x720 ->
INTER_AREA 848x480, wrist 848x480, 8-D state, task "open drawer"). Its 8-D
action (joint1..7 + gripper_width) goes through the same position servos the
expert used; lift and head stay at the ready pose. Success = drawer pulled
more than --success-m within --timeout s.
"""
import argparse, json, os, sys, time
import cv2, imageio, mujoco, numpy as np, torch, yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import cabinet, camera_model, check_sim as cs, drawer_ready as dr, expert_open_drawer as ex, randomize  # noqa: E402
from gen_dataset import IMG_WH, TASK  # noqa: E402

ARM = ex.ARM


def load_policy(ckpt, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt,
                                         preprocessor_overrides={'device_processor': {'device': device}})
    return policy, pre, post


def to_chw(img):
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def run(m, d, task, cfg, rand, policy, pre, post, timeout, success_m, frames=None, cams=None, rng=None):
    q_ready = dr.ready_q(task, task['mode'], cfg['home'])
    for j, dv in rand['q_extra'].items():
        q_ready[j] += dv
    n_sub = max(1, round(1.0 / ex.HZ / m.opt.timestep))
    m.opt.timestep = 1.0 / ex.HZ / n_sub
    target = rand.get('target', cabinet.STYLES[task['cabinet']]['target'])
    drawer = f'cab_drawer{target}'
    n_dr = len(cabinet.STYLES[task['cabinet']]['drawers'])
    others = [m.jnt_qposadr[m.joint(f'cab_drawer{i}_slide').id] for i in range(n_dr) if i != target]
    adr = m.jnt_qposadr[m.joint(f'{drawer}_slide').id]
    cs.set_q(m, d, q_ready, cfg)
    d.qpos[adr] = rand['drawer0']
    d.ctrl[:] = cs.ctrl_from(m, q_ready, cfg)
    mujoco.mj_forward(m, d)

    r_head = mujoco.Renderer(m, 720, 1280)
    r_wrist = mujoco.Renderer(m, IMG_WH[1], IMG_WH[0])
    policy.reset()
    opened, t_success = 0.0, None
    try:
        for step in range(int(timeout * ex.HZ)):
            r_head.update_scene(d, 'head_rgb')
            head = r_head.render()
            r_wrist.update_scene(d, 'wrist_rgb')
            wrist = r_wrist.render().copy()
            if cams is not None:                     # same camera model as the dataset
                head = cams['head'].apply(head, rng, cam_T=camera_model.cam_pose(m, d, 'head_rgb'), t=d.time)
                wrist = cams['wrist'].apply(wrist, rng, cam_T=camera_model.cam_pose(m, d, 'wrist_rgb'), t=d.time)
            head = cv2.resize(head, IMG_WH, interpolation=cv2.INTER_AREA)
            state = [cs.get_q(m, d, j) for j in ARM] + [cs.get_q(m, d, 'virtual_gripper_joint')]
            obs = {'observation.state': torch.tensor(state, dtype=torch.float32),
                   'observation.images.head': to_chw(head), 'observation.images.wrist': to_chw(wrist),
                   'task': rand.get('prompt', TASK)}
            with torch.inference_mode():
                action = post(policy.select_action(pre(obs)))
            a = action.squeeze(0).float().cpu().numpy()
            q_cmd = dict(q_ready, **dict(zip(ARM, a[:7])), virtual_gripper_joint=float(np.clip(a[7], 0.0, 0.1)))
            d.ctrl[:] = cs.ctrl_from(m, q_cmd, cfg)
            if frames is not None:
                frames.append(np.concatenate([cv2.resize(head, (424, 240)), cv2.resize(wrist, (424, 240))], 1))
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            opened = float(d.qpos[adr] - rand['drawer0'])
            done = d.qpos[adr] < 0.02 if task.get('expert') == 'close_drawer' else opened > success_m   # close: shut
            if t_success is None and done:
                t_success = (step + 1) / ex.HZ
            if t_success is not None and (step + 1) / ex.HZ > t_success + 1.0:   # show 1 s after success
                break
    finally:
        r_head.close(); r_wrist.close()
    wrong = max([float(d.qpos[a]) for a in others], default=0.0)          # a drawer that was not asked for
    return dict(success=t_success is not None and wrong < 0.03, opened=opened, t_success=t_success,
                target=int(target), prompt=rand.get('prompt', TASK), wrong_drawer_m=wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='.../checkpoints/<step>/pretrained_model')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--videos', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=15.0)
    ap.add_argument('--success-m', type=float, default=0.20)
    ap.add_argument('--seed0', type=int, default=10 ** 9, help='far from the generator seeds')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--task', default='task_open_drawer.yaml', help='config/ file; task_open_drawer_generic.yaml = '
                    'the scenes of the 2026-09-24 runs')
    ap.add_argument('--rand-config', default='randomization.yaml', help='config/ file the scenes are drawn from')
    ap.add_argument('--camera-model', action='store_true', help='pass renders through camera_model.py')
    ap.add_argument('--tag', default='', help='suffix for the results file, e.g. "real" -> <step>_real_results.json')
    a = ap.parse_args()

    ckpt = os.path.abspath(a.ckpt)
    step = os.path.basename(os.path.dirname(ckpt))                    # e.g. 020000 / last
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt)))
    run_name = os.path.basename(run_dir)
    eval_dir = os.path.join(run_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(f'{SIM}/out/eval', exist_ok=True)

    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/{a.task}'))
    rcfg = yaml.safe_load(open(f'{SIM}/config/{a.rand_config}'))
    cams = None
    if a.camera_model:
        import camera_model
        cams = camera_model.load_camera_models(profile=task.get('camera_profile'))
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    tcp_T = ref.fk(dr.ready_q(task, task['mode'], cfg['home']))['tool_tcp_link']
    policy, pre, post = load_policy(ckpt, a.device)

    scene_path = f'{SIM}/mjcf/_eval_scene_{os.getpid()}.xml'
    results = []
    t0 = time.time()
    try:
        for k in range(a.n):
            seed = a.seed0 + k
            ep_rng = np.random.default_rng(seed)
            rand = randomize.sample(ep_rng, rcfg)
            if task.get('expert') == 'close_drawer':      # as gen_dataset: starts open, "close ..." phrasings
                rand['drawer0'] = float(ep_rng.uniform(*task.get('drawer0_range', (0.12, 0.25))))
                rand['prompt'] = rand.get('prompt', 'open drawer').replace('open', 'close')
            if cams is not None:
                for c in cams.values():
                    c.episode_jitter(ep_rng)
            xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
            open(scene_path, 'w').write(xml)
            m = mujoco.MjModel.from_xml_path(scene_path)
            d = mujoco.MjData(m)
            frames = [] if k < a.videos else None
            r = run(m, d, task, cfg, rand, policy, pre, post, a.timeout, a.success_m, frames, cams, ep_rng)
            r['seed'] = seed
            results.append(r)
            if frames:
                imageio.mimwrite(f'{SIM}/out/eval/{run_name}_{step}{"_" + a.tag if a.tag else ""}_ep{k}.mp4', frames, fps=ex.HZ,
                                 quality=6, macro_block_size=1)
            sr = np.mean([x['success'] for x in results])
            print(f"[{time.time() - t0:5.0f}s] ep {k:3d} {'OK  ' if r['success'] else 'FAIL'} "
                  f"opened {r['opened'] * 100:5.1f} cm   success so far {sr * 100:.0f}% ({len(results)})", flush=True)
    finally:
        if os.path.exists(scene_path):
            os.remove(scene_path)
    summary = dict(ckpt=ckpt, n=len(results), success_rate=float(np.mean([x['success'] for x in results])),
                   mean_opened_m=float(np.mean([x['opened'] for x in results])),
                   timeout_s=a.timeout, success_m=a.success_m, seed0=a.seed0, task=a.task,
                   rand_config=a.rand_config, camera_model=a.camera_model, episodes=results)
    out = f'{eval_dir}/{step}_{a.tag}_results.json' if a.tag else f'{eval_dir}/{step}_results.json'
    json.dump(summary, open(out, 'w'), indent=2)
    print(f"success rate {summary['success_rate'] * 100:.1f}% over {summary['n']} episodes, "
          f"mean opened {summary['mean_opened_m'] * 100:.1f} cm -> {out}")


if __name__ == '__main__':
    main()
