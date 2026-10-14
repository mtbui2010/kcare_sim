#!/usr/bin/env python3
"""Closed-loop evaluation of a PICK phase-A ("reach to view") policy served by vla-serve, in the house episodes of
pick_scene (the same sampler as the dataset, held-out seeds 10**9 + k).

    # a policy on a vla-serve server (the model it has loaded, or --model)
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 /sim/.venv/bin/python scripts/eval_pick_policy.py \
        --url http://127.0.0.1:11451 --n 50 --out /nas/checkpoints/<run>/eval/<step>_pick_results.json
    # the scripted expert on the same seeds (reference)
    ... scripts/eval_pick_policy.py --expert --n 50 --out /sim/out/eval/pick_expert_results.json

Per episode: the scene / robot stop / objects of pick_scene.sample(seed), the mode's start pose and the expert's start
checks (expert_pick_approach.setup); then 30 Hz control in SIM time: the dataset's observation (head 1280x720 ->
camera model -> INTER_AREA 848x480, wrist 848x480, 17-D state, the episode's prompt) is sent to the server with
PolicyClient.act, and the returned chunk's first n_action_steps actions are executed one per step before the next
request (synchronous: the simulation waits for the policy, so the result does not depend on inference speed — the
ChunkExecutor's wall-clock scheduling is for the robot). The 10-D action goes through the expert's own stepping
(apply_step): arm + gripper + lift position targets (lift limited to 0.08 m/s, gripper clipped to 0-0.1 m), base
forward target clamped to +-0.2 m and moved at <= 0.15 m/s along the heading. Stops when the commanded action has
been still for 0.5 s (after >= 2 s) or at --timeout. Scored by expert_pick_approach.score (target centred in the
wrist view, finger tips 0.08-0.18 m, no contact / clearance violation, target not moved, arm tracking).
"""
import argparse, collections, json, math, os, sys, time

import cv2
import mujoco
import numpy as np
import yaml

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
sys.path.insert(0, os.path.join(SIM, 'third_party'))
sys.path.append(os.path.join(SIM, 'third_party', 'vlaclient_deps'))     # msgpack / httpx for vlaserve.client (appended: the venv's own packages win)
import expert_pick_approach as ex  # noqa: E402

IMG_WH = (848, 480)
LIFT_LIM = (0.13, 0.67)                   # lift_joint: kcare lift 0.56 (its floor) .. URDF top
LIFT_SPEED, BASE_SPEED, BASE_MAX = 0.08, 0.15, 0.2
START_FAIL = ('robot starts in contact', 'target fell off', 'target hidden')   # ex.setup's refusals


def observe(m, d, ctx, r_head, r_wrist, cams, rng):
    r_head.update_scene(d, 'head_rgb')
    head = r_head.render()
    r_wrist.update_scene(d, 'wrist_rgb')
    wrist = r_wrist.render().copy()
    if cams is not None:
        import camera_model
        head = cams['head'].apply(head, rng, cam_T=camera_model.cam_pose(m, d, 'head_rgb'), t=d.time)
        wrist = cams['wrist'].apply(wrist, rng, cam_T=camera_model.cam_pose(m, d, 'wrist_rgb'), t=d.time)
    head = cv2.resize(head, IMG_WH, interpolation=cv2.INTER_AREA)
    return head, wrist, np.array(ex.state_vec(m, d, ctx), np.float32)


def run_policy(m, d, task, cfg, rand, info, client, a, cams, rng, frames=None):
    ctx, why = ex.setup(m, d, task, cfg, rand, info)
    if ctx is None:
        return dict(success=False, skipped=True, reason=why)
    q0 = ctx['q0']
    r_head, r_wrist = mujoco.Renderer(m, 720, 1280), mujoco.Renderer(m, IMG_WH[1], IMG_WH[0])
    client.reset()
    lift_cmd, base_cmd = q0['lift_joint'], 0.0
    hist, last, chunks, rt = [], None, 0, []
    queue = []
    t0 = time.time()
    try:
        for step in range(int(a.timeout * ex.HZ)):
            head, wrist, state = observe(m, d, ctx, r_head, r_wrist, cams, rng)
            if frames is not None:
                frames.append(np.concatenate([cv2.resize(head, (424, 240)), cv2.resize(wrist, (424, 240))], 1))
            if not queue:
                ch = client.act(images={'head': head, 'wrist': wrist}, state=state, task=rand['prompt'])
                n_exec = a.execute or ch.n_action_steps or len(ch.actions)
                queue = [np.asarray(x, float) for x in ch.actions[:n_exec]]
                chunks += 1; rt.append(ch.roundtrip_ms)
            act = queue.pop(0)
            x = act[:7]
            grip = float(np.clip(act[7], 0.0, 0.1))
            lift_cmd += float(np.clip(np.clip(act[8], *LIFT_LIM) - lift_cmd, -LIFT_SPEED / ex.HZ, LIFT_SPEED / ex.HZ))
            base_cmd += float(np.clip(np.clip(act[9], -BASE_MAX, BASE_MAX) - base_cmd, -BASE_SPEED / ex.HZ, BASE_SPEED / ex.HZ))
            cmd = np.concatenate([x, [grip, lift_cmd, base_cmd]])
            ex.apply_step(m, d, ctx, cfg, x, grip, lift_cmd, base_cmd)
            hist.append(cmd); last = cmd
            if step >= 2 * ex.HZ and len(hist) > 15:      # the command has been still for 0.5 s: the policy is done
                w = np.array(hist[-15:])
                if (np.ptp(w[:, :7], 0).max() < 0.003 and np.ptp(w[:, 8]) < 0.001 and np.ptp(w[:, 9]) < 0.001):
                    break
    finally:
        r_head.close(); r_wrist.close()
    res = ex.score(m, d, ctx, list(last))
    base_used = float(np.max(np.abs(np.array(hist)[:, 9]))) if hist else 0.0
    res.update(skipped=False, t_end=round(len(hist) / ex.HZ, 2), chunks=chunks,
               roundtrip_ms=round(float(np.median(rt)), 1) if rt else None, base_used=round(base_used, 3),
               wall_s=round(time.time() - t0, 1))
    return res


def run_expert(m, d, task, cfg, rand, info):
    rec = ex.run_episode(m, d, task, cfg, rand=rand, info=info)
    out = {k: v for k, v in rec.items() if k not in ('action', 'state', 'phase')}
    out['skipped'] = str(rec['reason']).startswith(START_FAIL)     # same start checks as the policy runs
    out.update(t_end=round(len(rec['action']) / ex.HZ, 2),
               base_used=round(abs(float(rec['plan']['base'])), 3) if rec.get('plan') else 0.0)
    return out


class HoldClient:
    """--mock: a stand-in policy that holds the start pose (exercises the policy path without a server)."""
    model = 'hold-still mock'

    def reset(self):
        pass

    def act(self, images, state, task):
        from types import SimpleNamespace
        a = np.concatenate([state[:7], [0.1, state[8], state[11]]])
        return SimpleNamespace(actions=np.tile(a, (50, 1)), n_action_steps=50, roundtrip_ms=0.0)


def table(rows, key):
    c = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        c[key(r)][1] += 1; c[key(r)][0] += bool(r['success'])
    return {str(k): f'{s}/{n}' for k, (s, n) in sorted(c.items(), key=lambda kv: str(kv[0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:11451')
    ap.add_argument('--model', default=None, help='served model id (default: the first one loaded)')
    ap.add_argument('--expert', action='store_true', help='score the scripted expert instead (reference)')
    ap.add_argument('--mock', action='store_true', help='a hold-still stand-in policy (tests the policy path)')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed0', type=int, default=10 ** 9, help='held out: the generator uses small seeds')
    ap.add_argument('--timeout', type=float, default=12.0)
    ap.add_argument('--execute', type=int, default=0, help='actions executed per chunk (0 = the model n_action_steps)')
    ap.add_argument('--task', default='task_pick_approach.yaml')
    ap.add_argument('--rand-config', default='', help='config/ file (default: the task rand_config)')
    ap.add_argument('--no-camera-model', action='store_true')
    ap.add_argument('--videos', type=int, default=3)
    ap.add_argument('--out', required=True, help='results json')
    a = ap.parse_args()

    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(f'{SIM}/config/{a.task}'))
    rcfg = yaml.safe_load(open(f'{SIM}/config/{a.rand_config}' if a.rand_config else task['rand_config']))
    cams = None
    if not a.no_camera_model:
        import camera_model
        cams = camera_model.load_camera_models(profile=task.get('camera_profile'))
    client = HoldClient() if a.mock else None
    if not a.expert and not a.mock:
        from vlaserve.client import PolicyClient
        client = PolicyClient(a.url, model=a.model)
        if not a.model:
            import httpx
            client.model = httpx.get(f'{a.url}/v1/models', timeout=10).json()['models'][0]['model']
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    vid_dir = f'{SIM}/out/eval'
    os.makedirs(vid_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.out))[0]
    rows, t0 = [], time.time()
    for k in range(a.n):
        seed = a.seed0 + k
        rng = np.random.default_rng(seed)
        rand = ex.sample_rand(rng, rcfg, task)
        if cams is not None:
            for c in cams.values():
                c.episode_jitter(rng)
        m, d, info = ex.episode_model(task, rand)
        frames = [] if (k < a.videos and not a.expert) else None
        r = run_expert(m, d, task, cfg, rand, info) if a.expert else run_policy(m, d, task, cfg, rand, info, client, a,
                                                                                cams, rng, frames)
        r.update(seed=seed, spot=rand['spot'], mode=rand['mode'], band=rand['band'], base_band=rand['base_band'],
                 target=rand['target'], prompt=rand['prompt'])
        rows.append(r)
        if frames:
            import imageio
            imageio.mimwrite(f'{vid_dir}/{tag}_ep{k}.mp4', frames, fps=ex.HZ, quality=6, macro_block_size=1)
        valid = [x for x in rows if not x['skipped']]
        sr = np.mean([x['success'] for x in valid]) if valid else 0.0
        print(f"[{time.time() - t0:5.0f}s] ep {k:3d} {rand['spot']:24s} {rand['mode']:5s} "
              f"{'SKIP' if r['skipped'] else 'OK  ' if r['success'] else 'FAIL'} {str(r.get('reason', ''))[:70]}   "
              f"success so far {sr * 100:.0f}% ({len(valid)} valid)", flush=True)
    valid = [x for x in rows if not x['skipped']]
    ok = [x for x in valid if x['success']]
    summary = dict(policy='expert' if a.expert else client.model, url=None if a.expert else a.url, n=len(rows),
                   valid=len(valid), skipped=len(rows) - len(valid), success_rate=len(ok) / max(1, len(valid)),
                   by_spot=table(valid, lambda r: r['spot']), by_mode=table(valid, lambda r: r['mode']),
                   by_base_band=table(valid, lambda r: r['base_band']),
                   collisions=sum(1 for x in valid if x.get('collisions')),
                   base_used=sum(1 for x in valid if x.get('base_used', 0) > 0.05),
                   mean_t_success=round(float(np.mean([x['t_end'] for x in ok])), 2) if ok else None,
                   fail_reasons=dict(collections.Counter(str(x.get('reason', ''))[:40] for x in valid if not x['success'])),
                   seed0=a.seed0, timeout=a.timeout, task=a.task, episodes=rows)
    json.dump(summary, open(a.out, 'w'), indent=1, default=str)
    print(f"success {summary['success_rate'] * 100:.1f}% of {summary['valid']} valid ({summary['skipped']} skipped at the "
          f"start checks); collisions {summary['collisions']}, base used {summary['base_used']}, mean time "
          f"{summary['mean_t_success']} s -> {a.out}")
    print('by spot', summary['by_spot']); print('by mode', summary['by_mode'], 'base band', summary['by_base_band'])


if __name__ == '__main__':
    main()
