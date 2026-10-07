#!/usr/bin/env python3
"""Generate a LeRobot "open drawer" dataset from the scripted expert.

Run with the LeRobot venv (lerobot 0.4.4 lives there, not in the ROS python):

    /sim/.venv/bin/python scripts/gen_dataset.py --n 50 --workers 8
      -> $KAAIR_DATA/kaair_sim_open_drawer          (LeRobot v3 dataset; NAS, /nas/datasets)
         $KAAIR_DATA/kaair_sim_open_drawer_episodes.jsonl  (per-episode randomization + result)
         /sim/out/dataset_preview/                   (preview mp4s + first-frame mosaic)

Schema = keti_assist_ai_robot_ros2/scripts/data_collect_kcare.py build_features():
  action, observation.state : float32[8]  joint1..7 [rad] + gripper_width [m]
  observation.images.head / .wrist : video, both 848x480 — the real recorder
      resizes every camera to its img_shape (INTER_AREA); head is rendered at
      the Femto's 1280x720 and downsized the same way
  fps 30, task = the episode's prompt ("open the top drawer", ... see drawer_targets in the
      rand config; plain "open drawer" = middle drawer)

Workers render (EGL) round-robin on --gpu-ids, each writes its own shard, and only
successful episodes are saved; the shards are aggregated at the end.
"""
import argparse, json, math, os, shutil, sys, time
import multiprocessing as mp

SIM = os.environ.get('SIM_ROOT', '/sim')
DATA = os.environ.get('KAAIR_DATA', '/nas/datasets')     # NAS — keep the A6000 disk free
IMG_WH = (848, 480)
TASK = 'open drawer'          # default prompt; with drawer_targets in the rand config each episode has its own
NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'gripper_width']


def features():
    w, h = IMG_WH
    f = {'action': {'dtype': 'float32', 'shape': (8,), 'names': NAMES},
         'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': NAMES}}
    for cam in ('head', 'wrist'):
        f[f'observation.images.{cam}'] = {'dtype': 'video', 'shape': (h, w, 3), 'names': ['height', 'width', 'channels']}
    return f


def worker(wid, n_target, max_attempts, seed, shard_root, gpu_ids, preview_n, rand_cfg, task_cfg, use_cam_model, q):
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu_ids[wid % len(gpu_ids)])
    sys.path.insert(0, os.path.join(SIM, 'scripts'))
    import cv2, imageio, mujoco, numpy as np, yaml
    import check_sim as cs, drawer_ready as dr, expert_open_drawer as ex, randomize
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(task_cfg))
    cams = None
    if use_cam_model:                                   # real camera look (camera_model.py)
        import camera_model
        cams = camera_model.load_camera_models(profile=task.get('camera_profile'))   # e.g. 'home' colours
    rcfg = yaml.safe_load(open(rand_cfg))
    mode = task['mode']
    ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    tcp_T = ref.fk(dr.ready_q(task, mode, cfg['home']))['tool_tcp_link']

    ds = LeRobotDataset.create(repo_id=f'kaair_sim_shard{wid}', fps=ex.HZ, features=features(), root=shard_root,
                               robot_type='kaair_xarm7', use_videos=True, image_writer_threads=4)
    scene_path = f'{SIM}/mjcf/_gen_scene_w{wid}.xml'      # next to kaair_slamtec.xml for <include>
    saved = attempts = 0
    while saved < n_target and attempts < max_attempts:
        ep_seed = seed * 1_000_003 + wid * 10_007 + attempts
        attempts += 1
        rng = np.random.default_rng(ep_seed)
        rand = randomize.sample(rng, rcfg)
        if cams is not None:
            for c in cams.values():
                c.episode_jitter(rng)                   # this episode's AE/AWB/JPEG state
        xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
        open(scene_path, 'w').write(xml)
        m = mujoco.MjModel.from_xml_path(scene_path)
        d = mujoco.MjData(m)
        r_head = mujoco.Renderer(m, 720, 1280)
        r_wrist = mujoco.Renderer(m, IMG_WH[1], IMG_WH[0])
        preview = [] if saved < preview_n else None

        def on_step(m, d, phase, drawer, state, action):
            r_head.update_scene(d, 'head_rgb')
            head = r_head.render()
            r_wrist.update_scene(d, 'wrist_rgb')
            wrist = r_wrist.render().copy()
            if cams is not None:
                head = cams['head'].apply(head, rng, cam_T=camera_model.cam_pose(m, d, 'head_rgb'), t=d.time)
                wrist = cams['wrist'].apply(wrist, rng, cam_T=camera_model.cam_pose(m, d, 'wrist_rgb'), t=d.time)
            head = cv2.resize(head, IMG_WH, interpolation=cv2.INTER_AREA)
            ds.add_frame({'observation.state': np.asarray(state, np.float32),
                          'action': np.asarray(action, np.float32),
                          'observation.images.head': head, 'observation.images.wrist': wrist,
                          'task': rand.get('prompt', TASK)})
            if preview is not None:
                preview.append(np.concatenate([cv2.resize(head, (424, 240)), cv2.resize(wrist, (424, 240))], 1))

        t0 = time.time()
        rec = ex.run_episode(m, d, task, cfg, mode, render=on_step, log=lambda *a: None,
                             q_extra=rand['q_extra'], drawer0=rand['drawer0'], target=rand.get('target'))
        r_head.close(); r_wrist.close()
        opened = float(rec['drawer'][-1] - rand['drawer0'])
        if rec['success']:
            ds.save_episode()
            saved += 1
            if preview is not None:
                imageio.mimwrite(f'{SIM}/out/dataset_preview/w{wid}_ep{saved - 1}.mp4', preview, fps=ex.HZ,
                                 quality=6, macro_block_size=1)
        else:
            ds.clear_episode_buffer()
        q.put(dict(worker=wid, seed=int(ep_seed), success=bool(rec['success']), opened=opened,
                   frames=int(len(rec['action'])), sec=round(time.time() - t0, 1),
                   rand={k: (list(v) if isinstance(v, tuple) else v) for k, v in rand.items()}))
    ds.finalize()
    os.remove(scene_path)
    q.put(dict(worker=wid, done=True, saved=saved, attempts=attempts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50, help='successful episodes to keep')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gpu-ids', default='0', help='comma list of GPUs to render on (only free ones!)')
    ap.add_argument('--rand-config', default=f'{SIM}/config/randomization.yaml')
    ap.add_argument('--task', default=f'{SIM}/config/task_open_drawer.yaml')
    ap.add_argument('--camera-model', action='store_true', help='pass renders through camera_model.py (real look)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repo-id', default='kaair_sim_open_drawer')
    ap.add_argument('--preview', type=int, default=1, help='preview mp4s per worker')
    a = ap.parse_args()

    out_root = f'{DATA}/{a.repo_id}'
    shard_dir = f'{DATA}/_shards/{a.repo_id}'
    for p in (out_root, shard_dir):
        if os.path.exists(p):
            raise SystemExit(f'{p} exists — remove it or pick another --repo-id')
    os.makedirs(shard_dir)
    os.makedirs(f'{SIM}/out/dataset_preview', exist_ok=True)

    per = [a.n // a.workers + (1 if i < a.n % a.workers else 0) for i in range(a.workers)]
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    gpu_ids = [int(g) for g in a.gpu_ids.split(',')]
    procs = [ctx.Process(target=worker, args=(i, per[i], 3 * per[i] + 2, a.seed, f'{shard_dir}/w{i}', gpu_ids,
                                              a.preview, a.rand_config, a.task, a.camera_model, q)) for i in range(a.workers) if per[i] > 0]
    t0 = time.time()
    for p in procs:
        p.start()
    log = open(f'{DATA}/{a.repo_id}_episodes.jsonl', 'w')
    done, n_ok, n_all = 0, 0, 0
    while done < len(procs):
        msg = q.get()
        if msg.get('done'):
            done += 1
            continue
        n_all += 1; n_ok += msg['success']
        log.write(json.dumps(msg) + '\n'); log.flush()
        print(f"[{time.time() - t0:6.0f}s] w{msg['worker']} {'OK  ' if msg['success'] else 'FAIL'} "
              f"opened {msg['opened'] * 100:4.1f} cm  ({msg['sec']} s)   kept {n_ok}/{a.n}, tried {n_all}", flush=True)
    for p in procs:
        p.join()

    from lerobot.datasets.aggregate import aggregate_datasets
    shards = sorted(os.listdir(shard_dir))
    aggregate_datasets(repo_ids=[f'kaair_sim_shard{s[1:]}' for s in shards], aggr_repo_id=a.repo_id,
                       roots=[f'{shard_dir}/{s}' for s in shards], aggr_root=out_root)
    shutil.rmtree(shard_dir)
    print(f'done in {time.time() - t0:.0f}s: {n_ok} episodes kept of {n_all} tried ({100 * n_ok / max(n_all, 1):.0f}% success) '
          f'-> {out_root}')


if __name__ == '__main__':
    main()
