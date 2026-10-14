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
import argparse, json, math, os, queue, shutil, sys, time
import multiprocessing as mp

SIM = os.environ.get('SIM_ROOT', '/sim')
DATA = os.environ.get('KAAIR_DATA', '/nas/datasets')     # NAS — keep the A6000 disk free
IMG_WH = (848, 480)
TASK = 'open drawer'          # default prompt; with drawer_targets in the rand config each episode has its own
NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'gripper_width']


def features(ex=None):
    """The drawer schema (8-D state / action) unless the expert module declares its own STATE_NAMES / ACTION_NAMES
    (expert_pick_approach: 17-D state, 10-D action)."""
    w, h = IMG_WH
    sn, an = getattr(ex, 'STATE_NAMES', NAMES), getattr(ex, 'ACTION_NAMES', NAMES)
    f = {'action': {'dtype': 'float32', 'shape': (len(an),), 'names': an},
         'observation.state': {'dtype': 'float32', 'shape': (len(sn),), 'names': sn}}
    for cam in ('head', 'wrist'):
        f[f'observation.images.{cam}'] = {'dtype': 'video', 'shape': (h, w, 3), 'names': ['height', 'width', 'channels']}
    return f


def worker(wid, n_target, max_attempts, seed, shard_root, gpu_ids, preview_n, rand_cfg, task_cfg, use_cam_model, q):
    os.environ['MUJOCO_GL'] = 'egl'
    for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):   # many workers side by side
        os.environ.setdefault(v, '2')
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu_ids[wid % len(gpu_ids)])
    sys.path.insert(0, os.path.join(SIM, 'scripts'))
    import cv2, imageio, mujoco, numpy as np, yaml
    cv2.setNumThreads(2)
    import importlib
    import check_sim as cs, drawer_ready as dr, randomize
    task0 = yaml.safe_load(open(task_cfg))
    ex = importlib.import_module(f"expert_{task0.get('expert', 'open_drawer')}")   # task['expert'] picks the expert
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
    task = yaml.safe_load(open(task_cfg))
    cams = None
    if use_cam_model:                                   # real camera look (camera_model.py)
        import camera_model
        cams = camera_model.load_camera_models(profile=task.get('camera_profile'))   # e.g. 'home' colours
    # an expert with its own scene / randomization (build_scene + sample_rand, e.g. expert_pick_approach); the drawer
    # experts keep the path below unchanged
    custom = hasattr(ex, 'build_scene')
    if custom and rand_cfg == f'{SIM}/config/randomization.yaml' and task.get('rand_config'):
        rand_cfg = task['rand_config']
    rcfg = yaml.safe_load(open(rand_cfg))
    if not custom:
        mode = task['mode']
        ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
        tcp_T = ref.fk(dr.ready_q(task, mode, cfg['home']))['tool_tcp_link']

    ds = LeRobotDataset.create(repo_id=f'kaair_sim_shard{wid}', fps=ex.HZ, features=features(ex if custom else None), root=shard_root,
                               robot_type='kaair_xarm7', use_videos=True, image_writer_threads=4)
    scene_path = f'{SIM}/mjcf/_gen_scene_{os.getpid()}_w{wid}.xml'   # next to kaair_slamtec.xml for <include>; per process
    saved = attempts = 0
    while saved < n_target and attempts < max_attempts:
        ep_seed = seed * 1_000_003 + wid * 10_007 + attempts
        attempts += 1
        rng = np.random.default_rng(ep_seed)
        rand = ex.sample_rand(rng, rcfg, task) if custom else randomize.sample(rng, rcfg)
        if custom:
            pass
        elif task.get('expert') == 'close_drawer':          # starts open; same drawer targets, "close ..." phrasings
            rand['drawer0'] = float(rng.uniform(*task.get('drawer0_range', (0.12, 0.25))))
            rand['prompt'] = rand.get('prompt', 'open drawer').replace('open', 'close')
        elif task.get('retry_prob'):                       # open drawer: some episodes miss the bar first and retry
            rand['retry'] = bool(rng.random() < float(task['retry_prob']))
        if cams is not None:
            for c in cams.values():
                c.episode_jitter(rng)                   # this episode's AE/AWB/JPEG state
        if hasattr(ex, 'episode_model'):                  # cached model, objects placed per episode (pick in the house)
            m, d, info = ex.episode_model(task, rand, wid)
        else:
            if custom:
                xml, info = ex.build_scene(task, rand)
            else:
                xml, *_ = dr.build_scene(task, tcp_T, rand=rand)
            open(scene_path, 'w').write(xml)
            m = mujoco.MjModel.from_xml_path(scene_path)
            os.remove(scene_path)
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
        if custom:
            rec = ex.run_episode(m, d, task, cfg, rand=rand, info=info, render=on_step, log=lambda *a: None)
        else:
            kw = {'retry': rand['retry']} if 'retry' in rand else {}
            rec = ex.run_episode(m, d, task, cfg, mode, render=on_step, log=lambda *a: None,
                                 q_extra=rand['q_extra'], drawer0=rand['drawer0'], target=rand.get('target'), **kw)
        r_head.close(); r_wrist.close()
        opened = float(rec['metric']) if custom else float(rec['drawer'][-1] - rand['drawer0'])
        if rec['success']:
            ds.save_episode()
            saved += 1
            if preview is not None:
                imageio.mimwrite(f'{SIM}/out/dataset_preview/w{wid}_ep{saved - 1}.mp4', preview, fps=ex.HZ,
                                 quality=6, macro_block_size=1)
        else:
            ds.clear_episode_buffer()
        q.put(dict(worker=wid, seed=int(ep_seed), success=bool(rec['success']), opened=opened,
                   **({k_: rec.get(k_) for k_ in ('reason', 'tip_dist', 'plan', 'clearance', 'head_px', 'track_err')} if custom else {}),
                   frames=int(len(rec['action'])), sec=round(time.time() - t0, 1),
                   rand={k: (list(v) if isinstance(v, tuple) else v) for k, v in rand.items()}))
    ds.finalize()
    if os.path.exists(scene_path):
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
    ap.add_argument('--attempts-factor', type=int, default=3, help='per-worker attempt cap = factor x episodes + 2')
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
    procs = [ctx.Process(target=worker, args=(i, per[i], a.attempts_factor * per[i] + 2, a.seed, f'{shard_dir}/w{i}', gpu_ids,
                                              a.preview, a.rand_config, a.task, a.camera_model, q)) for i in range(a.workers) if per[i] > 0]
    t0 = time.time()
    for p in procs:
        p.start()
    log = open(f'{DATA}/{a.repo_id}_episodes.jsonl', 'w')
    done, gone, n = set(), set(), dict(ok=0, all=0)
    wids = [i for i in range(a.workers) if per[i] > 0]

    def handle(msg):
        if msg.get('done'):
            done.add(msg['worker'])
            return
        n['all'] += 1; n['ok'] += msg['success']
        log.write(json.dumps(msg) + '\n'); log.flush()
        print(f"[{time.time() - t0:6.0f}s] w{msg['worker']} {'OK  ' if msg['success'] else 'FAIL'} "
              f"opened {msg['opened'] * 100:4.1f} cm  ({msg['sec']} s)   kept {n['ok']}/{a.n}, tried {n['all']}", flush=True)

    while len(done | gone) < len(procs):
        try:
            handle(q.get(timeout=10))
        except queue.Empty:
            # a worker that died (crash, OOM, killed) never sends 'done': stop waiting for it and aggregate what the
            # others (and its own finished episodes) saved
            for wid, p in zip(wids, procs):
                if wid not in done and wid not in gone and not p.is_alive():
                    gone.add(wid)
                    print(f'worker w{wid} exited (code {p.exitcode}) without finishing — not waiting for it', flush=True)
    while True:                                             # whatever a dead worker queued before it died
        try:
            handle(q.get(timeout=1))
        except queue.Empty:
            break
    for p in procs:
        p.join(timeout=30)
    n_ok, n_all = n['ok'], n['all']

    from lerobot.datasets.aggregate import aggregate_datasets
    shards = sorted(os.listdir(shard_dir))

    def aggregate(names):
        aggregate_datasets(repo_ids=[f'kaair_sim_shard{s[1:]}' for s in names], aggr_repo_id=a.repo_id,
                           roots=[f'{shard_dir}/{s}' for s in names], aggr_root=out_root)
    try:
        aggregate(shards)
    except Exception as e:                                  # a dead worker's shard may be unreadable: without them
        ok = [s for s in shards if int(s[1:]) not in gone]
        if not gone or not ok:
            raise
        print(f'aggregating all shards failed ({e!r}); retrying without the dead workers {sorted(gone)}', flush=True)
        shutil.rmtree(out_root, ignore_errors=True)
        aggregate(ok)
    shutil.rmtree(shard_dir)
    print(f'done in {time.time() - t0:.0f}s: {n_ok} episodes kept of {n_all} tried ({100 * n_ok / max(n_all, 1):.0f}% success) '
          f'-> {out_root}')


if __name__ == '__main__':
    main()
