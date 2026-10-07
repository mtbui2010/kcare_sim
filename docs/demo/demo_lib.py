"""Helpers for kaair_demo.ipynb. The notebook runs on the A6000 host (kernel "kaair demo (host)");
every simulation / ML command runs inside the kaair_sim_humble container through `docker exec`.

    sh('python3 scripts/demo.py expert --drawer top')     # run in the container, output streamed live
    show_video('/sim/out/demo/expert_top_seed7.mp4')      # container path -> shown inline
"""
import base64, json, os, re, subprocess

from IPython.display import HTML, Image, Markdown, display

CONTAINER = 'kaair_sim_humble'
HOME = os.path.expanduser('~/trung_workdir/kaair_sim')
# container path prefix -> host path prefix (the notebook reads result files directly)
MOUNTS = {'/sim': f'{HOME}/sim', '/nas': '/mnt/nas/lerobot_shared/kaair_sim'}
OUR_GPUS = (0, 3)                       # GPUs 1 and 2 are usually another user's jobs
RUN = '/nas/checkpoints/smolvla_open_drawer_real500_multidrawer_20260928'


def host(path):
    for c, h in MOUNTS.items():
        if path.startswith(c + '/') or path == c:
            return h + path[len(c):]
    return path


def sh(cmd, keep=None, quiet=False, ret=False):
    """Run `cmd` in the container (cwd /sim). keep: regex; only matching lines are printed.
    ret=True returns the full output (str)."""
    full = f'source /workspace/.entrypoint.sh >/dev/null 2>&1; cd /sim; export MUJOCO_GL=egl PYTHONWARNINGS=ignore; {cmd}'
    p = subprocess.Popen(['docker', 'exec', CONTAINER, 'bash', '-c', full],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    out, buf = [], ''
    for ch in iter(lambda: p.stdout.read(1), ''):
        if ch not in '\r\n':
            buf += ch
            continue
        if buf:
            out.append(buf)
            if not quiet and (keep is None or re.search(keep, buf)) and not re.search(r'UserWarning|warnings.warn', buf):
                print(buf, flush=True)
        buf = ''
    p.wait()
    if p.returncode:
        print(f'[exit code {p.returncode}]')
    return '\n'.join(out) if ret else None


def gpu_status():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return [tuple(int(x) for x in line.split(',')) for line in q.strip().splitlines()]


def pick_gpu():
    """Our GPU with the most free memory."""
    st = {g[0]: g for g in gpu_status()}
    return max(OUR_GPUS, key=lambda i: st[i][2] - st[i][1])


def status():
    ok = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout.split()
    rows = ['| | |', '|---|---|',
            f"| container `{CONTAINER}` | {'running' if CONTAINER in ok else '**NOT running** (cd sim && make up)'} |"]
    for i, used, tot, util in gpu_status():
        tag = 'ours' if i in OUR_GPUS else 'other user'
        rows.append(f'| GPU {i} ({tag}) | {used / 1024:.1f} / {tot / 1024:.0f} GB, {util}% busy |')
    for name, p in (('NAS datasets', '/nas/datasets'), ('NAS checkpoints', '/nas/checkpoints')):
        rows.append(f"| {name} | {'ok' if os.path.isdir(host(p)) else 'missing'} (`{host(p)}`) |")
    ck = sorted(os.listdir(host(RUN + '/checkpoints'))) if os.path.isdir(host(RUN + '/checkpoints')) else []
    rows.append(f"| trained model | {', '.join(c for c in ck if c != 'last') or 'none yet'} |")
    display(Markdown('\n'.join(rows)))


def show_img(path, width=900):
    display(Image(filename=host(path), width=width))


def show_video(path, width=848):
    data = base64.b64encode(open(host(path), 'rb').read()).decode()
    display(HTML(f'<video width="{width}" controls autoplay muted loop '
                 f'src="data:video/mp4;base64,{data}"></video>'))


def latest_ckpt(run=RUN):
    return f'{run}/checkpoints/last/pretrained_model'


def plot_loss(log_text, title='training loss'):
    import matplotlib.pyplot as plt
    pts = [(int(float(s.rstrip('K')) * (1000 if s.endswith('K') else 1)), float(l))
           for s, l in re.findall(r'step:([\d.]+K?)\s.*?loss:([\d.]+)', log_text)]
    if not pts:
        print('no loss lines found'); return
    x, y = zip(*pts)
    plt.figure(figsize=(7, 3)); plt.plot(x, y, 'o-'); plt.yscale('log')
    plt.xlabel('training step'); plt.ylabel('loss (log scale)'); plt.title(title); plt.grid(alpha=0.3)
    plt.show()


def eval_table(run=RUN, tag='real'):
    """Success per checkpoint and per drawer, from <run>/eval/<step>_<tag>_results.json."""
    import pandas as pd
    d = host(run + '/eval')
    names = {0: 'bottom', 1: 'middle', 2: 'top'}
    rows = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(f'_{tag}_results.json'):
            continue
        r = json.load(open(f'{d}/{f}'))
        row = {'step': int(f.split('_')[0]), 'success': f"{r['success_rate'] * 100:.0f}% ({r['n']} scenes)"}
        for t, nm in names.items():
            ep = [e for e in r['episodes'] if e.get('target') == t]
            row[nm] = f"{sum(e['success'] for e in ep)}/{len(ep)}" if ep else '-'
        row['wrong drawer'] = sum(e.get('wrong_drawer_m', 0) >= 0.03 for e in r['episodes'])
        rows.append(row)
    if not rows:
        print('no evaluation finished yet'); return None
    return pd.DataFrame(rows).set_index('step')
