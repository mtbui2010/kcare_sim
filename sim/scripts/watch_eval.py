#!/usr/bin/env python3
"""Evaluate every checkpoint of a training run as soon as it is saved.

    CUDA_VISIBLE_DEVICES=0 nohup /sim/.venv/bin/python scripts/watch_eval.py \
        --run /nas/checkpoints/smolvla_open_drawer_500_kaairbase_20260924 > /sim/out/watch_eval.log 2>&1 &

Polls <run>/checkpoints/<step>/ every minute; a checkpoint counts as complete
once its pretrained_model/ has model.safetensors and nothing in it changed for
`--settle` s (lerobot writes the folder in several steps). Each new one is
evaluated with eval_policy.py (same 50 unseen scenes as every other run), and
<run>/eval/summary.json + summary.md are rewritten after each. Stops after the
final step (`steps` in a checkpoint's train_config.json) has been evaluated.
Also fills in missing steps of a finished run (e.g. 10k/15k of an older run).
"""
import argparse, glob, json, os, subprocess, sys, time

SIM = os.environ.get('SIM_ROOT', '/sim')


def complete(step_dir, settle):
    pm = os.path.join(step_dir, 'pretrained_model')
    files = glob.glob(os.path.join(pm, '*'))
    if not os.path.exists(os.path.join(pm, 'model.safetensors')):
        return False
    return time.time() - max(os.path.getmtime(f) for f in files) > settle


def result_name(step, tag):
    return f'{step}_{tag}_results.json' if tag else f'{step}_results.json'


def write_summary(run, tag=''):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, 'eval', '*_results.json'))):
        step = os.path.basename(f).split('_')[0]
        if not step.isdigit() or os.path.basename(f) != result_name(step, tag):
            continue
        r = json.load(open(f))
        ts = [e['t_success'] for e in r['episodes'] if e['success']]
        rows.append(dict(step=int(step), success_rate=r['success_rate'], n=r['n'],
                         mean_opened_cm=round(r['mean_opened_m'] * 100, 1),
                         mean_t_success_s=round(sum(ts) / len(ts), 2) if ts else None))
    sfx = f'_{tag}' if tag else ''
    json.dump(rows, open(os.path.join(run, 'eval', f'summary{sfx}.json'), 'w'), indent=2)
    with open(os.path.join(run, 'eval', f'summary{sfx}.md'), 'w') as fh:
        fh.write('| step | success | mean opened [cm] | mean time to success [s] |\n|---|---|---|---|\n')
        for r in rows:
            fh.write(f"| {r['step']} | {r['success_rate'] * 100:.0f}% ({round(r['success_rate'] * r['n'])}/{r['n']}) "
                     f"| {r['mean_opened_cm']} | {r['mean_t_success_s']} |\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--settle', type=float, default=120)
    ap.add_argument('--poll', type=float, default=60)
    ap.add_argument('--tag', default='', help='eval set name (see eval_policy.py --tag)')
    ap.add_argument('--eval-args', default='', help='extra eval_policy.py args, e.g. "--task task_open_drawer.yaml '
                    '--rand-config randomization_real.yaml --camera-model"')
    a = ap.parse_args()
    print(f'watching {a.run}', flush=True)
    final = None
    while True:
        if final is None:   # train_config.json is saved inside each checkpoint, not at the run root
            cfgs = glob.glob(os.path.join(a.run, 'checkpoints', '*', 'pretrained_model', 'train_config.json'))
            if cfgs:
                final = json.load(open(cfgs[0]))['steps']
                print(f'final step {final}', flush=True)
        if not os.path.isdir(os.path.join(a.run, 'checkpoints')):   # nothing saved yet
            time.sleep(a.poll)
            continue
        done = {os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(a.run, 'eval', '*_results.json'))
                if os.path.basename(f) == result_name(os.path.basename(f).split('_')[0], a.tag)}
        steps = sorted(d for d in os.listdir(os.path.join(a.run, 'checkpoints')) if d.isdigit())
        todo = [s for s in steps if s not in done and complete(os.path.join(a.run, 'checkpoints', s), a.settle)]
        for s in todo:
            print(f'[{time.strftime("%H:%M")}] evaluating step {s}', flush=True)
            subprocess.run([sys.executable, '-W', 'ignore', os.path.join(SIM, 'scripts', 'eval_policy.py'),
                            '--ckpt', os.path.join(a.run, 'checkpoints', s, 'pretrained_model'),
                            '--n', str(a.n), '--videos', '2'] + (['--tag', a.tag] if a.tag else []) + a.eval_args.split(),
                           check=False)
            rows = write_summary(a.run, a.tag)
            print('  ' + ', '.join(f"{r['step']}: {r['success_rate'] * 100:.0f}%" for r in rows), flush=True)
            done.add(s)
        if final is not None and f'{final:06d}' in done:
            print('final checkpoint evaluated — done', flush=True)
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    main()
