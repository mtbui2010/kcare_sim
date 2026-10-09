#!/usr/bin/env python3
"""Offline: SAGE plan -> checker -> SAGE edit (AssistLoop's planning stage) for the task-runner goals.

    /sim/.kcare_venv/bin/python scripts/bench_plan.py [--rounds 2] [--model qwen3-vl:8b-instruct]
"""
import argparse, importlib, json, os, sys, time, urllib.request
os.environ.setdefault("OLLAMA_NUM_CTX", "8192")
sys.path.insert(0, os.path.dirname(__file__))
from robot_agent.core.planning.assist_loop import Places, AssistLoop
from robot_agent.core.planning.base import WorldState
from robot_agent.core.planning.registry import get_planner
from bench_llm import CASES

ap = argparse.ArgumentParser(); ap.add_argument('--rounds', type=int, default=2); ap.add_argument('--model', default='qwen3-vl:8b-instruct')
ap.add_argument('--reps', type=int, default=1)
a = ap.parse_args()
env = json.load(urllib.request.urlopen(os.environ.get('KCARE_API', 'http://127.0.0.1:8003') + '/skill-configs/ENV')); P = Places(env)
S = AssistLoop.__new__(AssistLoop); S._namemap = importlib.import_module('kcare_robot.configs.grace_namemap')
tot = dict(ok0=0, ok=0, n=0, sec=0.0, steps=0)
for rep in range(a.reps):
    planner = get_planner('grace', host='http://127.0.0.1:11434', model=a.model, live_path=f'/tmp/bench_plan_{rep}.jsonl', safe_refine=True)
    for cid, (task, answers, goals) in CASES.items():
        if not isinstance(goals, list) or cid.startswith(('F', 'V')) or cid == 't13':
            continue
        restated = task + (' (' + '; '.join(answers) + ')' if answers else '')
        gt = '; '.join(f"{g['type']} {g['object']}" + (f" from {g['from']}" if g['from'] else '') + (f" to {g['to']}" if g['to'] else '') for g in goals)
        obs = 'Places the robot can go to (use exactly these):\n' + '\n'.join(f'- {k}' for k in P.keys) + f'\nGoals: {gt}\nRobot state: arrived=-'
        ground = lambda seq: [S._ground_step(x, P, goals, task) for x in seq]
        t0 = time.time()
        steps = ground(planner.generate_plan(restated, obs, [])[0])
        crit0 = [p for p in S._check(steps, WorldState(), P, goals) if p[2]]
        if crit0:
            S.cfg = {'edit_rounds': a.rounds}
            ed, notes = S._sage_edit(planner, restated, obs, ground, WorldState(), P, goals, gt)
            steps = ed if ed is not None else steps
        crit = [p for p in S._check(steps, WorldState(), P, goals) if p[2]]
        sec = time.time() - t0
        tot['n'] += 1; tot['ok0'] += not crit0; tot['ok'] += not crit; tot['sec'] += sec; tot['steps'] += len(steps)
        print(f"{cid} {sec:5.1f}s first {'OK ' if not crit0 else 'BAD'} final {'OK ' if not crit else 'BAD'} | {' > '.join(x['action'] + ' ' + x['object'] for x in steps)}"
              + ('' if not crit else f"\n      left: {[p[1] for p in crit]}"), flush=True)
print(f"\n{a.model} rounds={a.rounds}: valid first plan {tot['ok0']}/{tot['n']}, after edits {tot['ok']}/{tot['n']}, {tot['sec'] / tot['n']:.1f} s/task, {tot['steps'] / tot['n']:.1f} steps")
