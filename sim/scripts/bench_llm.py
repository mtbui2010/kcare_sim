#!/usr/bin/env python3
"""Compare LLMs for every LLM role of the SAGE loop, offline (no robot motion):

    translate  ko -> en of the command (UnifiedAgent), en -> ko / vi of the robot's questions (SageLoop._in_lang)
    understand task -> goals + the clarification dialogue (scripted user), grounded on the live ENV
    plan       GRACE generate_plan on the restated task, then the SAGE gate: did it need repairs / a recompile?

    /sim/.kcare_venv/bin/python scripts/bench_llm.py --models qwen3-vl:8b-instruct qwen3-vl:32b-instruct --reps 2

ENV comes from a running kcare (KCARE_API). Writes out/bench_llm.jsonl and prints per-model scores + latency.
"""
import argparse, json, os, sys, time, urllib.request

os.environ.setdefault("OLLAMA_NUM_CTX", "8192")
from robot_agent.core.planning.assist_loop import Places, AssistLoop as SageLoop, Memory, words
from robot_agent.core.planning.base import WorldState
from robot_agent.connect.llm import init_llm_client

API = os.environ.get('KCARE_API', 'http://127.0.0.1:8003')
URL = os.environ.get('OLLAMA', 'http://127.0.0.1:11434')

G = lambda t, o, f, to: {'type': t, 'object': o, 'from': f, 'to': to}
# task, scripted answers (in order; a read-back gets "yes" unless scripted), expected outcome
CASES = {
    't01': ('Bring the apple from the desk to the dining table.', [], [G('on', 'apple', 'desk@study room', 'dining table@kitchen')]),
    't02': ('Put the banana on the kitchen counter.', [], [G('on', 'banana', '', 'counter@kitchen')]),
    't03': ('Move the mug from the desk to the chest in the bedroom.', [], [G('on', 'mug', 'desk@study room', 'chest@bedroom')]),
    't04': ('Take the tennis ball from the stool and put it on the desk.', [], [G('on', 'tennis ball', 'stool@bedroom', 'desk@study room')]),
    't05': ('Open the drawer in the kitchen.', [], [G('open', 'drawer@kitchen', '', 'drawer@kitchen')]),
    't06': ('Put the orange in the kitchen drawer.', [], [G('in', 'orange', '', 'drawer@kitchen')]),
    't07': ('Bring the sponge to the dining table and then put the bowl on the counter.', [],
            [G('on', 'sponge', '', 'dining table@kitchen'), G('on', 'bowl', '', 'counter@kitchen')]),
    't08': ('Bring me a cup.', ['the red cup on the chest, bring it to the dining table'],
            [G('on', 'red cup', 'chest@bedroom', 'dining table@kitchen')]),
    't09': ('Put the bleach bottle on the shelf.', ['the shelf in the restroom'], [G('on', 'bleach bottle', '', 'shelf@restroom')]),
    't10': ('Clean the table.', ['move the plate and the bowl from the dining table to the counter'],
            [G('on', 'plate', 'dining table@kitchen', 'counter@kitchen'), G('on', 'bowl', 'dining table@kitchen', 'counter@kitchen')]),
    't11': ('Throw the spray bottle out of the window.', [], 'refused'),
    't12': ('Pick up the cracker box from the kitchen shelf and give it to me at the sofa.', [],
            [G('on', 'cracker box', 'shelf@kitchen', 'sofa@living room')]),
    't13': ('책상 위에 있는 머그컵을 식탁으로 옮겨 줘.', [], [G('on', 'mug', 'desk@study room', 'dining table@kitchen')]),
    't14': ('Bring me the umbrella.', ['I am at the sofa'], [G('on', 'umbrella', '', 'sofa@living room')]),
    't15': ('Wash the dishes in the sink.', [], 'refused'),
    't16': ('Put the mug on the chest and the tennis ball on the desk.', [],
            [G('on', 'mug', '', 'chest@bedroom'), G('on', 'tennis ball', '', 'desk@study room')]),
    'F02': ('Put the bleach bottle on the shelf.', ['the shell in the rest room', 'the restroom'], [G('on', 'bleach bottle', '', 'shelf@restroom')]),
    'F03': ('Put the bleach bottle on the shelf.', ['the shelf in the kitchen', 'no, the shelf in the restroom'],
            [G('on', 'bleach bottle', '', 'shelf@restroom')]),
    'F15': ('Put the bleach bottle on the shelf.', ['화장실 선반'], [G('on', 'bleach bottle', '', 'shelf@restroom')]),
    'V01': ('Đặt quả táo lên bàn ăn.', [], [G('on', 'apple', '', 'dining table@kitchen')]),
}
QUESTIONS = ['Which shelf do you mean: shelf in the kitchen or shelf in the restroom?',
             'I will put the bleach bottle on the shelf in the restroom. Is that right?',
             "I couldn't find the mug. Where is it?"]


def chat(model, prompt, fmt=None):
    body = {'model': model, 'stream': False, 'think': False, 'keep_alive': '30m',
            'options': {'temperature': 0, 'num_ctx': 8192}, 'messages': [{'role': 'user', 'content': prompt}]}
    if fmt:
        body['format'] = fmt
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + '/api/chat', data=json.dumps(body).encode(),
                                                                 headers={'Content-Type': 'application/json'}), timeout=300).read())
    return r['message']['content'], time.time() - t


def goal_match(got, want):
    if want == 'refused':
        return got == 'refused'
    if not isinstance(got, list) or len(got) != len(want):
        return False
    for g, w in zip(sorted(got, key=lambda x: x['object']), sorted(want, key=lambda x: x['object'])):
        if g['type'] != w['type'] or (w['type'] in ('on', 'in') and w['object'] not in g['object'] and g['object'] not in w['object']):
            return False
        if g.get('to') != w['to']:
            return False
        if w['from'] and g.get('from') not in (w['from'], ''):       # a missing source is fine (search)
            return False
    return True


def run_case(sl, places, task, answers):
    """SageLoop's clarification logic, offline. Returns (outcome, n_llm, llm_sec, dialogue)."""
    t_llm, n = 0.0, 0
    original = task
    if any(ord(c) > 0x2000 for c in task) or any(c in task for c in 'ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹđ'):
        t0 = time.time()
        task = str(sl._llm().chat(prompt=f'Translate to daily english. Output result only: {task}')).strip()
        t_llm += time.time() - t0; n += 1
    sl.original = original
    mem = Memory('/tmp/bench_llm_mem.json'); mem.data = {'objects': {}, 'answers': []}
    dialogue, answers, pin_to, confirmed = [], list(answers), None, 0
    for _round in range(6):
        t0 = time.time()
        u = sl._understand(task, places, mem, WorldState(), dialogue)
        t_llm += time.time() - t0; n += 1
        if pin_to:
            for g in u['goals']:
                if g.get('type') in ('on', 'in'):
                    g['to'] = pin_to
        ans_ = [t for w, t in dialogue if w == 'user']
        goals, q = sl._ground_goals(u['goals'], places, mem, ' '.join([task, original] + ans_), latest=ans_[-1] if ans_ else '')
        if u['status'] == 'infeasible' and dialogue and goals:
            u['status'] = 'clear'
        if u['status'] in ('unsafe', 'infeasible'):
            return 'refused', n, t_llm, dialogue
        if u['status'] == 'ambiguous' and u.get('question') and not goals:
            q = u['question'] if not q else f"{u['question']} {q}"
        if not goals and not q:
            q = 'What should I do exactly?'
        if not q:
            if not dialogue or confirmed >= 2:
                return goals, n, t_llm, dialogue
            confirmed += 1
            cq = sl.summary(goals) + ' Is that right?'
            a = answers.pop(0) if answers and answers[0].lower().startswith('no') else 'yes'
            ok, rest = sl.yes_no(a)
            if ok:
                return goals, n, t_llm, dialogue
            dialogue += [('robot', cq), ('user', a)]
            k = places.ground(rest, rest)
            if k:
                pin_to = k
            continue
        if not answers:
            return ('asked: ' + q), n, t_llm, dialogue
        dialogue += [('robot', q), ('user', answers.pop(0))]
    return 'no decision', n, t_llm, dialogue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['qwen3-vl:8b-instruct', 'qwen3-vl:32b-instruct'])
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--no-plan', action='store_true')
    ap.add_argument('--out', default='/sim/out/bench_llm.jsonl')
    a = ap.parse_args()
    env = json.load(urllib.request.urlopen(API + '/skill-configs/ENV'))
    places = Places(env)
    out = open(a.out, 'a')
    for model in a.models:
        chat(model, 'hello')                                         # load it (not timed)
        sl = SageLoop.__new__(SageLoop)
        sl._llm_cfg = {'name': 'llama', 'url': URL, 'model': model, 'num_ctx': 8192}
        sl.original = None
        planner = None
        if not a.no_plan:
            from robot_agent.core.planning.registry import get_planner
            planner = get_planner('grace', host=URL, model=model, live_path=f'/tmp/bench_grace_{model.replace(":", "_")}.jsonl',
                                  safe_refine=True)
        for rep in range(a.reps):
            # translation of the robot's questions
            for qq in QUESTIONS:
                for lang in ('Korean', 'Vietnamese'):
                    txt, sec = chat(model, f'Translate into natural spoken {lang}. Keep it short and polite. Output only the translation:\n{qq}')
                    rec = dict(model=model, rep=rep, role='translate', lang=lang, src=qq, out=txt.strip(), sec=round(sec, 2))
                    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            for cid, (task, answers, want) in CASES.items():
                got, n, sec, dlg = run_case(sl, places, task, answers)
                ok = goal_match(got, want)
                rec = dict(model=model, rep=rep, role='understand', case=cid, ok=ok, got=got, n_llm=n, sec=round(sec, 2),
                           dialogue=dlg)
                print(f"{model:24} r{rep} {cid} {'OK ' if ok else 'BAD'} {sec:5.1f}s n={n} {json.dumps(got, ensure_ascii=False)[:150]}", flush=True)
                out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()
                if planner is not None and isinstance(want, list) and rep == 0:
                    restated = task + ('' if not dlg else ' (' + '; '.join(t for w, t in dlg if w == 'user') + ')')
                    goals = want
                    obs = ('Places the robot can go to (use exactly these):\n' + '\n'.join(f'- {k}' for k in places.keys)
                           + '\nGoals: ' + '; '.join(f"{g['type']} {g['object']}" + (f" from {g['from']}" if g['from'] else '')
                                                     + (f" to {g['to']}" if g['to'] else '') for g in goals) + '\nRobot state: arrived=-')
                    t0 = time.time()
                    try:
                        raw, meta = planner.generate_plan(restated, obs, [])
                    except Exception as e:
                        raw, meta = [], {'error': str(e)}
                    psec = time.time() - t0
                    raw_g = [sl._ground_step(s, places, goals, task) for s in raw]
                    import importlib
                    sl._namemap = importlib.import_module('kcare_robot.configs.grace_namemap')
                    probs = sl._check(raw_g, WorldState(), places, goals); steps = raw_g
                    fixes = [p_[1] for p_ in probs]; compiled = any(p_[2] for p_ in probs)
                    rec = dict(model=model, role='plan', case=cid, sec=round(psec, 2), n_raw=len(raw), n_repairs=len(fixes),
                               compiled=compiled, raw=[f"{s.get('action')} {s.get('object')}" for s in raw_g],
                               final=[f"{s['action']} {s['object']}" for s in steps], error=meta.get('error') if isinstance(meta, dict) else None)
                    print(f"{model:24} plan {cid} {psec:5.1f}s raw {len(raw)} repairs {len(fixes)} {'COMPILED' if compiled else ''}", flush=True)
                    out.write(json.dumps(rec, ensure_ascii=False) + '\n'); out.flush()


if __name__ == '__main__':
    main()
