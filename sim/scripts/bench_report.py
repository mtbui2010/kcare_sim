#!/usr/bin/env python3
"""Summarise skill_bench runs: success / failure modes per skill, reach vs. success, and where the time goes.

    python3 scripts/bench_report.py out/bench/base055 [out/bench/near042 ...] [--log out/house/kcare.log]

Time breakdown: kcare logs every ROS call as `<YYYYmmddHHMMSSffffff>@<agent>..<conn>@sent|returned`; for each skill
window the busy intervals are grouped by device (arm, lift, head, base, gripper, vision 3D, other) and the rest of
the window is "no device busy" (detector requests, sleeps, planning, Python).
"""
import argparse, collections, datetime as dt, json, math, re, sys

GROUP = [('arm', r'arm_'), ('lift', r'^lift'), ('head', r'head_'), ('base', r'mobile_'), ('gripper', r'grip'),
         ('3d', r'get3d'), ('sim', r'sim_command')]


def load(path):
    return [json.loads(l) for l in open(f'{path}/trials.jsonl')]


def calls(logpath):
    """[(t_sent, t_returned, agent)] from kcare.log."""
    pat = re.compile(r'^(\d{20})@([^.@]+)\.\.([^@]*)@(sent|retunred|returned) saved')
    open_, out = collections.defaultdict(list), []
    for line in open(logpath, errors='replace'):
        m = pat.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), '%Y%m%d%H%M%S%f').timestamp()
        key = m.group(2)
        if m.group(4) == 'sent':
            open_[key].append(t)
        elif open_[key]:
            t0 = open_[key].pop()                               # the latest unanswered send of this agent
            if 0 <= t - t0 < 40:                                # older sends never got a 'returned' line (wait=False)
                out.append((t0, t, key))
            open_[key].clear()
    return out


def group_of(agent):
    for g, p in GROUP:
        if re.search(p, agent):
            return g
    return 'other'


def union(iv):
    tot, end = 0.0, -1e18
    for a, b in sorted(iv):
        if b <= end:
            continue
        tot += b - max(a, end); end = b
    return tot


def breakdown(cl, t0, t1):
    by = collections.defaultdict(list)
    for a, b, k in cl:
        if b > t0 and a < t1:
            by[group_of(k)].append((max(a, t0), min(b, t1)))
    res = {g: union(v) for g, v in by.items() if g != 'sim'}
    res['no device busy'] = (t1 - t0) - union([x for g, v in by.items() if g != 'sim' for x in v])
    return res


def pct(n, d):
    return f'{n}/{d} ({100 * n / d:.0f}%)' if d else '0/0'


def report(name, R, cl):
    print(f'\n## {name}: {len(R)} trials, standoff {R[0].get("standoff")}')
    R = [r for r in R if 'skipped' not in r]
    mv = [r['move'] for r in R]
    print(f"move   reported ok {pct(sum(m['isdone'] for m in mv), len(mv))}; nav err > 5 cm: {sum(m['nav_err'] > 0.05 for m in mv)}; "
          f"median {sorted(m['sec'] for m in mv)[len(mv) // 2]:.1f} s")
    fv = collections.Counter(r['find']['verdict'].split(':')[0] for r in R)
    errs = sorted(r['find']['err_xy'] for r in R if r['find'].get('verdict') == 'right')
    cams = collections.Counter(r['find'].get('cam') for r in R if r['find'].get('cam'))
    print(f"find   {dict(fv)}  cams {dict(cams)}; err (right) median {errs[len(errs) // 2] if errs else '-'} m, "
          f"p90 {errs[int(len(errs) * 0.9)] if errs else '-'} m; median {sorted(r['find']['sec'] for r in R)[len(R) // 2]:.1f} s")
    wrong = collections.Counter(f"{r['object']} -> {r['find']['verdict']}" for r in R if r['find']['verdict'] not in ('right',))
    if wrong:
        print('       misses:', '; '.join(f'{k} x{v}' for k, v in wrong.most_common(12)))
    pv = collections.Counter(r['pick']['verdict'] for r in R)
    print(f"pick   {dict(pv)}; median {sorted(r['pick']['sec'] for r in R)[len(R) // 2]:.1f} s")
    held = [r for r in R if r['pick']['held']]
    print(f"       true success {pct(len(held), len(R))}; reported ok {pct(sum(r['pick']['isdone'] for r in R), len(R))}; "
          f"false ok {sum(r['pick']['isdone'] and not r['pick']['held'] for r in R)}; disturbed others in {sum(bool(r['pick']['disturbed']) for r in R)}; "
          f"knocked to the floor {sum(bool(r['pick']['fell']) or r['pick']['object_now'] == 'floor' for r in R)}")
    # reach: lateral distance of the object from the base centre vs true pick success
    bins = [(0, 0.55), (0.55, 0.62), (0.62, 0.68), (0.68, 0.74), (0.74, 2)]
    line = []
    for lo, hi in bins:
        sub = [r for r in R if lo <= abs(r['obj_in_base'][1]) < hi]
        line.append(f"{lo:.2f}-{hi:.2f} m: {pct(sum(r['pick']['held'] for r in sub), len(sub))}")
    print('       held vs |y| (object to base centre, sideways):', ' | '.join(line))
    fails = collections.Counter(re.sub(r"[\\'\"{}]", '', r['pick']['msg'])[:110] for r in R if r['pick']['msg'])
    for k, v in fails.most_common(6):
        print(f'       x{v} {k}')
    pl = [r for r in R if 'place' in r]
    if pl:
        print(f"place  {dict(collections.Counter(r['place']['verdict'] for r in pl))} of {len(pl)} attempted; "
              f"tilt>30deg {sum(r['place']['tilt_vs_start'] > 30 for r in pl if r['place']['verdict'] == 'ok')}; "
              f"median {sorted(r['place']['sec'] for r in pl)[len(pl) // 2]:.1f} s")
        fails = collections.Counter(re.sub(r"[\\'\"{}]", '', r['place']['msg'])[:110] for r in pl if r['place']['msg'])
        for k, v in fails.most_common(5):
            print(f'       x{v} {k}')
    e2e = sum(r.get('place', {}).get('verdict') == 'ok' and r['pick']['held'] for r in R)
    print(f'end-to-end (object really on B): {pct(e2e, len(R))}')
    by_obj = collections.defaultdict(lambda: [0, 0])
    for r in R:
        by_obj[r['object']][0] += r['pick']['held']; by_obj[r['object']][1] += 1
    print('pick by object:', ', '.join(f'{k} {a}/{n}' for k, (a, n) in sorted(by_obj.items(), key=lambda kv: kv[1][0] / kv[1][1])))
    by_s = collections.defaultdict(lambda: [0, 0])
    for r in R:
        by_s[r['from']][0] += r['pick']['held']; by_s[r['from']][1] += 1
    print('pick by surface:', ', '.join(f'{k} {a}/{n}' for k, (a, n) in by_s.items()))
    if cl:
        for sk in ('move', 'find', 'pick', 'place'):
            agg = collections.defaultdict(float); n = 0
            for r in R:
                s = r.get(sk)
                if not s or 't0' not in s:
                    continue
                for g, v in breakdown(cl, s['t0'], s['t0'] + s['sec']).items():
                    agg[g] += v
                n += 1
            if n:
                tot = sum(agg.values())
                print(f'time {sk:5s} mean wall {sum(r[sk]["sec"] for r in R if sk in r) / n:5.1f} s, device busy (overlapping): ' + ', '.join(f'{g} {v / n:.1f}' for g, v in sorted(agg.items(), key=lambda kv: -kv[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--log', default='/sim/out/house/kcare.log')
    a = ap.parse_args()
    try:
        cl = calls(a.log)
    except OSError:
        cl = []
    for p in a.runs:
        report(p, load(p), cl)


if __name__ == '__main__':
    main()
