#!/usr/bin/env python3
"""Run the visually-impaired demo scenarios (the owner's scenario sheet) against kcare in the simulated house.

    source /workspace/ros2/ros2_ws/install/setup.bash; export ROS_DOMAIN_ID=79 SIM_ROOT=/sim/inst3
    KCARE_API=http://127.0.0.1:8006 python3 scripts/vi_demo.py [s1 s2 ...] --out out/vi_demo

Each scenario: load its scene (scripts/make_vi_scenes.py), add what the sheet's dialogue needs to kcare's ENV
(`qa` place layouts for left / middle / right answers, an `entrance` spot), reset the robot, then play the
dialogue through the agent WebSocket exactly as the dashboard does (direct-mode plans; `listen` events are
answered with the user's typed line, `speak` / `say` lines are recorded). Ground truth from /sim/command grades
what happened (where the object ended up, what the robot said vs. where things really are).
Writes <out>/<scenario>.json (+ head / overview frames) and prints a summary line per step.
"""
import argparse, asyncio, json, math, os, sys, threading, time, urllib.request

import cv2
import numpy as np
import rclpy

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
from skill_bench import Bench, http                                       # noqa: E402

API = os.environ.get('KCARE_API', 'http://127.0.0.1:8001')
WS = API.replace('http', 'ws') + '/ws/agent'

# qa place layouts (ENV[...]['qa']), the shape kcare's qa uses on the real site (counter2@kitchen, dressroom)
QA_SHELF = {'views': ['down', 'straight'], 'frame': 'person', 'scene': 'an open shelf niche in a white kitchen cabinet',
            'places': {'left': {'desc': 'the left half of the niche', 'ko': '왼쪽 선반', 'en': 'the left shelf', 'vi': 'kệ bên trái'},
                       'right': {'desc': 'the right half of the niche', 'ko': '오른쪽 선반', 'en': 'the right shelf', 'vi': 'kệ bên phải'}}}
QA_RACK = {'views': ['straight', 'up'], 'frame': 'person', 'scene': 'a clothes rack with shirts on hangers',
           'places': {'left': {'desc': 'the left third of the rail', 'ko': '왼쪽', 'en': 'the left', 'vi': 'bên trái'},
                      'middle': {'desc': 'the middle of the rail', 'ko': '가운데', 'en': 'the middle', 'vi': 'ở giữa'},
                      'right': {'desc': 'the right third of the rail', 'ko': '오른쪽', 'en': 'the right', 'vi': 'bên phải'}}}
QA_CHEST = {'views': ['down', 'straight'], 'frame': 'person', 'scene': 'handbags standing on top of a white chest of drawers',
            'places': {'left': {'desc': 'the left end of the chest top', 'ko': '왼쪽', 'en': 'the left', 'vi': 'bên trái'},
                       'middle': {'desc': 'the middle of the chest top', 'ko': '가운데', 'en': 'the middle', 'vi': 'ở giữa'},
                       'right': {'desc': 'the right end of the chest top', 'ko': '오른쪽', 'en': 'the right', 'vi': 'bên phải'}}}
QA = "qa::once=True, input='text', source='dashboard', lang='ko'"

# step: ('ws', plan, [typed answers]) | ('skill', name, inputs) ; checks run after the scenario
SCENARIOS = {
    's1': dict(title='높은 위치 정보 전달 (선반 물건 위치)', scene='vi_s1', env={'shelf@kitchen': {'qa': QA_SHELF}},
               steps=[('skill', 'move', 'shelf@kitchen'),
                      ('ws', QA, ['텀블러가 선반 어디에 있어?']), ('ws', QA, ['머그컵이 선반 어디에 있어?']),
                      ('ws', QA, ['접시는 어디에 있어?']), ('ws', QA, ['와인잔은 어디에 있어?'])],
               expect={'tumbler': 'right', 'mug': 'left', 'plate': 'right', 'wine glass': 'left'}),
    's2': dict(title='유사물체 정보/물건 전달 (라면)', scene='vi_s2', env={'shelf@kitchen': {'qa': QA_SHELF}},
               steps=[('skill', 'move', 'shelf@kitchen'), ('ws', QA, ['선반에 무슨 라면 있어?']),
                      ('skill', 'pick', '신라면'), ('skill', 'place', 'counter@kitchen')],
               moved={'shin ramyun': 'counter@kitchen'}),
    's3': dict(title='물건 전달 (뜨거운 냄비)', scene='vi_s3',
               steps=[('skill', 'move', 'counter@kitchen'), ('skill', 'pick', 'blue handle'), ('skill', 'place', 'dining table@kitchen')],
               moved={'pot': 'dining table@kitchen'}),
    's4': dict(title='정보 전달 (떨어진 숟가락)', scene='vi_s4',
               steps=[('skill', 'move', 'dining table@kitchen'), ('skill', 'arm_down', ''), ('skill', 'find', 'spoon'),
                      ('ws', QA, ['숟가락 어디에 떨어졌어?'])]),
    's5': dict(title='정리 정돈 (식탁 정리)', scene='vi_s5',
               steps=[('skill', 'pick', 'bowl@dining table@kitchen'), ('skill', 'place', 'counter@kitchen'),
                      ('skill', 'pick', 'mug@dining table@kitchen'), ('skill', 'place', 'counter@kitchen'),
                      ('skill', 'pick', 'cloth@dining table@kitchen'), ('skill', 'place', 'washer@laundry room')],
               moved={'bowl': 'counter@kitchen', 'mug': 'counter@kitchen', 'cloth': 'washer@laundry room'}),
    's6': dict(title='사용자 상태 인식 (낙상)', scene='vi_s6', env={'hall@restroom': {'loc': {'x': 3.1, 'y': 0.3, 'rz': -20.0},
                                                                              'default_mode': 'front', 'height': 0.6}},
               steps=[('skill', 'move', 'hall@restroom'), ('skill', 'moveh', 'down'), ('skill', 'detect', 'person lying on the floor'),
                      ('skill', 'detect', 'person'), ('skill', 'rest_detect', 'person')]),
    's8': dict(title='유사물체 정보 전달 (옷 / 가방)', scene='vi_s8',
               env={'hanger@living room': {'qa': QA_RACK}, 'chest@bedroom': {'qa': QA_CHEST}},
               steps=[('skill', 'move', 'hanger@living room'), ('ws', QA, ['행거에 흰색 셔츠 어디 있어?']),
                      ('ws', QA, ['체크무늬 셔츠는 어디 있어?']),
                      ('skill', 'move', 'chest@bedroom'), ('ws', QA, ['파란색 핸드백 어디 있어?']),
                      ('skill', 'pick', 'blue bag'), ('skill', 'place', 'sofa@living room')],
               moved={'blue bag': 'sofa@living room'}),
    's9': dict(title='정리 정돈 (신발)', scene='vi_s9', env={'entrance@hall': {'loc': {'x': 4.35, 'y': 0.1, 'rz': 0.0},
                                                                         'default_mode': 'front', 'height': 0.4}},
               steps=[('skill', 'move', 'entrance@hall'), ('skill', 'align_shoes', 'black shoes')]),
}


async def run_ws(plan, answers, log, timeout=600):
    """Play one direct-mode plan through /ws/agent like the dashboard; answer `listen` events with `answers`."""
    import websockets
    said, events, t0 = [], [], time.time()
    answers = list(answers)
    async with websockets.connect(WS, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({'prompt': plan, 'direct': True, 'lang': 'ko', 'log_data': False}))
        while time.time() - t0 < timeout:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            ev = json.loads(raw)
            small = {k: v for k, v in ev.items() if k not in ('image', 'log_image', 'world', 'rgb') and not
                     (isinstance(v, str) and len(v) > 400)}
            events.append(small)
            if ev.get('speak'):
                said.append(('speak', ev['speak']))
            if ev.get('say'):
                said.append(('say', ev['say']))
            lis = ev.get('listen')
            if isinstance(lis, dict) and lis.get('id'):
                if lis.get('prompt'):
                    said.append(('prompt', lis['prompt']))
                text = answers.pop(0) if answers else ''
                log(f'      listen {lis.get("id")[:8]} -> user: {text!r}')
                await asyncio.get_event_loop().run_in_executor(None, lambda: http('POST', f'/agent/listen/{lis["id"]}', {'text': text}))
            if ev.get('event') in ('done', 'error') or ev.get('type') == 'done':
                break
    return {'plan': plan, 'said': said, 'events': events[-40:], 'sec': round(time.time() - t0, 1)}


def skill(name, inputs):
    t0 = time.time()
    try:
        r = http('POST', f'/skill/{name}', {'inputs': inputs})
    except Exception as e:
        r = {'isdone': False, 'msg': f'HTTP: {e}'}
    m = r.get('msg')
    return {'skill': name, 'inputs': inputs, 'isdone': bool(r.get('isdone')), 'sec': round(time.time() - t0, 1),
            'msg': (m if isinstance(m, str) else json.dumps(m, ensure_ascii=False, default=str))[:500] if m else '',
            'keys': sorted(k for k in r if k not in ('msg',))[:20],
            'ins': {k: {kk: vv for kk, vv in v.items() if kk in ('pose_3d', 'score', 'box', 'islying')}
                    for k, v in (r.get('ins') or {}).items()} if isinstance(r.get('ins'), dict) else None}


def side_of(spots, key, centre, n):
    """Which of n equal parts (left .. right, for a person facing the front) a point is in."""
    sp = spots[key]
    fy = math.radians(sp['front_yaw_deg'])
    d = np.array(centre[:2]) - np.array(sp['centre'])
    u = float(d @ np.array([-math.sin(fy), math.cos(fy)]))      # +u = to the RIGHT of someone facing the front
    k = min(n - 1, max(0, int((u + sp['width'] / 2) / sp['width'] * n)))
    return (['left', 'right'] if n == 2 else ['left', 'middle', 'right'])[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scenarios', nargs='*', default=list(SCENARIOS))
    ap.add_argument('--out', default=f'{SIM}/out/vi_demo')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    b = Bench()
    threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start()
    b.cli.wait_for_service(timeout_sec=10)
    import yaml
    qa_cfg = http('GET', '/skill-configs/QA_CONFIGS') or {}
    if os.environ.get('QA_URL'):                                   # the VLM on this host, not through the tunnel
        qa_cfg['url'] = os.environ['QA_URL']
        http('PUT', '/skill-configs/QA_CONFIGS', qa_cfg)
    print('QA_CONFIGS', {k: qa_cfg.get(k) for k in ('url', 'model')})
    for sid in a.scenarios:
        sc = SCENARIOS[sid]
        rec = {'id': sid, 'title': sc['title'], 'scene': sc['scene'], 'steps': []}
        log = lambda s: print(f'[{time.strftime("%T")}] {sid} {s}', flush=True)
        r = b.sim(cmd='load', name=sc['scene'])
        env = r['env']
        for k, v in sc.get('env', {}).items():
            env[k] = {**env.get(k, {}), **v}
        http('PUT', '/skill-configs/ENV', env); http('PUT', '/skill-configs/MAP', r['map'])
        if os.environ.get('QA_URL'):                           # the reload rewrote the site override (QA_CONFIGS too)
            q = http('GET', '/skill-configs/QA_CONFIGS') or {}
            q['url'] = os.environ['QA_URL']; http('PUT', '/skill-configs/QA_CONFIGS', q)
        b.fresh(time.time())
        b.sim(cmd='reset', what='all')
        http('PUT', '/agent/world', {'holding': None, 'found': None, 'opened': []})
        skill('detect', 'apple')                                  # wake the ROS services after a reload
        lay = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))
        spots = lay['spots']
        T0 = b.sim(cmd='truth')
        log(f"{sc['title']} — scene {sc['scene']} loaded")
        for i, st in enumerate(sc['steps']):
            if st[0] == 'skill':
                res = skill(st[1], st[2])
                log(f"  {st[1]}::{st[2]} -> {res['isdone']} ({res['sec']} s) {res['msg'][:160]}")
            else:
                res = asyncio.run(run_ws(st[1], st[2], log))
                for kind, text in res['said']:
                    log(f'      robot {kind}: {text}')
            res['truth'] = {k: v['centre'] for k, v in b.sim(cmd='truth')['objects'].items()}
            b.save_imgs(f"{a.out}/{sid}_{i:02d}")
            rec['steps'].append({'step': list(st), **res})
        T1 = b.sim(cmd='truth')
        checks = []
        for obj, part in (sc.get('expect') or {}).items():          # where things really are (left / right ...)
            body = 'obj_' + obj.replace(' ', '_')
            key = 'shelf@kitchen' if sid in ('s1', 's2') else None
            if key and body in T0['objects']:
                real = side_of(spots, key, T0['objects'][body]['centre'], 2)
                checks.append({'object': obj, 'sheet_expects': part, 'really': real})
        for obj, dest in (sc.get('moved') or {}).items():
            body = 'obj_' + obj.replace(' ', '_')
            c = T1['objects'].get(body, {}).get('centre')
            from skill_bench import surface_of
            checks.append({'object': obj, 'should_be_on': dest, 'is_on': surface_of(spots, c) if c else None, 'at': c})
        rec['checks'] = checks
        for c in checks:
            log(f'  check {c}')
        json.dump(rec, open(f'{a.out}/{sid}.json', 'w'), indent=1, ensure_ascii=False, default=str)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
