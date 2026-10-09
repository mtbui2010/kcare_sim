"""Test kcare's vla_open_drawer plan skill in the house: reset, run, read /sim/drawer_states."""
import json, sys, threading, time
sys.path.insert(0, '/sim/scripts')
import rclpy
from sensor_msgs.msg import JointState
from skill_bench import Bench, http, skill
rclpy.init(); b = Bench()
st = {}
b.create_subscription(JointState, '/sim/drawer_states', lambda m: st.update(dict(zip(m.name, m.position))), 1)
threading.Thread(target=rclpy.spin, args=(b,), daemon=True).start(); b.cli.wait_for_service(timeout_sec=10)
r = b.sim(cmd='apply', scene={})
http('PUT', '/skill-configs/ENV', r['env']); http('PUT', '/skill-configs/MAP', r['map']); b.fresh(time.time())
skill('detect', 'apple')
time.sleep(2); print('drawer joints:', {k: round(v, 3) for k, v in st.items()})
res = []
for prompt in ['open drawer', 'open the top drawer', 'open the bottom drawer']:
    for k in range(3):
        b.sim(cmd='reset', what='all'); time.sleep(1.5)
        r, t0, t1 = skill('vla_open_drawer', prompt)
        time.sleep(1.0)
        op = {k2: round(v * 100, 1) for k2, v in st.items() if k2.startswith('A')} or {k2: round(v * 100, 1) for k2, v in st.items()}
        res.append({'prompt': prompt, 'isdone': bool(r.get('isdone')), 'sec': round(t1 - t0, 1), 'open_cm': op, 'msg': str(r.get('msg', ''))[:120]})
        print(json.dumps(res[-1], ensure_ascii=False), flush=True)
json.dump(res, open('/sim/out/vla_skill_test.json', 'w'), indent=1)
