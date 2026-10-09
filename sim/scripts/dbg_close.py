import sys, yaml, mujoco, numpy as np
sys.path.insert(0, '/sim/scripts')
import check_sim as cs, drawer_ready as dr, expert_close_drawer as ex
cfg = yaml.safe_load(open('/sim/config/mjcf.yaml')); task = yaml.safe_load(open('/sim/config/task_close_drawer_vla2.yaml'))
ref = cs.UrdfFK('/sim/urdf/raw/kaair_slamtec.urdf'); q = dr.ready_q(task, task['mode'], cfg['home'])
xml, *_ = dr.build_scene(task, ref.fk(q)['tool_tcp_link']); open('/sim/mjcf/_dbg.xml', 'w').write(xml)
for tgt in (1, 2):
    m = mujoco.MjModel.from_xml_path('/sim/mjcf/_dbg.xml'); d = mujoco.MjData(m)
    def rend(m, d, name, drw, **k):
        if len(trace) % 15 == 0:
            tcp = d.body('tool_tcp_link').xpos; h = d.geom_xpos[m.geom(f'cab_drawer{tgt}_handle').id]
            print(f'   {name:8s} drawer {drw*100:5.1f}  tcp-handle {np.round(tcp - h, 3)}  grip {cs.get_q(m, d, "virtual_gripper_joint"):.3f}')
        trace.append(1)
    trace = []
    print('target', tgt); r = ex.run_episode(m, d, task, cfg, task['mode'], render=rend, drawer0=0.18, target=tgt); print('success', r['success'])
