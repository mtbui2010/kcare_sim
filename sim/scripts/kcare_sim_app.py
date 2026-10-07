"""kcare_robot API for the house simulation, running the real robot's code read-only.

    uvicorn kcare_sim_app:app --app-dir /sim/scripts --host 0.0.0.0 --port 8001     (scripts/run_house.sh)

Code:    /kcare_mnt/remote_dir/{robot_agent, kcare_robot} = the dev PC's remote_dir,
         sshfs read-only (on the venv's path through kcare_remote.pth). Edits on .11 reach the sim at the
         next restart (bash scripts/run_house.sh restart-kcare).
Configs: /sim/kcare_sim_configs (common/ + locations/sim_house/), NOT the package's configs/: the real
         robot reads active_location, world_state.json, guides... from that shared folder, and the sim
         must never switch the robot's site or overwrite its world state. skills.json / buttons.json are
         refreshed from the robot's copy at every start; guides.json only seeded once (the sim may edit it).
Patches: two robot_agent bugs found in the sim, fixed here at import time so the shared source is not
         touched (drop a patch once the same fix is upstream; each one checks before applying):
           1. ActionClientAgent kept ONE result slot per agent: two concurrent goals on one action
              (kcare `move` sends two moveh 70 ms apart) took each other's result and one waited forever.
           2. POST /skill/<name> returned raw ndarrays (find via the head camera) -> HTTP 500;
              only that HTTP reply is converted, skills still pass each other real arrays.
"""
import logging, os, shutil, threading, time

CODE = os.environ.get('KCARE_CODE', '/kcare_mnt/remote_dir')
CFG = os.environ.get('KCARE_SIM_CONFIGS', '/sim/kcare_sim_configs')
LOCATION = os.environ.get('KCARE_SIM_LOCATION', 'sim_house')
log = logging.getLogger('kcare_sim_app')


def sync_common():
    src = f'{CODE}/kcare_robot/kcare_robot/configs/common'
    dst = f'{CFG}/common'
    os.makedirs(dst, exist_ok=True)
    for f, always in (('skills.json', True), ('buttons.json', True), ('guides.json', False)):
        if os.path.exists(f'{src}/{f}') and (always or not os.path.exists(f'{dst}/{f}')):
            shutil.copy(f'{src}/{f}', f'{dst}/{f}')
    open(f'{dst}/active_location', 'w').write(LOCATION + '\n')


def patch_action_client():
    import rclpy
    from robot_agent.connect.ros import node as rnode
    A = rnode.ActionClientAgent
    if getattr(A, '_kcare_sim_patched', False) or 'slot' in A.result_callback.__code__.co_varnames:
        return
    canceled = getattr(rnode, '_STATUS_CANCELED', 5)

    def _set_result(self, slot, data):
        self.rev_data = data
        if slot is not None:
            slot['data'] = data

    def send(self, data, **kwargs):
        try:
            assert self.encode_func is not None and self.decode_func is not None
            if self._node_cancelled():
                return {'isdone': False, 'msg': f'{self.id}: cancelled'}
            self._cancelled = False
            if self.do_log_msg:
                self.log_msg(data, msg_type='sent')
            goal_msg = self.encode_func(data, self.data_interface.Goal())
            if not self.executor.wait_for_server(timeout_sec=3.0):
                return {'isdone': False, 'msg': f'{self.id}: action server not available'}
            slot = {}
            fut = self.executor.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
            fut.add_done_callback(lambda f, s=slot: self.goal_response_callback(f, s))
            tic = time.perf_counter()
            while self.wait_until_done and rclpy.ok() and 'data' not in slot:
                time.sleep(0.02)
                if self.timeout is not None and time.perf_counter() - tic > self.timeout:
                    slot['data'] = {'isdone': False, 'msg': f'{self.id}: timed out waiting for result'}
                    break
            self.rev_data = None
            return slot.get('data')
        except Exception as e:                                   # the original is @exception_handler-wrapped
            return {'isdone': False, 'msg': f'{self.id}: {e}'}

    def goal_response_callback(self, future, slot=None):
        try:
            gh = future.result()
        except Exception as e:
            return _set_result(self, slot, {'isdone': False, 'msg': f'{self.id}: error getting goal handle: {e}'})
        if not gh.accepted:
            return _set_result(self, slot, {'isdone': False, 'msg': f'{self.id}: goal rejected by server'})
        with self._goal_lock:
            self._goal_handles.append(gh)
        if self._cancelled:
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
        gh.get_result_async().add_done_callback(lambda f, s=slot: self.result_callback(f, s))

    def result_callback(self, future, slot=None):
        try:
            out = future.result()
            if getattr(out, 'status', None) == canceled:
                _set_result(self, slot, {'isdone': False, 'msg': f'{self.id}: goal cancelled'})
            else:
                d = self.decode_func(out.result)
                _set_result(self, slot, {'ret': None} if d is None else d)
        except Exception as e:
            _set_result(self, slot, {'isdone': False, 'msg': f'{self.id}: error processing result: {e}'})
        self._forget_goal(getattr(future, '_goal_handle', None))
        if self.do_log_msg and slot is not None:
            self.log_msg(slot.get('data'), msg_type='retunred')

    A.send, A.goal_response_callback, A.result_callback = send, goal_response_callback, result_callback
    A._kcare_sim_patched = True
    log.warning('kcare_sim_app: patched ActionClientAgent (per-goal result slot)')


def patch_skill_results():
    """Replace the POST /skill/{name} route (before create_app includes the router) with one that makes the
    result JSON-safe. Only the HTTP reply is converted: skills still hand each other real arrays in a plan."""
    from robot_agent.api import skills as api_skills
    from robot_agent.core.unified_agent import _serialize_result
    r = api_skills.router
    if getattr(r, '_kcare_sim_patched', False):
        return
    r.routes = [rt for rt in r.routes if getattr(rt, 'path', '') != '/skill/{name}']

    @r.post('/skill/{name}')
    def execute_skill(name: str, params: dict = {}):
        from robot_agent.core.run_control import begin_run
        state = api_skills.current()
        node = state.dm._ros_node
        begin_run(node)          # a cancel must not leak into the next command
        return _serialize_result(state.sr.execute(name, params, node=node))
    r._kcare_sim_patched = True
    log.warning('kcare_sim_app: patched POST /skill/{name} (JSON-safe results)')


sync_common()
patch_action_client()
patch_skill_results()
from robot_agent import create_app  # noqa: E402

app = create_app(robot_pkg='kcare_robot', config_dir=CFG, location=LOCATION)
