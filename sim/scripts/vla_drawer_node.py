#!/usr/bin/env python3
"""ROS 2 action server /drawer/open: open a drawer with the SmolVLA policy served by vla-serve.

    source /workspace/ros2/ros2_ws/install/setup.bash
    PYTHONPATH=/sim/third_party:/sim/third_party/vlaclient_deps:$PYTHONPATH python3 scripts/vla_drawer_node.py \
        [--url http://127.0.0.1:11450] [--model <served model id>] [--seconds 15]

kcare_robot's `vla_drawer_open` skill sends {mode, prompt} to /drawer/open
(max_interfaces/action/OpenDrawer). This node:
  1. moves the robot to the pose the policy was trained from (scripts/drawer_ready.ready_q: head down to the
     side, lift low, arm at approach_drawer, gripper open) with the robot's own actions (head_move, lift_move,
     arm_moveJ, gripper_cmd);
  2. runs vla-serve's ChunkExecutor at 30 Hz: observation = head image (1280x720 -> 848x480, INTER_AREA, as the
     recorder did) + wrist image (848x480) + state [joint1..7, gripper width]; each action (absolute joint
     targets) goes to /sim/arm_servo;
  3. reports how far the drawers opened. /sim/drawer_states is SIMULATION ground truth: on the robot this node
     would need its own success check (depth / VLM), so here it only grades the result.
The model is whatever the vla-serve server has loaded (or --model).
"""
import argparse, math, os, sys, threading, time

import cv2
import numpy as np
import rclpy
import yaml
from control_msgs.action import GripperCommand
from kaair_msgs.action import HeadMove, LiftMove, MoveJoint
from max_interfaces.action import OpenDrawer
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, JointState

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import drawer_ready as dr  # noqa: E402
from vlaserve.client import ChunkExecutor, PolicyClient  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 8)]
GRIP = 'virtual_gripper_joint'
IMG_WH = (848, 480)


class VlaDrawerNode(Node):
    def __init__(self, a):
        super().__init__('vla_drawer_node')
        self.a = a
        cg = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.head = self.wrist = None
        self.q, self.drawers = {}, {}
        self.create_subscription(Image, '/femto/color/image_raw', self.on_head, 1, callback_group=cg)
        self.create_subscription(CompressedImage, '/hand/camera/color/image_raw/compressed', self.on_wrist, 1, callback_group=cg)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10, callback_group=cg)
        self.create_subscription(JointState, '/sim/drawer_states', self.on_drawers, 10, callback_group=cg)
        self.servo = self.create_publisher(JointState, '/sim/arm_servo', 10)
        self.ac = {'head': ActionClient(self, HeadMove, '/kaair_worker/head_move', callback_group=cg),
                   'lift': ActionClient(self, LiftMove, '/kaair_worker/lift_move', callback_group=cg),
                   'arm': ActionClient(self, MoveJoint, '/kaair_worker/arm_moveJ', callback_group=cg),
                   'grip': ActionClient(self, GripperCommand, '/body/tool_controller/gripper_cmd', callback_group=cg)}
        cfg = yaml.safe_load(open(f'{SIM}/config/mjcf.yaml'))
        self.task = yaml.safe_load(open(f'{SIM}/config/task_open_drawer.yaml'))
        self.home = cfg['home']
        self.client = PolicyClient(a.url, model=a.model)
        if not a.model:                                        # use what the server has loaded
            self.client.model = self._first_model()
        self.server = ActionServer(self, OpenDrawer, '/drawer/open', execute_callback=self.execute, callback_group=cg,
                                   goal_callback=lambda g: GoalResponse.ACCEPT,
                                   cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.get_logger().info(f'/drawer/open ready, policy {self.client.model} at {a.url}')

    def _first_model(self):
        import httpx
        return httpx.get(f'{self.a.url}/v1/models', timeout=10).json()['models'][0]['model']

    # ── sensors ──
    def on_head(self, m):
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        with self.lock:
            self.head = cv2.resize(im, IMG_WH, interpolation=cv2.INTER_AREA)

    def on_wrist(self, m):
        im = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        with self.lock:
            self.wrist = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def on_js(self, m):
        with self.lock:
            self.q.update(zip(m.name, m.position))

    def on_drawers(self, m):
        with self.lock:
            self.drawers = dict(zip(m.name, m.position))

    def observation(self):
        with self.lock:
            return {'images': {'head': self.head.copy(), 'wrist': self.wrist.copy()},
                    'state': np.array([self.q[j] for j in ARM] + [self.q[GRIP]], np.float32)}

    def send(self, action):
        js = JointState(); js.header.stamp = self.get_clock().now().to_msg()
        js.name = ARM + [GRIP]; js.position = [float(v) for v in action[:8]]
        self.servo.publish(js)

    # ── robot actions for the ready pose ──
    def _goal(self, key, goal, timeout=40.0):
        ac = self.ac[key]
        if not ac.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f'{key} action server not available')
        ev, box = threading.Event(), {}

        def on_result(f):
            box['r'] = f.result(); ev.set()

        def on_goal(f):
            gh = f.result()
            if not gh.accepted:
                box['r'] = None; ev.set(); return
            gh.get_result_async().add_done_callback(on_result)
        ac.send_goal_async(goal).add_done_callback(on_goal)
        if not ev.wait(timeout):
            raise RuntimeError(f'{key}: no result in {timeout:.0f} s')
        return box['r']

    def ready_pose(self, mode):
        q = dr.ready_q(self.task, mode if mode in ('left', 'right') else self.task['mode'], self.home)
        h = HeadMove.Goal(); h.head_joint1, h.head_joint2 = q['head_joint1'], q['head_joint2']
        l = LiftMove.Goal(); l.target_height = q['lift_joint']
        g = GripperCommand.Goal(); g.command.position = 0.1
        j = MoveJoint.Goal(); j.target_joints = [q[k] for k in ARM]; j.velocity_scale = 0.6
        ts = [threading.Thread(target=self._goal, args=args) for args in (('head', h), ('lift', l), ('grip', g))]
        [t.start() for t in ts]
        self._goal('arm', j)
        [t.join() for t in ts]
        time.sleep(0.5)

    # ── the action ──
    def execute(self, gh):
        prompt = gh.request.prompt or 'open drawer'
        mode = gh.request.mode or self.task['mode']
        res = OpenDrawer.Result()
        fb = OpenDrawer.Feedback()
        t0 = time.time()
        try:
            fb.status = 'moving to the ready pose'; gh.publish_feedback(fb)
            self.ready_pose(mode)
            for _ in range(50):                              # fresh frames after the move
                if self.head is not None and self.wrist is not None and GRIP in self.q:
                    break
                time.sleep(0.1)
            start = dict(self.drawers)
            self.client.reset()
            stop = threading.Event()

            def watch():
                while not stop.is_set():
                    if gh.is_cancel_requested:
                        stop.set()
                    time.sleep(0.05)
            threading.Thread(target=watch, daemon=True).start()
            fb.status = f'policy running: "{prompt}"'; gh.publish_feedback(fb)
            self.get_logger().info(f'/drawer/open "{prompt}" (mode {mode})')
            stats = ChunkExecutor(self.client, get_observation=self.observation, send_action=self.send,
                                  fps=self.a.fps).run(task=prompt, seconds=self.a.seconds, stop=stop)
            stop.set()
            gain = {n.replace('_slide', ''): self.drawers.get(n, 0.0) - start.get(n, 0.0) for n in start}
            best = max(gain, key=gain.get) if gain else None
            opened = gain.get(best, 0.0) if best else 0.0
            res.success = bool(opened > self.a.success_m) and not gh.is_cancel_requested
            res.message = (f'{"opened" if res.success else "did not open"} {best} by {opened * 100:.1f} cm '
                           f'(sim ground truth) in {time.time() - t0:.1f} s; {stats.summary()}')
        except Exception as e:
            res.success, res.message = False, f'vla_drawer_node: {e}'
        self.get_logger().info(res.message)
        if gh.is_cancel_requested:
            gh.canceled()
        elif res.success:
            gh.succeed()
        else:
            gh.abort()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:11450')
    ap.add_argument('--model', default=None, help='served model id (default: the first one the server has loaded)')
    ap.add_argument('--seconds', type=float, default=15.0, help='policy run time (the eval timeout)')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--success-m', type=float, default=0.20, help='drawer travel counted as opened (as in eval)')
    a = ap.parse_args()
    rclpy.init()
    node = VlaDrawerNode(a)
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
