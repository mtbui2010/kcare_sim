#!/usr/bin/env python3
"""PICK phase A ("reach to view") with a VLA, as a ROS action for kcare's `vla_pick_approach` skill.

    PYTHONPATH=/sim/third_party:/sim/third_party/vlaclient_deps python3 scripts/vla_pick_node.py \
        [--url http://127.0.0.1:11451] [--model <served model id>] [--seconds 12]

/pick/approach (max_interfaces/action/OpenDrawer: {mode, prompt} -> {success, message}):
  1. the start pose of expert_pick_approach for the mode — movej::give, head panned to the mode side and tilted
     down, lift at kcare's home, gripper open — through the normal arm / head / lift / gripper actions;
  2. vla-serve's ChunkExecutor at 30 Hz. observation = head image (1280x720 -> 848x480) + wrist image + the 17-D
     state of expert_pick_approach.STATE_NAMES (joints, gripper, lift, head pan / tilt, base motion since the
     start in the start frame, mode one-hot); each 10-D action (ACTION_NAMES: joint targets, gripper, lift,
     base_forward) goes to /sim/arm_servo;
  3. stops when the commanded action has settled (or after --seconds) and reports the base offset it used.
Whether the wrist camera now sees the object is checked by the caller (fine_move re-detects); this node only
runs the policy.
"""
import argparse, json, math, os, sys, threading, time

import cv2
import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from kaair_msgs.action import HeadMove, LiftMove, MoveJoint
from max_interfaces.action import OpenDrawer
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, JointState

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import expert_pick_approach as ex  # noqa: E402  (start pose + state / action layout of the training data)
from vlaserve.client import ChunkExecutor, PolicyClient  # noqa: E402

ARM = ex.ARM
GRIP = 'virtual_gripper_joint'
IMG_WH = (848, 480)


def yaw_of(o):
    return math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))


class VlaPickNode(Node):
    def __init__(self, a):
        super().__init__('vla_pick_node')
        self.a = a
        cg = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.head = self.wrist = self.base = None
        self.q = {}
        self.create_subscription(Image, '/femto/color/image_raw', self.on_head, 1, callback_group=cg)
        self.create_subscription(CompressedImage, '/hand/camera/color/image_raw/compressed', self.on_wrist, 1, callback_group=cg)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10, callback_group=cg)
        self.create_subscription(PoseStamped, '/robot_pose/mobile_pose', self.on_base, 10, callback_group=cg)
        self.servo = self.create_publisher(JointState, '/sim/arm_servo', 10)
        self.ac = {'head': ActionClient(self, HeadMove, '/kaair_worker/head_move', callback_group=cg),
                   'lift': ActionClient(self, LiftMove, '/kaair_worker/lift_move', callback_group=cg),
                   'arm': ActionClient(self, MoveJoint, '/kaair_worker/arm_moveJ', callback_group=cg),
                   'grip': ActionClient(self, GripperCommand, '/body/tool_controller/gripper_cmd', callback_group=cg)}
        self.client = PolicyClient(a.url, model=a.model)
        if not a.model:
            import httpx
            self.client.model = httpx.get(f'{a.url}/v1/models', timeout=10).json()['models'][0]['model']
        self.server = ActionServer(self, OpenDrawer, '/pick/approach', execute_callback=self.execute, callback_group=cg,
                                   goal_callback=lambda g: GoalResponse.ACCEPT,
                                   cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.mode, self.base0, self.last = 'right', None, []
        self.get_logger().info(f'/pick/approach ready, policy {self.client.model} at {a.url}')

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

    def on_base(self, m):
        with self.lock:
            self.base = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))

    def base_delta(self):
        """Base motion since the episode start, in the start frame: (forward, left, yaw)."""
        if self.base0 is None or self.base is None:
            return 0.0, 0.0, 0.0
        x0, y0, a0 = self.base0
        x, y, a = self.base
        c, s = math.cos(a0), math.sin(a0)
        dx, dy = x - x0, y - y0
        return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(a - a0), math.cos(a - a0))

    def observation(self):
        with self.lock:
            q = dict(self.q)
            imgs = {'head': self.head.copy(), 'wrist': self.wrist.copy()}
        bdx, bdy, byaw = self.base_delta()
        onehot = [float(self.mode == m_) for m_ in ex.MODES]
        state = [q[j] for j in ARM] + [q[GRIP], q['lift_joint'], q['head_joint1'], q['head_joint2'], bdx, bdy, byaw] + onehot
        return {'images': imgs, 'state': np.array(state, np.float32)}

    def send(self, action):
        a = [float(v) for v in action[:10]]
        js = JointState(); js.header.stamp = self.get_clock().now().to_msg()
        js.name = ARM + [GRIP, 'lift_joint', 'base_forward']; js.position = a
        self.servo.publish(js)
        self.last = (self.last + [a])[-int(self.a.fps):]

    def settled(self):
        """The last second of commands moved less than ~2 mm / 1 deg: the policy holds still."""
        if len(self.last) < int(self.a.fps):
            return False
        arr = np.array(self.last)
        return bool(np.max(np.ptp(arr[:, :7], axis=0)) < math.radians(1.0) and np.ptp(arr[:, 8]) < 0.003
                    and np.ptp(arr[:, 9]) < 0.003)

    # ── start pose (the same as the expert's) ──
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

    def start_pose(self, mode):
        h = HeadMove.Goal(); h.head_joint1 = math.radians(ex.HEAD_RZ[mode]); h.head_joint2 = math.radians(ex.EXPERT['head_tilt'])
        l = LiftMove.Goal(); l.target_height = ex.lift_joint(ex.LIFT['home'][mode])
        g = GripperCommand.Goal(); g.command.position = ex.EXPERT['gripper']
        j = MoveJoint.Goal(); j.target_joints = [math.radians(v) for v in ex.GIVE[mode]]; j.velocity_scale = 0.6
        ts = [threading.Thread(target=self._goal, args=args) for args in (('head', h), ('lift', l), ('grip', g))]
        [t.start() for t in ts]
        self._goal('arm', j)
        [t.join() for t in ts]
        time.sleep(0.5)

    # ── the action ──
    def execute(self, gh):
        prompt = gh.request.prompt or 'reach the object'
        mode = gh.request.mode if gh.request.mode in ex.MODES else 'right'
        res, fb, t0 = OpenDrawer.Result(), OpenDrawer.Feedback(), time.time()
        try:
            fb.status = 'moving to the start pose'; gh.publish_feedback(fb)
            self.start_pose(mode)
            for _ in range(50):
                if self.head is not None and self.wrist is not None and GRIP in self.q and self.base is not None:
                    break
                time.sleep(0.1)
            self.mode, self.base0, self.last = mode, self.base, []
            self.client.reset()
            stop = threading.Event()

            def watch():
                while not stop.is_set():
                    if gh.is_cancel_requested or self.settled():
                        stop.set()
                    time.sleep(0.05)
            threading.Thread(target=watch, daemon=True).start()
            fb.status = f'policy running: "{prompt}"'; gh.publish_feedback(fb)
            self.get_logger().info(f'/pick/approach "{prompt}" (mode {mode})')
            stats = ChunkExecutor(self.client, get_observation=self.observation, send_action=self.send,
                                  fps=self.a.fps).run(task=prompt, seconds=self.a.seconds, stop=stop)
            settled = self.settled()
            stop.set()
            bdx, bdy, byaw = self.base_delta()
            res.success = settled and not gh.is_cancel_requested
            res.message = json.dumps({'settled': settled, 'sec': round(time.time() - t0, 1), 'base_forward': round(bdx, 3),
                                      'base_side': round(bdy, 3), 'stats': stats.summary()})
        except Exception as e:
            res.success, res.message = False, json.dumps({'error': f'vla_pick_node: {e}'})
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
    ap.add_argument('--url', default='http://127.0.0.1:11451')
    ap.add_argument('--model', default='')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--seconds', type=float, default=12.0)
    a = ap.parse_args()
    rclpy.init()
    node = VlaPickNode(a)
    ex_ = MultiThreadedExecutor(num_threads=6)
    ex_.add_node(node)
    try:
        ex_.spin()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
