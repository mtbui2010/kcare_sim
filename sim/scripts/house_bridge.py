#!/usr/bin/env python3
"""The simulated kcare robot in the house, behind the same ROS 2 interfaces as the real one.

    source /workspace/ros2/ros2_ws/install/setup.bash
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python3 scripts/house_bridge.py [--rate 10]

kcare_robot's connections (configs/locations/<site>/connections.json) talk to
these exactly as on the robot, so its skills (move, find, ...) run unchanged:

  topics    /joint_states                         [lift, head pan, head tilt, joint1..7, gripper]
            /femto/color/image_raw (rgb8)  /femto/depth/aligned (16UC1 mm)  /femto/{color,depth}/camera_info
            /hand/camera/color/image_raw/compressed  /hand/camera/depth/image_rect_raw  /hand/camera/depth/camera_info
            /robot_pose/{mobile_pose, mobile_base_tool_pose, arm_base_tool_pose, map_base_tool_pose}
            /sim/overview/image_raw/compressed    follow camera, for people watching (not a robot sensor)
            /sim/topdown/image_raw/compressed     fixed camera above the whole house (ceiling hidden, walls shown)
            /sim/drawer_states (JointState)       how far each drawer is open — sim ground truth, for evaluation only
  /sim/arm_servo (JointState, subscribed)          streamed joint targets (joint1..7, virtual_gripper_joint) for a
                                                   learned policy at 30 Hz: set straight on the servos, no trajectory
  actions   /navigate_to_pose (nav2)  /kaair_worker/{arm_moveJ, arm_moveL, arm_moveT, lift_move, head_move}
            /body/tool_controller/gripper_cmd (control_msgs GripperCommand)
  services  /mobile/{shift_pose, rotate, absolute_rotate}  /create_object_marker  /create_tool_marker  /arm/init_set

The base is kinematic (mocap): navigation plans A* on an occupancy grid made from
the house geometry and drives the base along it. Lift, head, arm and gripper
are the simulated position servos; arm goals are joint-space moves (MoveL /
MoveT go through IK first). Frames: map = world, base_footprint = the robot base.
"""
import argparse, array, heapq, math, os, sys, threading, time

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from builtin_interfaces.msg import Time as TimeMsg
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
from kaair_msgs.action import HeadMove, LiftMove, MoveJoint, MoveLinear, MoveTool
from kaair_msgs.srv import CreateObjectMarker, MobileRotate, MobileShift
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from std_srvs.srv import Trigger

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, os.path.join(SIM, 'scripts'))
import check_sim as cs  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 8)]
JOINTS = ['lift_joint', 'head_joint1', 'head_joint2'] + ARM + ['virtual_gripper_joint']   # kcare's /joint_states order
ACT = {'lift_joint': 'lift', 'head_joint1': 'head_pan', 'head_joint2': 'head_tilt',
       'virtual_gripper_joint': 'gripper', **{j: j for j in ARM}}
SPEED = {'lift_joint': 0.08, 'head_joint1': 1.0, 'head_joint2': 1.0, 'virtual_gripper_joint': 0.05}  # per s
ARM_SPEED = 0.9                     # rad/s at velocity_scale 1
BASE_V, BASE_W = 0.45, 0.9          # m/s, rad/s
WALL_TOP = 2.5                       # top-down camera: frame the house at wall height
ROBOT_R = 0.30                      # planner inflation [m]: slamtec base ~0.28 + 2 cm (0.36 closed the 0.9 m doors)
MAP_FRAME = 'slamware_map'
DEG = math.pi / 180


def u8(b):
    """bytes -> uint8[] field value; assigning plain bytes makes rclpy convert element by element (~250 ms/frame)."""
    return array.array('B', b)


def stamp(t):
    return TimeMsg(sec=int(t), nanosec=int((t % 1) * 1e9))


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Sim:
    """MuJoCo model + a real-time stepping thread. All access to `d` holds `lock`."""

    def __init__(self, xml, start):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.lock = threading.RLock()
        self.aid = {j: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for j, a in ACT.items()}
        self.qadr = {j: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS}
        self.fingers = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f)]
                        for f in ('left_finger_tip_joint', 'right_finger_tip_joint')]
        self.base = np.array([start['x'], start['y'], start['yaw_deg'] * DEG])
        # start like the robot after init: lift home (lift::0.6), head straight, arm folded (front), gripper open
        q0 = {'lift_joint': 0.17, 'head_joint1': 0.0, 'head_joint2': -40 * DEG, 'virtual_gripper_joint': 0.1,
              **dict(zip(ARM, np.radians([90, 0, 20, 12, -40, 97, -10])))}
        for j, v in q0.items():
            self.d.qpos[self.qadr[j]] = v
            self.d.ctrl[self.aid[j]] = v
        for f in self.fingers:
            self.d.qpos[f] = 0.05
        self.traj = {}                       # joint -> (q0, q1, t0, T)
        self.base_goal = None                # ('turn', yaw) | ('drive', x, y, v, check) ; advanced in sim time
        self.base_done = threading.Event(); self.base_done.set()
        self.base_ok = True
        self._set_mocap()
        mujoco.mj_forward(self.m, self.d)
        self.t = 0.0
        self.running = True

    # base pose (kinematic)
    def _set_mocap(self):
        x, y, a = self.base
        self.d.mocap_pos[0] = [x, y, 0.0]
        self.d.mocap_quat[0] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]

    def set_base(self, x, y, a):
        with self.lock:
            self.base[:] = (x, y, wrap(a))

    def base_cmd(self, goal):
        with self.lock:
            self.base_ok = True
            self.base_done.clear()
            self.base_goal = goal

    def base_stop(self):
        with self.lock:
            self.base_goal = None
            self.base_done.set()

    def _base_step(self, dt):
        g = self.base_goal
        if g is None:
            return
        x, y, a = self.base
        if g[0] == 'turn':
            e = wrap(g[1] - a)
            if abs(e) <= BASE_W * dt:
                self.base[2] = wrap(g[1]); self.base_goal = None; self.base_done.set()
            else:
                self.base[2] = wrap(a + math.copysign(BASE_W * dt, e))
            return
        _, gx, gy, v, check = g
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        if dist <= v * dt:
            nx, ny = gx, gy
        else:
            nx, ny = x + dx / dist * v * dt, y + dy / dist * v * dt
        if check is not None and not check(nx, ny):
            self.base_ok = False; self.base_goal = None; self.base_done.set()
            return
        self.base[0], self.base[1] = nx, ny
        if dist <= v * dt:
            self.base_goal = None; self.base_done.set()

    def q(self, j):
        return float(self.d.qpos[self.qadr[j]])

    def move_joints(self, targets, duration):
        """Start smooth joint moves; returns when they are scheduled (not when done)."""
        with self.lock:
            for j, v in targets.items():
                self.traj[j] = (self.d.ctrl[self.aid[j]], float(v), self.t, max(duration, 1e-3))

    def busy(self, joints):
        with self.lock:
            return any(j in self.traj for j in joints)

    def loop(self):
        dt = self.m.opt.timestep
        wall0, sim0 = time.time(), self.t
        while self.running:
            with self.lock:
                for _ in range(8):                           # 16 ms of physics per lock
                    for j, (a, b, t0, T) in list(self.traj.items()):
                        s = (self.t - t0) / T
                        self.d.ctrl[self.aid[j]] = a + (b - a) * smooth(s)
                        if s >= 1:
                            del self.traj[j]
                    self._base_step(dt)
                    self._set_mocap()
                    mujoco.mj_step(self.m, self.d)
                    self.t += dt
            lag = (self.t - sim0) - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:                                 # fell behind (heavy render): don't try to catch up
                wall0, sim0 = time.time(), self.t


class Planner:
    """Occupancy grid from the static house geometry + A*."""

    def __init__(self, sim, res=0.05, bounds=None):
        m, d = sim.m, sim.d
        robot_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_footprint')
        robot = {b for b in range(m.nbody) if self._under(m, b, robot_root)}
        objs = {b for b in range(m.nbody) if m.body(b).name.startswith('obj_')}
        (bx0, bx1), (by0, by1) = (bounds['x'], bounds['y']) if bounds else ((-2.0, 10.0), (-5.0, 5.0))
        self.res, self.x0, self.y0 = res, bx0 - 0.5, by0 - 0.5          # grid = house bounds + 0.5 m margin
        self.W, self.H = int((bx1 - bx0 + 1.0) / res), int((by1 - by0 + 1.0) / res)
        occ = np.zeros((self.H, self.W), np.uint8)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b in robot or b in objs or m.geom(g).name in ('floor',) or m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 \
                    and not m.geom(g).name.startswith(('wall', 'lintel')):
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
            w = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            if w[:, 2].max() < 0.03 or w[:, 2].min() > 1.6:
                continue
            i0, j0 = self.cell(w[:, 0].min(), w[:, 1].min())
            i1, j1 = self.cell(w[:, 0].max(), w[:, 1].max())
            occ[max(j0, 0):j1 + 1, max(i0, 0):i1 + 1] = 1
        self.occ = occ
        k = int(math.ceil(ROBOT_R / res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        self.inflated = cv2.dilate(occ, ((xx ** 2 + yy ** 2) <= k * k).astype(np.uint8))

    @staticmethod
    def _under(m, b, root):
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False

    def cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def xy(self, i, j):
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def free(self, i, j, grid=None):
        g = self.inflated if grid is None else grid
        return 0 <= i < self.W and 0 <= j < self.H and not g[j, i]

    def nearest_free(self, i, j, max_r=20):
        for r in range(max_r):
            best = [(abs(di) + abs(dj), i + di, j + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                    if max(abs(di), abs(dj)) == r and self.free(i + di, j + dj)]
            if best:
                return min(best)[1:]
        return None

    def plan(self, start, goal):
        s, g = self.cell(*start), self.cell(*goal)
        if not self.free(*s):
            s = self.nearest_free(*s) or s
        g_free = g if self.free(*g) else self.nearest_free(*g)
        if g_free is None:
            return None
        openq, came, cost = [(0.0, s)], {s: None}, {s: 0.0}
        nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while openq:
            _, c = heapq.heappop(openq)
            if c == g_free:
                break
            for di, dj, w in nb:
                n = (c[0] + di, c[1] + dj)
                if not self.free(*n) and n != g_free:
                    continue
                nc = cost[c] + w
                if nc < cost.get(n, 1e18):
                    cost[n], came[n] = nc, c
                    heapq.heappush(openq, (nc + math.hypot(n[0] - g_free[0], n[1] - g_free[1]), n))
        if g_free not in came:
            return None
        path, c = [], g_free
        while c is not None:
            path.append(c); c = came[c]
        path = path[::-1]
        # line-of-sight shortcutting
        pts, k = [path[0]], 0
        while k < len(path) - 1:
            nxt = len(path) - 1
            while nxt > k + 1 and not self._los(path[k], path[nxt]):
                nxt -= 1
            pts.append(path[nxt]); k = nxt
        way = [self.xy(*p) for p in pts[1:]]
        if g == g_free:                                      # end exactly at the requested point
            way[-1] = tuple(goal)
        return way

    def _los(self, a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        for t in np.linspace(0, 1, n + 1):
            if not self.free(int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))):
                return False
        return True


class ArmIK:
    """Joint angles for a TCP pose in base_footprint (URDF FK, like the expert)."""

    def __init__(self):
        self.ref = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
        self.lo = np.array([self.ref.joints[j]['limit'][0] for j in ARM])
        self.hi = np.array([self.ref.joints[j]['limit'][1] for j in ARM])

    def fk(self, q):
        return self.ref.fk(q)

    def solve(self, q_now, T_goal):
        fixed = dict(q_now)
        x0 = np.array([q_now[j] for j in ARM])

        def res(x):
            T = self.ref.fk(dict(fixed, **dict(zip(ARM, x))))['tool_tcp_link']
            return np.concatenate([(T[:3, 3] - T_goal[:3, 3]) * 20, (T[:3, 2] - T_goal[:3, 2]) * 2,
                                   (T[:3, 1] - T_goal[:3, 1]) * 2, (x - x0) * 0.01])
        sol = least_squares(res, np.clip(x0, self.lo, self.hi), bounds=(self.lo, self.hi), xtol=1e-9, ftol=1e-9)
        T = self.ref.fk(dict(fixed, **dict(zip(ARM, sol.x))))['tool_tcp_link']
        return sol.x, float(np.linalg.norm(T[:3, 3] - T_goal[:3, 3]))


class NoiseBank:
    """Stands in for the rng camera_model.apply draws its per-pixel noise from: a few pre-drawn
    standard-normal fields, cycled (drawing 2.7 M normals per 720p frame halved the camera rate)."""

    def __init__(self, n=6, seed=0):
        self.gen, self.n, self.bank, self.k = np.random.default_rng(seed), n, {}, 0

    def standard_normal(self, shape):
        if shape not in self.bank:
            self.bank[shape] = [self.gen.standard_normal(shape).astype(np.float32) for _ in range(self.n)]
        self.k += 1
        return self.bank[shape][self.k % self.n]


class Bridge(Node):
    def __init__(self, sim, planner, rate, camera_model=True, depth_model=True):
        super().__init__('kcare_house_sim')
        self.sim, self.planner, self.ik = sim, planner, ArmIK()
        cg = ReentrantCallbackGroup()
        cam = yaml.safe_load(open(f'{SIM}/config/camera_real.yaml'))
        self.K = {c: (cam[c]['fx'], cam[c]['fy'], cam[c]['cx'], cam[c]['cy'], cam[c]['width'], cam[c]['height'])
                  for c in ('head', 'wrist')}
        self.frames = {}                                     # cam -> dict(rgb, depth, T_cam, base, t)
        self.D = {c: [float(v) for v in cam[c]['distortion']] for c in ('head', 'wrist')}
        # the real sensors' look (camera_model.py, fitted to the robot's Femto / D405 images): lens distortion,
        # auto-exposure, colour, noise, JPEG. The depth image gets the same distortion remap, so pixel -> 3D
        # (on_marker, pinhole K like the robot's object_marker_server) sees what the robot would.
        self.cams = None
        if camera_model:
            import camera_model as cmod
            self.cams = cmod.load_camera_models(profile='home')   # colours fitted at home (calibrate_home.py)
            self.dcams = cmod.load_depth_models(profile='home') if depth_model else {}   # Femto ToF / D405 stereo depth artefacts
            self.cam_rng = np.random.default_rng()
            self.noise = NoiseBank()
            for c in self.cams.values():
                c.episode_jitter(self.cam_rng)
        self.flock = threading.Lock()
        pub = lambda t, typ: self.create_publisher(typ, t, 10)
        self.p_js = pub('/joint_states', JointState)
        self.p_pose = {k: pub(f'/robot_pose/{k}', PoseStamped)
                       for k in ('mobile_pose', 'mobile_base_tool_pose', 'arm_base_tool_pose', 'map_base_tool_pose')}
        self.p_head = (pub('/femto/color/image_raw', Image), pub('/femto/depth/aligned', Image),
                       pub('/femto/color/camera_info', CameraInfo), pub('/femto/depth/camera_info', CameraInfo))
        self.p_wrist = (pub('/hand/camera/color/image_raw/compressed', CompressedImage),
                        pub('/hand/camera/depth/image_rect_raw', Image), pub('/hand/camera/depth/camera_info', CameraInfo),
                        pub('/hand/camera/color/camera_info', CameraInfo))
        self.p_over = pub('/sim/overview/image_raw/compressed', CompressedImage)
        self.p_top = pub('/sim/topdown/image_raw/compressed', CompressedImage)
        self.p_drawers = pub('/sim/drawer_states', JointState)
        m = sim.m
        self.drawer_joints = [m.joint(i).name for i in range(m.njnt) if m.joint(i).name.endswith('_slide')]
        self.create_subscription(JointState, '/sim/arm_servo', self.on_servo, 10, callback_group=cg)
        self.create_timer(1 / 30, self.pub_state, callback_group=cg)

        A = lambda t, name, cb: ActionServer(self, t, name, execute_callback=cb, callback_group=cg,
                                             goal_callback=lambda g: GoalResponse.ACCEPT,
                                             cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.actions = [A(NavigateToPose, '/navigate_to_pose', self.on_nav),
                        A(MoveJoint, '/kaair_worker/arm_moveJ', self.on_movej),
                        A(MoveLinear, '/kaair_worker/arm_moveL', self.on_movel),
                        A(MoveTool, '/kaair_worker/arm_moveT', self.on_movet),
                        A(LiftMove, '/kaair_worker/lift_move', self.on_lift),
                        A(HeadMove, '/kaair_worker/head_move', self.on_head),
                        A(GripperCommand, '/body/tool_controller/gripper_cmd', self.on_grip)]
        S = lambda t, name, cb: self.create_service(t, name, cb, callback_group=cg)
        S(MobileShift, '/mobile/shift_pose', self.on_shift)
        S(MobileRotate, '/mobile/rotate', lambda rq, rs: self.on_rotate(rq, rs, relative=True))
        S(MobileRotate, '/mobile/absolute_rotate', lambda rq, rs: self.on_rotate(rq, rs, relative=False))
        S(CreateObjectMarker, '/create_object_marker', lambda rq, rs: self.on_marker(rq, rs, 'head'))
        S(CreateObjectMarker, '/create_tool_marker', lambda rq, rs: self.on_marker(rq, rs, 'wrist'))
        S(Trigger, '/arm/init_set', self.on_init)
        self.rate = rate
        threading.Thread(target=self.render_loop, daemon=True).start()
        self.get_logger().info('kcare house sim ready')

    # ── state ────────────────────────────────────────────────────────────────
    def q_all(self):
        with self.sim.lock:
            return {j: self.sim.q(j) for j in JOINTS}

    def T_base_tcp(self, q=None):
        return self.ik.fk(q or self.q_all())['tool_tcp_link']

    def T_map_base(self):
        x, y, a = self.sim.base
        T = np.eye(4); T[:3, :3] = Rot.from_euler('z', a).as_matrix(); T[:3, 3] = [x, y, 0]
        return T

    def pose_msg(self, T, frame, t):
        p = PoseStamped(); p.header.frame_id = frame; p.header.stamp = stamp(t)
        p.pose.position.x, p.pose.position.y, p.pose.position.z = map(float, T[:3, 3])
        qx, qy, qz, qw = Rot.from_matrix(T[:3, :3]).as_quat()
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = map(float, (qx, qy, qz, qw))
        return p

    def on_servo(self, msg):
        """Streamed targets from a policy: no interpolation (the policy already runs at 30 Hz), ctrl range clipped."""
        with self.sim.lock:
            for n, v in zip(msg.name, msg.position):
                if n in self.sim.aid:
                    self.sim.traj.pop(n, None)
                    lo, hi = self.sim.m.actuator_ctrlrange[self.sim.aid[n]]
                    self.sim.d.ctrl[self.sim.aid[n]] = float(np.clip(v, lo, hi))

    def pub_state(self):
        t = time.time()
        q = self.q_all()
        js = JointState(); js.header.stamp = stamp(t); js.name = JOINTS; js.position = [q[j] for j in JOINTS]
        self.p_js.publish(js)
        if self.drawer_joints:
            with self.sim.lock:
                op = [float(self.sim.d.qpos[self.sim.m.jnt_qposadr[self.sim.m.joint(n).id]]) for n in self.drawer_joints]
            ds = JointState(); ds.header.stamp = stamp(t); ds.name = self.drawer_joints; ds.position = op
            self.p_drawers.publish(ds)
        fk = self.ik.fk(q)
        Tb = self.T_map_base()
        self.p_pose['mobile_pose'].publish(self.pose_msg(Tb, MAP_FRAME, t))
        self.p_pose['mobile_base_tool_pose'].publish(self.pose_msg(fk['tool_tcp_link'], 'base_footprint', t))
        self.p_pose['map_base_tool_pose'].publish(self.pose_msg(Tb @ fk['tool_tcp_link'], MAP_FRAME, t))
        if 'link_base' in fk:
            self.p_pose['arm_base_tool_pose'].publish(
                self.pose_msg(np.linalg.inv(fk['link_base']) @ fk['tool_tcp_link'], 'link_base', t))

    # ── cameras ──────────────────────────────────────────────────────────────
    def caminfo(self, cam, t, frame):
        fx, fy, cx, cy, w, h = self.K[cam]
        ci = CameraInfo(); ci.header.stamp = stamp(t); ci.header.frame_id = frame
        ci.width, ci.height = w, h
        ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.d = self.D[cam] if self.cams is not None else [0.0] * 5; ci.distortion_model = 'plumb_bob'
        return ci

    def render_loop(self):
        m, sim = self.sim.m, self.sim
        R = {'head': mujoco.Renderer(m, 720, 1280), 'wrist': mujoco.Renderer(m, 480, 848), 'over': mujoco.Renderer(m, 480, 848)}
        cam_id = {'head': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'head_rgb'),
                  'wrist': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_rgb')}
        over = mujoco.MjvCamera()
        over_opt = mujoco.MjvOption(); over_opt.geomgroup[2] = 0      # walls hidden: see the robot from above
        # fixed top-down camera over the whole house: walls kept (they read as the floor plan), ceiling and
        # downlights dropped from its scene after each update; frame sized to the house bounds
        pl = self.planner
        bx0, by0 = pl.x0 + 0.5, pl.y0 + 0.5
        bx1, by1 = pl.x0 + pl.W * pl.res - 0.5, pl.y0 + pl.H * pl.res - 0.5
        TW = 960; TH = int(round(TW * (by1 - by0 + 0.6) / (bx1 - bx0 + 0.6))) // 2 * 2
        R['top'] = mujoco.Renderer(m, TH, TW)
        top = mujoco.MjvCamera()
        top.lookat[:] = [(bx0 + bx1) / 2, (by0 + by1) / 2, 0.0]
        top.azimuth, top.elevation = 90.0, -89.9                       # x (east) to the right, y (north) up, as the drawio plan
        top.distance = ((by1 - by0) / 2 + 0.3) / math.tan(math.radians(m.vis.global_.fovy / 2)) + WALL_TOP
        hide_top = {g for g in range(m.ngeom) if m.geom(g).name == 'ceiling' or m.geom(g).name.startswith('downlight')}
        period = 1.0 / self.rate
        while rclpy.ok() and sim.running:
            t0 = time.time()
            with sim.lock:
                snap = {}
                for c in ('head', 'wrist'):
                    R[c].update_scene(sim.d, cam_id[c])
                    T = np.eye(4); T[:3, :3] = sim.d.cam_xmat[cam_id[c]].reshape(3, 3); T[:3, 3] = sim.d.cam_xpos[cam_id[c]]
                    snap[c] = T
                x, y, a = sim.base
                over.lookat[:] = [x + 0.6 * math.cos(a), y + 0.6 * math.sin(a), 0.6]
                over.distance, over.azimuth, over.elevation = 3.6, math.degrees(a) - 30, -45
                R['over'].update_scene(sim.d, over, scene_option=over_opt)
                R['top'].update_scene(sim.d, top)
                sc = R['top'].scene
                for k in range(sc.ngeom):
                    if sc.geoms[k].objtype == mujoco.mjtObj.mjOBJ_GEOM and sc.geoms[k].objid in hide_top:
                        sc.geoms[k].rgba[3] = 0.0
                base = self.T_map_base()
            t = time.time()
            out = {}
            for c in ('head', 'wrist'):
                rgb = R[c].render()
                R[c].enable_depth_rendering(); depth = R[c].render().copy(); R[c].disable_depth_rendering()
                depth[depth > 10.0] = 0.0
                if self.cams is not None:
                    cm = self.cams[c]
                    lum = cv2.remap(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cm.map_x, cm.map_y, cv2.INTER_LINEAR)
                    rgb = cm.apply(rgb, self.noise, cam_T=snap[c], t=t)      # + motion blur / rolling shutter
                    depth = cv2.remap(depth, cm.map_x, cm.map_y, cv2.INTER_NEAREST, borderValue=0.0)
                    if c in self.dcams:
                        depth = self.dcams[c].apply(depth, lum, self.noise)
                out[c] = dict(rgb=rgb, depth=depth, T_cam=snap[c], base=base, t=t)
            with self.flock:
                self.frames.update(out)
            dmm = lambda dm: (dm * 1000).clip(0, 65535).astype(np.uint16)
            h = out['head']
            im = Image(); im.header.stamp = stamp(t); im.header.frame_id = 'femto_color_optical_frame'
            im.height, im.width, im.encoding, im.step = 720, 1280, 'rgb8', 1280 * 3; im.data = u8(h['rgb'].tobytes())
            dm = Image(); dm.header = im.header; dm.height, dm.width, dm.encoding, dm.step = 720, 1280, '16UC1', 1280 * 2
            dm.data = u8(dmm(h['depth']).tobytes())
            self.p_head[0].publish(im); self.p_head[1].publish(dm)
            self.p_head[2].publish(self.caminfo('head', t, 'femto_color_optical_frame'))
            self.p_head[3].publish(self.caminfo('head', t, 'femto_color_optical_frame'))
            w = out['wrist']
            ci = CompressedImage(); ci.header.stamp = stamp(t); ci.header.frame_id = 'camera_color_optical_frame'
            ci.format = 'rgb8; jpeg compressed bgr8'
            ci.data = u8(cv2.imencode('.jpg', cv2.cvtColor(w['rgb'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
            wd = Image(); wd.header = ci.header; wd.height, wd.width, wd.encoding, wd.step = 480, 848, '16UC1', 848 * 2
            wd.data = u8(dmm(w['depth']).tobytes())
            self.p_wrist[0].publish(ci); self.p_wrist[1].publish(wd)
            self.p_wrist[2].publish(self.caminfo('wrist', t, 'camera_color_optical_frame'))
            self.p_wrist[3].publish(self.caminfo('wrist', t, 'camera_color_optical_frame'))
            ov = R['over'].render()
            oc = CompressedImage(); oc.header.stamp = stamp(t); oc.format = 'rgb8; jpeg compressed bgr8'
            oc.data = u8(cv2.imencode('.jpg', cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            self.p_over.publish(oc)
            tp = R['top'].render()
            tc = CompressedImage(); tc.header.stamp = stamp(t); tc.format = 'rgb8; jpeg compressed bgr8'
            tc.data = u8(cv2.imencode('.jpg', cv2.cvtColor(tp, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            self.p_top.publish(tc)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def on_marker(self, req, res, cam):
        """Pixels (+ optional depths) of the last published frame -> points in req.base_frame."""
        with self.flock:
            f = self.frames.get(cam)
        empty = lambda msg: (setattr(res, 'success', False), setattr(res, 'message', msg), res)[-1]
        if f is None:
            return empty('no_depth: no frame rendered yet')
        u, v, dd = list(req.u_array), list(req.v_array), list(req.d_array)
        if not u or len(u) != len(v):
            return empty('u_array and v_array must be non-empty and the same length')
        if dd and len(dd) != len(u):
            dd = []
        fx, fy, cx, cy, W, H = self.K[cam]
        frame = (req.base_frame or '').strip()
        T_out = np.linalg.inv(f['base']) if frame in ('base_footprint', 'base_link') else np.eye(4)
        xs, ys, zs, raw, r_, g_, b_ = [], [], [], [], [], [], []
        ok = 0
        for i, (uu, vv) in enumerate(zip(u, v)):
            uu, vv = int(np.clip(uu, 0, W - 1)), int(np.clip(vv, 0, H - 1))
            z_img = float(f['depth'][vv, uu])
            raw.append(z_img * 1000)
            r, g, b = f['rgb'][vv, uu]; r_.append(int(r)); g_.append(int(g)); b_.append(int(b))
            z = dd[i] if dd and math.isfinite(dd[i]) and dd[i] > 0 else z_img
            if z <= 0:
                patch = f['depth'][max(vv - 3, 0):vv + 4, max(uu - 3, 0):uu + 4]
                z = float(np.median(patch[patch > 0])) if (patch > 0).any() else 0.0
            if z <= 0:
                xs.append(float('nan')); ys.append(float('nan')); zs.append(float('nan')); continue
            p_opt = np.array([(uu - cx) / fx * z, (vv - cy) / fy * z, z])
            p_cam = np.array([p_opt[0], -p_opt[1], -p_opt[2], 1.0])       # optical -> MuJoCo camera axes
            p = T_out @ (f['T_cam'] @ p_cam)
            xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2])); ok += 1
        res.success, res.message = ok > 0, 'ok' if ok else 'all_no_depth'
        res.x_array, res.y_array, res.z_array, res.depth_raw_array = xs, ys, zs, raw
        res.r_array, res.g_array, res.b_array = r_, g_, b_
        return res

    # ── base ─────────────────────────────────────────────────────────────────
    def _base(self, goal, gh=None):
        """Run one base primitive in sim time; False if cancelled or blocked."""
        self.sim.base_cmd(goal)
        while not self.sim.base_done.wait(0.05):
            if gh is not None and gh.is_cancel_requested:
                self.sim.base_stop()
                return False
        return self.sim.base_ok

    def _drive(self, gh, waypoints, final_yaw=None, v=BASE_V):
        for (gx, gy) in waypoints:
            x, y, _ = self.sim.base
            if math.hypot(gx - x, gy - y) > 0.02 and not self._base(('turn', math.atan2(gy - y, gx - x)), gh):
                return False
            if not self._base(('drive', gx, gy, v, None), gh):
                return False
        return final_yaw is None or self._base(('turn', final_yaw), gh)

    def _turn_to(self, yaw, gh=None):
        return self._base(('turn', yaw), gh)

    def on_nav(self, gh):
        p = gh.request.pose.pose
        goal = (p.position.x, p.position.y)
        x, y, _ = self.sim.base
        way = self.planner.plan((x, y), goal)
        res = NavigateToPose.Result()
        if way is None:
            self.get_logger().warn(f'navigate_to_pose: no path to {goal}')
            gh.abort(); return res
        self.get_logger().info(f'navigate_to_pose -> ({goal[0]:.2f}, {goal[1]:.2f}, {math.degrees(yaw_of(p.orientation)):.0f} deg), '
                               f'{len(way)} waypoints')
        ok = self._drive(gh, way, yaw_of(p.orientation))
        if gh.is_cancel_requested:
            gh.canceled(); return res
        gh.succeed() if ok else gh.abort()
        return res

    def _clear(self, x, y):
        i, j = self.planner.cell(x, y)
        k = int(0.25 / self.planner.res)                     # the base's own footprint only (no inflation)
        sub = self.planner.occ[max(j - k, 0):j + k + 1, max(i - k, 0):i + k + 1]
        return not sub.any()

    def _in_thread(self, fn, wait):
        if wait:
            return fn()
        threading.Thread(target=fn, daemon=True).start()
        return True

    def on_shift(self, req, res):
        x, y, a = self.sim.base
        goal = (x + req.distance * math.cos(a), y + req.distance * math.sin(a))
        res.successed = bool(self._in_thread(lambda: self._drive_straight(goal, a), req.wait))
        return res

    def _drive_straight(self, goal, a):
        """shift_pose moves along the heading without turning (backwards for a negative distance)."""
        ok = self._base(('drive', goal[0], goal[1], 0.25, self._clear))
        if not ok:
            self.get_logger().warn('shift_pose: blocked')
        return ok

    def on_rotate(self, req, res, relative):
        yaw = self.sim.base[2] + req.theta if relative else req.theta
        res.successed = bool(self._in_thread(lambda: self._turn_to(wrap(yaw)), req.wait))
        return res

    # ── lift / head / gripper / arm ──────────────────────────────────────────
    def _wait(self, gh, joints, timeout=30.0):
        t0 = time.time()
        while self.sim.busy(joints):
            if gh.is_cancel_requested:
                with self.sim.lock:                           # stop where we are
                    for j in joints:
                        self.sim.traj.pop(j, None)
                        self.sim.d.ctrl[self.sim.aid[j]] = self.sim.q(j)
                return False
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.02)
        time.sleep(0.15)                                      # let the servo settle
        return True

    def _go(self, gh, targets, speed):
        lo = {j: self.sim.m.actuator_ctrlrange[self.sim.aid[j]] for j in targets}
        targets = {j: float(np.clip(v, *lo[j])) for j, v in targets.items()}
        with self.sim.lock:
            T = max(abs(v - self.sim.d.ctrl[self.sim.aid[j]]) / speed[j] for j, v in targets.items())
        self.sim.move_joints(targets, T)
        return self._wait(gh, list(targets), timeout=T + 10), targets

    def on_lift(self, gh):
        ok, _ = self._go(gh, {'lift_joint': gh.request.target_height}, SPEED)
        res = LiftMove.Result(); res.success = ok; res.final_height = self.sim.q('lift_joint')
        res.message = '' if ok else 'cancelled'
        (gh.succeed() if ok else gh.abort()) if not gh.is_cancel_requested else gh.canceled()
        return res

    def on_head(self, gh):
        r = gh.request
        ok, _ = self._go(gh, {'head_joint1': r.head_joint1, 'head_joint2': r.head_joint2}, SPEED)
        res = HeadMove.Result(); res.success = ok; res.message = '' if ok else 'cancelled'
        res.final_head_joint1, res.final_head_joint2 = self.sim.q('head_joint1'), self.sim.q('head_joint2')
        (gh.succeed() if ok else gh.abort()) if not gh.is_cancel_requested else gh.canceled()
        return res

    def on_grip(self, gh):
        ok, t = self._go(gh, {'virtual_gripper_joint': gh.request.command.position}, SPEED)
        res = GripperCommand.Result()
        res.position = self.sim.q('virtual_gripper_joint')
        res.reached_goal = abs(res.position - t['virtual_gripper_joint']) < 0.004
        res.stalled = not res.reached_goal
        gh.succeed() if not gh.is_cancel_requested else gh.canceled()
        return res

    def _arm_to(self, gh, x, scale):
        speed = {j: ARM_SPEED * (scale if scale > 0 else 0.6) for j in ARM}
        return self._go(gh, dict(zip(ARM, x)), speed)[0]

    def on_movej(self, gh):
        r = gh.request
        ok = self._arm_to(gh, list(r.target_joints), r.velocity_scale)
        res = MoveJoint.Result(); res.success = ok; res.message = 'ok' if ok else 'cancelled or timeout'
        res.final_joints = [self.sim.q(j) for j in ARM]
        (gh.succeed() if ok else gh.abort()) if not gh.is_cancel_requested else gh.canceled()
        return res

    def _cartesian(self, gh, T_goal, scale, result):
        q = self.q_all()
        x, err = self.ik.solve(q, T_goal)
        if err > 0.01:
            result.success, result.message = False, f'IK failed: {err * 1000:.0f} mm from the goal'
            gh.abort(); return result
        ok = self._arm_to(gh, x, scale)
        T = self.T_base_tcp()
        result.final_pose = self.pose_msg(T, 'base_footprint', time.time()).pose
        result.success, result.message = ok, 'ok' if ok else 'cancelled or timeout'
        (gh.succeed() if ok else gh.abort()) if not gh.is_cancel_requested else gh.canceled()
        return result

    def on_movel(self, gh):
        r = gh.request
        Rg = Rot.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix() if (r.qx or r.qy or r.qz or r.qw) else np.eye(3)
        T_now = self.T_base_tcp()
        T = np.eye(4)
        if r.is_relative:
            T[:3, 3] = T_now[:3, 3] + [r.x, r.y, r.z]
            T[:3, :3] = T_now[:3, :3] @ Rg
        else:
            T[:3, 3] = [r.x, r.y, r.z]; T[:3, :3] = Rg
            if r.base_frame and r.base_frame not in ('base_footprint', 'base_link'):
                self.get_logger().warn(f'arm_moveL: base_frame {r.base_frame!r} treated as base_footprint')
        return self._cartesian(gh, T, r.velocity_scale, MoveLinear.Result())

    def on_movet(self, gh):
        r = gh.request
        T_now = self.T_base_tcp()
        D = np.eye(4); D[:3, 3] = [r.dx, r.dy, r.dz]
        if r.qx or r.qy or r.qz or r.qw:
            D[:3, :3] = Rot.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
        return self._cartesian(gh, T_now @ D, r.velocity_scale, MoveTool.Result())

    def on_init(self, req, res):
        res.success, res.message = True, 'sim arm ready'
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default=f'{SIM}/mjcf/house.xml')
    ap.add_argument('--rate', type=float, default=10.0, help='camera frame rate [Hz]')
    ap.add_argument('--no-camera-model', action='store_true', help='publish ideal renders (no Femto/D405 look)')
    ap.add_argument('--no-depth-model', action='store_true', help='camera model on RGB only, ideal depth')
    a = ap.parse_args()
    layout = yaml.safe_load(open(f'{SIM}/config/house_layout.yaml'))
    sim = Sim(a.xml, layout['start'])
    for _ in range(200):                                     # settle objects before anything reads the scene
        mujoco.mj_step(sim.m, sim.d)
    planner = Planner(sim, bounds=layout.get('bounds'))
    cv2.imwrite(f'{SIM}/out/house/occupancy.png', ((1 - planner.inflated) * 180 + (1 - planner.occ) * 75)[::-1].astype(np.uint8))
    threading.Thread(target=sim.loop, daemon=True).start()
    rclpy.init()
    node = Bridge(sim, planner, a.rate, camera_model=not a.no_camera_model,
                  depth_model=not a.no_depth_model)
    ex = MultiThreadedExecutor(num_threads=12)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        sim.running = False
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
