#!/usr/bin/env python3
"""IK reachability sweep for pick / place: which robot_mode (front / left / right) reaches floor, low, table and
high objects, with which lift, stand-off and grasp type, and can the head camera see them. Pure kinematics on the
robot URDF (no physics), CPU only.

    python3 scripts/reach_sweep.py [--procs 16] [--out /sim/out/reach]

Model
  * Lift only moves the arm vertically, so reachability depends on the object position RELATIVE to the arm
    (d, a, h = z_grasp - lift_joint). We solve IK once per (mode, grasp, d, a, h) and pick the best lift per
    object height afterwards (lift_joint 0.13-0.67 = kcare lift 0.56-1.10, minus mobile_height 0.43).
  * Mode frame: d = distance from the base centre along the side the arm works on (front: +x; left / right: the
    side the mode's approach pose reaches, from FK), a = offset along the other horizontal axis.
  * Base shift: front can drive along d, side modes only along a (non-holonomic base) — evaluated on the same grid.
  * Grasps: top-down (tool z down, yaw free) with a pre-grasp 0.12 m above; side (tool z horizontal along d,
    fingers horizontal) with a pre-grasp 0.12 m back towards the robot.
  * Reachable = IK position error < 1 cm and approach-axis error < 5 deg for grasp AND pre-grasp, joint limits
    respected, every arm link origin above the floor (z > 0.02) and outside the base body (cylinder r 0.24 m,
    z < 0.45 m), and for side grasps the TCP not below the support surface.
  * Head visibility: object in the Femto colour camera FOV (assumed 80 x 51 deg) for some tilt in [-50, +10] deg
    at the mode's head pan (HEAD_CONFIGS rz: front 0, left -70, right +70; + the sim's -0.12 rad calibration).
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool

import numpy as np

SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
import check_sim as cs                                   # noqa: E402

ARM = [f'joint{i}' for i in range(1, 8)]
DEG = math.pi / 180
MOBILE_H = 0.43
LIFT_J = np.round(np.arange(0.13, 0.6701, 0.03), 3)      # lift_joint values (kcare lift 0.56..1.10)
Z_OBJ = [0.0, 0.05, 0.15, 0.3, 0.45, 0.75, 0.9, 1.1, 1.3, 1.5, 1.7]
GRASP_ABOVE = 0.04                                        # grasp point above the support surface (object centre)
D_GRID = np.round(np.arange(0.3, 1.1001, 0.1), 2)
A_GRID = np.round(np.arange(-0.5, 0.5001, 0.1), 2)
H_GRID = np.round(np.arange(-0.70, 1.6001, 0.05), 3)     # z_grasp - lift_joint
PRE = 0.12
HEAD_PAN = {'front': 0.0, 'left': -70 * DEG, 'right': 70 * DEG}
PAN_OFFSET = -0.12
FOV_H, FOV_V = 80 * DEG, 51 * DEG
# seeds per mode: kcare named poses (ARM_CONFIGS, degrees)
SEEDS = {
    'front': [[0, 15, 0, 15, -180, 90, 0], [90, 0, 20, 12, -40, 97, -10], [0, -26.3, 18.8, 1.5, -160.5, 62.2, 0],
              [0, 60, 0, 90, -180, 60, 0], [0, -30, 0, 60, -180, 0, 0]],
    'left': [[-90, 15, 0, 15, -180, 90, 0], [96.85, 18.01, -49.85, 14.47, -33.29, 53.71, -14.79],
             [86, 23.3, -182.5, 37, -0.2, 60.4, -185.5], [-39.7, 30.6, 36.3, 45.2, -58.4, 100.0, -5.6],
             [3.83, 82.74, 17.51, 81.92, -81.44, 107.12, 3.0]],
    'right': [[90, 15, 0, 15, -180, 90, 0], [-78.91, 24.26, 45.87, 24.81, -318.36, 52.09, 20.36],
              [89.6, -26.8, 0.3, 28.4, 0.9, 55, -179.9], [36.4, 33.9, -31.2, 51.7, -295.6, 100.0, 13.2],
              [-6.21, 81.91, -15.78, 80.38, -277.15, 105.37, -3.08]],
}


class Chain:
    """Fast FK along one URDF chain (root -> tip): fixed transforms precomputed."""

    def __init__(self, ref, tip):
        steps, link = [], tip
        parent_of = {j['child']: (n, j) for n, j in ref.joints.items()}
        while link in parent_of:
            n, j = parent_of[link]
            steps.append((n, j))
            link = j['parent']
        self.steps = steps[::-1]

    def fk(self, q, upto=None, collect=False):
        T = np.eye(4); pts = []
        for n, j in self.steps:
            T = T @ j['T']
            if j['type'] in ('revolute', 'continuous'):
                M = np.eye(4); M[:3, :3] = cs.axis_angle(j['axis'], q.get(n, 0.0)); T = T @ M
            elif j['type'] == 'prismatic':
                M = np.eye(4); M[:3, 3] = j['axis'] / np.linalg.norm(j['axis']) * q.get(n, 0.0); T = T @ M
            if collect:
                pts.append((j['child'], T[:3, 3].copy()))
        return (T, pts) if collect else T


def setup():
    global REF, ARMC, HEADC, LO, HI
    REF = cs.UrdfFK(f'{SIM}/urdf/raw/kaair_slamtec.urdf')
    ARMC = Chain(REF, 'tool_tcp_link')
    HEADC = Chain(REF, 'femto_color_optical_frame')
    LO = np.array([REF.joints[j]['limit'][0] for j in ARM])
    HI = np.array([REF.joints[j]['limit'][1] for j in ARM])


def work_dirs():
    """Unit vector (base frame) of the side each mode's arm works on, from FK of its approach pose."""
    setup()
    out = {}
    for mode in ('left', 'right'):
        q = dict(zip(ARM, np.radians(SEEDS[mode][4])), lift_joint=0.2)
        p = ARMC.fk(q)[:3, 3]
        out[mode] = [0.0, float(np.sign(p[1]))]
    out['front'] = [1.0, 0.0]
    return out


def mode_frame(mode, dirs):
    u = np.array(dirs[mode] + [0.0]); v = np.array([-u[1], u[0], 0.0])
    return u, v


def ik(q_fixed, target, zdir, yhoriz, seeds):
    from scipy.optimize import least_squares

    def res(x):
        T = ARMC.fk(dict(q_fixed, **dict(zip(ARM, x))))
        r = [(T[:3, 3] - target) * 20, (T[:3, 2] - zdir) * 2]
        if yhoriz:
            r.append([T[2, 1] * 2])                     # fingers open horizontally
        return np.concatenate(r)
    best = None
    for s in seeds:
        sol = least_squares(res, np.clip(s, LO + 1e-6, HI - 1e-6), bounds=(LO, HI), xtol=1e-8, ftol=1e-8, max_nfev=400)
        T = ARMC.fk(dict(q_fixed, **dict(zip(ARM, sol.x))))
        perr = float(np.linalg.norm(T[:3, 3] - target)); aerr = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ zdir, -1, 1))))
        if best is None or perr + aerr * 0.002 < best[1] + best[2] * 0.002:
            best = (sol.x, perr, aerr)
        if perr < 0.01 and aerr < 5:
            break
    return best


def collides(q, surf_z_rel, grasp):
    _, pts = ARMC.fk(q, collect=True)
    for name, p in pts:
        if not (name.startswith('link') or name in ('hand_base_link', 'tool_tcp_link')) or name == 'link_base':
            continue
        if p[2] < 0.02:                                   # floor
            return 'floor'
        if p[2] < 0.45 and math.hypot(p[0], p[1]) < 0.24:  # mobile base body
            return 'base'
    return None


def solve_cell(args):
    mode, grasp, h, dirs = args
    if 'REF' not in globals():
        setup()
    u, v = mode_frame(mode, dirs)
    lift_ref = 0.4                                        # solve at one lift; reachability is lift-invariant in h
    q_fixed = {'lift_joint': lift_ref}
    seeds = [np.radians(s) for s in SEEDS[mode]]
    out = {}
    prev = None
    for d in D_GRID:
        for a in A_GRID:
            g = u * d + v * a; g[2] = lift_ref + h
            if grasp == 'top':
                zdir, yh, pre = np.array([0, 0, -1.0]), False, g + [0, 0, PRE]
            else:
                zdir, yh, pre = u.copy(), True, g - u * PRE
            sd = ([prev] if prev is not None else []) + seeds
            b = ik(q_fixed, g, zdir, yh, sd)
            ok = b[1] < 0.01 and b[2] < 5
            why = None if ok else 'ik'
            if ok:
                qg = dict(q_fixed, **dict(zip(ARM, b[0])))
                # floor / base checks are done in the world frame later (depend on lift): keep the arm-relative pose
                bp = ik(q_fixed, pre, zdir, yh, [b[0]] + seeds[:2])
                ok = bp[1] < 0.01 and bp[2] < 5
                why = None if ok else 'pre-ik'
                if ok:
                    prev = b[0]
                    out[f'{d:.2f},{a:.2f}'] = [list(map(float, b[0])), list(map(float, bp[0]))]
                    continue
            out[f'{d:.2f},{a:.2f}'] = why
    return mode, grasp, float(h), out


def head_visible(mode, p_obj):
    """Some tilt in [-50, +10] deg sees p_obj (base frame) in the colour camera FOV."""
    for ry in range(-50, 11, 5):
        T = HEADC.fk({'head_joint1': HEAD_PAN[mode] + PAN_OFFSET, 'head_joint2': ry * DEG})
        pc = np.linalg.inv(T) @ np.append(p_obj, 1.0)
        x, y, z = pc[:3]
        if z > 0.15 and abs(math.atan2(x, z)) < FOV_H / 2 and abs(math.atan2(y, z)) < FOV_V / 2:
            return ry
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--out', default=f'{SIM}/out/reach')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for v_ in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v_] = '1'
    dirs = work_dirs()
    print('working directions', dirs, flush=True)
    jobs = [(m, g, h, dirs) for m in ('front', 'left', 'right') for g in ('top', 'side') for h in H_GRID]
    t0 = time.time()
    raw = {}
    with Pool(a.procs, initializer=setup) as pool:
        for k, (m, g, h, out) in enumerate(pool.imap_unordered(solve_cell, jobs)):
            raw[(m, g, round(h, 3))] = out
            if k % 20 == 0:
                print(f'{k}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    import pickle
    pickle.dump({'raw': raw, 'dirs': dirs}, open(f'{a.out}/raw_ik.pkl', 'wb'))
    setup()
    # world-frame evaluation per object height, lift, base shift
    res = {}
    for m in ('front', 'left', 'right'):
        u, v = mode_frame(m, dirs)
        for g in ('top', 'side'):
            for zo in Z_OBJ:
                zg = zo + GRASP_ABOVE
                cells = {}
                for d in D_GRID:
                    for aa in A_GRID:
                        best = None
                        for lj in LIFT_J:
                            h = round(round((zg - lj) / 0.05) * 0.05, 3)
                            if h not in [round(x, 3) for x in H_GRID]:
                                continue
                            r = raw[(m, g, h)].get(f'{d:.2f},{aa:.2f}')
                            if not isinstance(r, list):
                                continue
                            bad = None
                            for qa in r:
                                q = dict(zip(ARM, qa), lift_joint=float(lj))
                                bad = collides(q, zo, g)
                                if bad:
                                    break
                            if bad:
                                continue
                            # prefer the lift closest to the mode's home (less lift travel)
                            home = {'front': 0.17, 'left': 0.37, 'right': 0.37}[m]
                            if best is None or abs(lj - home) < abs(best - home):
                                best = float(lj)
                        p = u * d + v * aa; p[2] = zg
                        cells[(float(d), float(aa))] = (best, head_visible(m, p))
                res[(m, g, zo)] = cells

    def frac(cells, shift_ok, dset=None):
        n = ok = 0
        for (d, aa), (lj, _) in cells.items():
            if dset is not None and d not in dset:
                continue
            n += 1
            if lj is not None:
                ok += 1
            elif shift_ok:
                for s in (0.1, -0.1, 0.2, -0.2):
                    key = (round(d - s, 2), aa) if shift_ok == 'd' else (d, round(aa - s, 2))
                    if key in cells and cells[key][0] is not None:
                        ok += 1; break
        return ok / max(n, 1)

    summary = {'assumptions': {'fov_deg': [80, 51], 'grasp_above_surface_m': GRASP_ABOVE, 'pre_grasp_m': PRE,
                               'lift_joint_range': [float(LIFT_J[0]), float(LIFT_J[-1])], 'base_cylinder': [0.24, 0.45],
                               'working_dirs': dirs}, 'cells': {}}
    for (m, g, zo), cells in res.items():
        reach_d = sorted({d for (d, aa), (lj, _) in cells.items() if lj is not None})
        by_d = {f'{d:.1f}': sum(1 for (dd, aa), (lj, _) in cells.items() if dd == d and lj is not None) / len(A_GRID) for d in D_GRID}
        lifts = [lj for (lj, _) in cells.values() if lj is not None]
        vis = [c for c in cells.values() if c[0] is not None]
        summary['cells'][f'{m}|{g}|{zo}'] = dict(
            frac=frac(cells, None), frac_shift=frac(cells, 'd' if m == 'front' else 'a'), by_d=by_d,
            lift_kcare_median=(float(np.median(lifts)) + MOBILE_H) if lifts else None,
            head_visible_of_reachable=(sum(1 for c in vis if c[1] is not None) / len(vis)) if vis else None,
            d_range=[min(reach_d), max(reach_d)] if reach_d else None)
    json.dump(summary, open(f'{a.out}/summary.json', 'w'), indent=1)

    # heatmaps: fraction of the a-grid reachable vs (object height, distance d)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for m in ('front', 'left', 'right'):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        for ax, g in zip(axs, ('top', 'side')):
            M = np.array([[summary['cells'][f'{m}|{g}|{zo}']['by_d'][f'{d:.1f}'] for d in D_GRID] for zo in Z_OBJ])
            im = ax.imshow(M, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1,
                           extent=[D_GRID[0] - 0.05, D_GRID[-1] + 0.05, -0.5, len(Z_OBJ) - 0.5])
            ax.set_yticks(range(len(Z_OBJ))); ax.set_yticklabels([f'{z:.2f}' for z in Z_OBJ])
            ax.set_xlabel('distance from base centre along the working side d [m]'); ax.set_ylabel('object support height [m]')
            ax.set_title(f'{m} mode, {g} grasp: share of lateral offsets reachable (best lift)')
            for i, zo in enumerate(Z_OBJ):
                for j, d in enumerate(D_GRID):
                    if M[i, j] > 0:
                        ax.text(d, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=6,
                                color='black' if M[i, j] > 0.6 else 'white')
        fig.colorbar(im, ax=axs, shrink=0.8)
        fig.savefig(f'{a.out}/reach_{m}.png', dpi=110); plt.close(fig)
    print('done', time.time() - t0, flush=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Collision-, clearance- and occlusion-aware evaluation (--collide): the MuJoCo robot model with full geometry,
# furniture per target (table slab; shelf with side panels, back board and a board above), the whole path
# start -> pre-grasp -> grasp -> retreat with the object (0.06 x 0.06 x 0.08 m box in the gripper) -> start,
# clearance margins (arm / object >= M_BODY to head, head camera, neck, lift column, torso, base;
# >= M_ENV to furniture) and the head camera's view.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════
ARM_BODIES = {f'link{i}' for i in range(1, 8)} | {'hand_base_link', 'gripper_body_link', 'right_finger_tip',
                                                   'left_finger_tip', 'vision_frame_link', 'camera_link',
                                                   'camera_usb_plug_link'}
HEAD_BODIES = {'head_base', 'head_link1', 'head_link2', 'head_link3', 'head_camera_mount', 'femto_base_link'}
TIPS = {'right_finger_tip', 'left_finger_tip'}
NEAR_SURFACE_OK = TIPS | {'gripper_body_link', 'hand_base_link', 'camera_link', 'camera_usb_plug_link', 'vision_frame_link', 'obj'}
M_BODY, M_ENV = 0.05, 0.03
POSES = {   # kcare ARM_CONFIGS (degrees)
    'give': {'left': [96.85, 18.01, -49.85, 14.47, -33.29, 53.71, -14.79],
             'right': [-78.91, 24.26, 45.87, 24.81, -318.36, 52.09, 20.36], 'front': [90, 0, 20, 12, -40, 97, -10]},
    'fold': {'front': [90, 0, 20, 12, -40, 97, -10], 'left': [82.75, -26.26, -32.95, 17.84, -41.13, 104.73, -14.83],
             'right': [-76.26, -35.63, 31.38, 21.48, -313.32, 110.09, 23.49]},
    'ready': {'front': [0, 15, 0, 15, -180, 90, 0], 'left': [-90, 15, 0, 15, -180, 90, 0], 'right': [90, 15, 0, 15, -180, 90, 0]},
    'approach_lying': {'front': [0.0, -4.91, 0.0, 48.86, -180.0, -53.77, 0.0], 'right': [89.6, -26.8, 0.3, 28.4, 0.9, 55, -179.9],
                       'left': [86, 23.3, -182.5, 37, -0.2, 60.4, -185.5]},
    'pre_pick_floor': {'front': [0.0, 64.64, 0.0, 79.37, -180.0, -14.72, 0.0], 'left': [-33.54, 106.3, -70.37, 91.75, -253.32, 70.67, -37.49],
                       'right': [33.54, 106.3, 70.37, 91.75, -106.68, 70.67, 37.49]},
    'pre_pick_high': {'front': [0.0, -13.39, 0.0, 6.35, -180.0, 70.27, 0.0], 'left': [-44.63, 10.06, -40.33, 31.39, -174.03, 65.8, 4.06],
                      'right': [45.28, 9.96, 39.7, 31.39, -185.93, 65.81, -3.93]},
}
GRASP_Z = {'top': 0.04, 'side': 0.08}          # TCP above the support surface


def start_poses(mode, zo):
    if mode == 'front':
        return ['pre_pick_floor'] if zo < 0.5 else (['approach_lying'] if zo < 1.0 else ['pre_pick_high'])
    return ['give'] + (['pre_pick_floor'] if zo < 0.5 else (['pre_pick_high'] if zo >= 1.0 else []))


def lift_order(zo, mode):
    """High: lift first (raise the whole arm, keep the elbow low); floor: lowest first; else nearest home."""
    L = list(LIFT_J[::2]) + [LIFT_J[-1]]
    if zo >= 1.0:
        return sorted(set(L), reverse=True)
    if zo < 0.1:
        return sorted(set(L))
    home = {'front': 0.17, 'left': 0.37, 'right': 0.37}[mode]
    return sorted(set(L), key=lambda x: abs(x - home))


ENV_NAMES = ('env_floor', 'env_table', 'env_shelf', 'env_top', 'env_back', 'env_sideL', 'env_sideR')


class Col:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        sp = mujoco.MjSpec.from_file(f'{SIM}/mjcf/kaair_slamtec_mobile.xml')
        G = mujoco.mjtGeom
        sp.worldbody.add_geom(name='env_floor', type=G.mjGEOM_PLANE, size=[5, 5, 0.1], pos=[0, 0, 0])
        for n in ENV_NAMES[1:]:
            sp.worldbody.add_geom(name=n, type=G.mjGEOM_BOX, size=[0.1, 0.1, 0.1], pos=[0, 0, -10])
        tcp = sp.body('tool_tcp_link')
        tcp.add_geom(name='obj_box', type=G.mjGEOM_BOX, size=[0.03, 0.03, 0.04], pos=[0, 0, 0])
        self.m = m = sp.compile(); self.d = mujoco.MjData(m)
        # the env boxes are moved at run time: the world body's precompiled midphase BVH would keep them at their
        # compile-time pose (no contacts at all), so turn the midphase off
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.env = {n: m.geom(n).id for n in ENV_NAMES}
        self.obj = m.geom('obj_box').id
        self.cat, self.bname = {}, {}
        for g in range(m.ngeom):
            b = m.body(m.geom_bodyid[g]).name
            if g in self.env.values():
                self.cat[g], self.bname[g] = 'env', m.geom(g).name
            elif g == self.obj:
                self.cat[g], self.bname[g] = 'obj', 'obj'
            else:
                self.cat[g] = 'arm' if b in ARM_BODIES else ('head' if b in HEAD_BODIES else 'torso')
                self.bname[g] = b
        for g in range(m.ngeom):                       # clearance: contacts are reported below this distance
            if self.cat[g] in ('arm', 'obj') and m.geom_contype[g]:
                m.geom_margin[g] = M_BODY + 0.01
        self.holding(False)
        self.qa = {n: m.jnt_qposadr[m.joint(n).id] for n in ARM + ['lift_joint', 'head_joint1', 'head_joint2']}
        self.cam = m.camera('head_rgb').id
        fovy = math.radians(m.cam_fovy[self.cam]); self.fovy = fovy
        self.fovx = 2 * math.atan(math.tan(fovy / 2) * 1280 / 720)
        nx, ny = 32, 18
        xs = np.tan(np.linspace(-self.fovx / 2, self.fovx / 2, nx)); ys = np.tan(np.linspace(-fovy / 2, fovy / 2, ny))
        self.rays_cam = np.array([[x, y, -1.0] for y in ys for x in xs])
        self.rays_cam /= np.linalg.norm(self.rays_cam, axis=1, keepdims=True)
        self.arm_body_ids = [m.body(n).id for n in ('link3', 'link4', 'link5', 'link6', 'link7', 'hand_base_link', 'gripper_body_link')]
        self.benign = set()
        self.set_env(None)
        for pose in ('fold', 'give', 'ready', 'approach_lying'):
            for mode in ('front', 'left', 'right'):
                for lj in (0.13, 0.4, 0.67):
                    self.set_q(np.radians(POSES[pose][mode]), lj, mode, -40)
                    for k, pair, dist in self.raw_contacts():
                        if dist < -0.003 and k != 'env':
                            self.benign.add(pair)

    def holding(self, on):
        g = self.obj
        self.m.geom_contype[g] = 1 if on else 0; self.m.geom_conaffinity[g] = 1 if on else 0

    def set_env(self, spec):
        m = self.m
        for n, g in self.env.items():
            if n != 'env_floor':
                m.geom_pos[g] = [0, 0, -10]
        for n, (pos, size, yaw) in (spec or {}).items():
            g = self.env[n]
            # boxes stay axis-aligned (the working directions are +-x / +-y): swap the half-sizes instead of
            # rotating — a geom_quat written at run time was not applied to the world geoms' xmat
            if abs(math.sin(yaw)) > 0.5:
                size = [size[1], size[0], size[2]]
            m.geom_pos[g] = pos; m.geom_size[g] = size
            m.geom_quat[g] = [1, 0, 0, 0]
            m.geom_rbound[g] = float(np.linalg.norm(size))
            if hasattr(m, 'geom_aabb'):
                m.geom_aabb[g] = np.concatenate([[0, 0, 0], size])

    def set_q(self, qarm, lj, mode, tilt_deg):
        d = self.d
        for n, v in zip(ARM, qarm):
            d.qpos[self.qa[n]] = v
        d.qpos[self.qa['lift_joint']] = lj
        d.qpos[self.qa['head_joint1']] = HEAD_PAN[mode] + PAN_OFFSET
        d.qpos[self.qa['head_joint2']] = tilt_deg * DEG
        self.mj.mj_kinematics(self.m, d)

    def raw_contacts(self):
        m, d = self.m, self.d
        self.mj.mj_collision(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            k1, k2 = self.cat[g1], self.cat[g2]
            if k1 not in ('arm', 'obj') and k2 not in ('arm', 'obj'):
                continue
            if k1 in ('arm', 'obj') and k2 in ('arm', 'obj'):
                if 'obj' in (k1, k2):
                    continue                          # the held object touches the fingers
                kind = 'self'
            else:
                kind = k2 if k1 in ('arm', 'obj') else k1
            mine = self.bname[g1] if k1 in ('arm', 'obj') else self.bname[g2]
            pair = tuple(sorted((self.bname[g1], self.bname[g2])))
            out.append((kind, pair, float(c.dist), mine) if False else (kind, pair, float(c.dist)))
        return out

    def check(self, near_surface=False, allow_tips_env=False, waive_torso=False):
        """-> (penetration?, clearance violation?, min distances by kind)."""
        pen, viol, mind = False, False, {}
        m, d = self.m, self.d
        self.mj.mj_collision(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            k1, k2 = self.cat[g1], self.cat[g2]
            a1, a2 = k1 in ('arm', 'obj'), k2 in ('arm', 'obj')
            if not (a1 or a2):
                continue
            dist = float(c.dist)
            if a1 and a2:
                if 'obj' in (k1, k2):
                    continue
                pair = tuple(sorted((self.bname[g1], self.bname[g2])))
                if dist < -0.003 and pair not in self.benign:
                    pen = True
                continue
            kind = k2 if a1 else k1
            mine = self.bname[g1] if a1 else self.bname[g2]
            mind[kind] = min(mind.get(kind, 9), dist)
            if kind == 'env':
                if dist < -0.003 and not (allow_tips_env and mine in TIPS | {'obj'}):
                    pen = True
                if dist < M_ENV and not (near_surface and mine in NEAR_SURFACE_OK):
                    viol = True
            else:
                if dist < -0.003:
                    pair = tuple(sorted((self.bname[g1], self.bname[g2])))
                    if pair not in self.benign:
                        pen = True
                if dist < M_BODY and not (waive_torso and kind == 'torso'):
                    viol = True
        return pen, viol, mind

    def cam_pose(self):
        self.mj.mj_comPos(self.m, self.d); self.mj.mj_camlight(self.m, self.d)
        return self.d.cam_xpos[self.cam].copy(), self.d.cam_xmat[self.cam].reshape(3, 3).copy()

    def head_flags(self):
        """(elbow or wrist above the head camera, an arm link in the frustum within 0.35 m of the lens)."""
        p0, R = self.cam_pose()
        d = self.d
        above = any(d.xpos[b][2] > p0[2] for b in self.arm_body_ids[1:4])      # link4 (elbow) .. link6 (wrist)
        lens = False
        for b in self.arm_body_ids:
            pc = R.T @ (d.xpos[b] - p0); z = -pc[2]
            if 0.0 < z < 0.35 and abs(math.atan2(pc[0], z)) < self.fovx / 2 and abs(math.atan2(pc[1], z)) < self.fovy / 2:
                lens = True
        return above, lens

    def view(self, target):
        m, d = self.m, self.d
        best = None
        for tilt in range(-50, 11, 5):
            d.qpos[self.qa['head_joint2']] = tilt * DEG
            self.mj.mj_kinematics(m, d)
            p0, R = self.cam_pose(); pc = R.T @ (target - p0)
            z = -pc[2]
            if z < 0.1:
                continue
            ax, ay = abs(math.atan2(pc[0], z)), abs(math.atan2(pc[1], z))
            if ax < self.fovx / 2 and ay < self.fovy / 2:
                score = ax / self.fovx + ay / self.fovy
                if best is None or score < best[0]:
                    best = (score, tilt)
        if best is None:
            return None, False, None, None
        tilt = best[1]
        d.qpos[self.qa['head_joint2']] = tilt * DEG
        self.mj.mj_kinematics(m, d)
        p0, R = self.cam_pose()
        vec = (R @ self.rays_cam.T).T
        geomid = np.zeros(len(vec), np.int32); dist = np.zeros(len(vec))
        start = p0 + R @ np.array([0, 0, -0.06])
        self.mj.mj_multiRay(m, d, start, vec.flatten(), None, 1, -1, geomid, dist, None, len(vec), 5.0)
        arm_hits = sum(1 for g in geomid if g >= 0 and self.cat[int(g)] == 'arm')
        v = target - start; L = float(np.linalg.norm(v))
        gid = np.zeros(1, np.int32)
        dd = self.mj.mj_ray(m, d, start, v / L, None, 1, -1, gid)
        occl = dd >= 0 and dd < L - 0.03 and self.cat[int(gid[0])] == 'arm'
        return tilt, True, bool(occl), arm_hits / len(vec)


def wrap_interp(qa, qb, n):
    diff = np.array(qb) - np.array(qa)
    for k in (0, 2, 4, 6):
        diff[k] = (diff[k] + math.pi) % (2 * math.pi) - math.pi
    return [np.array(qa) + diff * t for t in np.linspace(0, 1, n)]


def env_for(mode, zo, d, a, dirs, board_gap=0.45):
    u, v = mode_frame(mode, dirs)
    yaw = math.atan2(u[1], u[0])
    half_u = 0.238 if mode == 'front' else 0.215
    e = max(d - 0.10, half_u + 0.03)
    if zo <= 0.001:
        return {}
    if zo < 1.0:
        c = u * (e + 0.3); c[2] = zo - 0.02
        return {'env_table': (c, [0.3, 0.6, 0.02], yaw)}
    W = 0.4                                               # shelf bay 0.8 m wide, centred on the object
    mid = u * (e + 0.15) + v * a
    top_z = zo + board_gap
    sup = mid.copy(); sup[2] = zo - 0.01
    top = mid.copy(); top[2] = top_z + 0.01
    back = u * (e + 0.31) + v * a; back[2] = (top_z + 0.02) / 2
    sl = mid + v * (W + 0.01); sl[2] = (top_z + 0.02) / 2
    sr = mid - v * (W + 0.01); sr[2] = (top_z + 0.02) / 2
    hz = (top_z + 0.02) / 2
    return {'env_shelf': (sup, [0.15, W, 0.01], yaw), 'env_top': (top, [0.15, W, 0.01], yaw),
            'env_back': (back, [0.01, W + 0.02, hz], yaw), 'env_sideL': (sl, [0.15, 0.01, hz], yaw),
            'env_sideR': (sr, [0.15, 0.01, hz], yaw)}


RETRACT = 0.20


def retract_q(qp, qg, grasp):
    """Joint angles 0.20 m further back along the approach line than the pre-grasp (outside a shelf / above a
    table), so start -> retract is a joint move in free space and retract -> pre -> grasp is a short straight
    approach. None when IK fails there."""
    q0 = {'lift_joint': 0.4}
    Tg = ARMC.fk(dict(q0, **dict(zip(ARM, qg)))); Tp = ARMC.fk(dict(q0, **dict(zip(ARM, qp))))
    dv = Tp[:3, 3] - Tg[:3, 3]; n = np.linalg.norm(dv)
    if n < 1e-6:
        return None
    tgt = Tp[:3, 3] + dv / n * RETRACT
    b = ik(q0, tgt, Tg[:3, 2], grasp == 'side', [np.array(qp)])
    return b[0] if b[1] < 0.01 and b[2] < 5 else None


def eval_path(col, q0, qp, qg, lj, mode, tilt, qr=None):
    """Whole path -> (penetration?, clearance violation?, elbow above head?, lens intrusion?, min dists).
    With qr: start -> retract (joint move) -> pre -> grasp, and back the same way with the object."""
    pen = viol = above = lens = False; mind = {}
    if qr is not None:
        stages = [(wrap_interp(q0, qr, 24), False, False, False),
                  (wrap_interp(qr, qp, 8)[1:], False, False, False),
                  (wrap_interp(qp, qg, 10)[1:], False, True, False),
                  (wrap_interp(qg, qp, 10)[1:], True, True, True),
                  (wrap_interp(qp, qr, 8)[1:], True, False, False),
                  (wrap_interp(qr, q0, 24)[1:], True, False, False)]
    else:
        stages = [(wrap_interp(q0, qp, 24), False, False, False),            # start -> pre (empty hand)
                  (wrap_interp(qp, qg, 10)[1:], False, True, False),         # approach
                  (wrap_interp(qg, qp, 10)[1:], True, True, True),           # lift off with the object
                  (wrap_interp(qp, q0, 24)[1:], True, False, False)]         # retreat with the object
    for si, (samples, hold, near, tips_ok) in enumerate(stages):
        col.holding(hold)
        for k, q in enumerate(samples):
            col.set_q(q, lj, mode, tilt)
            # the kcare stow / start pose itself sits ~4.7 cm from the torso ('give'): the torso margin is waived on the
            # first / last 5 samples next to it (penetration is still checked; the head margin never is)
            wt = (si == 0 and k < 5) or (si == len(stages) - 1 and k >= len(samples) - 5)
            p, vio, md = col.check(near_surface=near or (hold and k < 4), allow_tips_env=tips_ok or (near and k == len(samples) - 1),
                                   waive_torso=wt)
            pen |= p; viol |= vio
            for kk, vv in md.items():
                mind[kk] = min(mind.get(kk, 9), vv)
            if k % 3 == 0:
                a_, l_ = col.head_flags(); above |= a_; lens |= l_
            if pen:
                col.holding(False)
                return pen, viol, above, lens, mind
    col.holding(False)
    return pen, viol, above, lens, mind


def eval_job(args):
    mode, g, zo, gap, raw_mg, dirs = args
    if 'COL' not in globals():
        setup()
        globals()['COL'] = Col()
    col = COL
    u, v = mode_frame(mode, dirs)
    zg = zo + GRASP_Z[g]
    hset = {round(x, 3) for x in H_GRID}
    cells, reasons = {}, {}
    for dd in D_GRID:
        for aa in A_GRID:
            col.set_env(env_for(mode, zo, float(dd), float(aa), dirs, gap))
            target = u * dd + v * aa; target[2] = zg
            c = dict(ik=False, free=None, safe=None, safe_strict=None)
            why = 'ik'
            for lj in lift_order(zo, mode):
                h = round(round((zg - lj) / 0.05) * 0.05, 3)
                if h not in hset:
                    continue
                r = raw_mg.get(h, {}).get(f'{dd:.2f},{aa:.2f}')
                if not isinstance(r, list):
                    continue
                c['ik'] = True
                qg, qp = np.array(r[0]), np.array(r[1])
                qr = retract_q(qp, qg, g)
                for sp_name in start_poses(mode, zo):
                    q0 = np.radians(POSES[sp_name][mode])
                    col.set_q(qp, lj, mode, -40); vp = col.view(target)
                    tilt = vp[0] if vp[0] is not None else -40
                    col.set_q(q0, lj, mode, tilt)
                    p, vio, _ = col.check(waive_torso=True)
                    if p:
                        why = 'start'; continue
                    pen, viol, above, lens, mind = eval_path(col, q0, qp, qg, lj, mode, tilt, qr)
                    if pen and qr is not None:          # the direct joint move may still be clear
                        pen, viol, above, lens, mind = eval_path(col, q0, qp, qg, lj, mode, tilt)
                    if pen:
                        why = 'collision'; continue
                    col.set_q(q0, lj, mode, -40); vs = col.view(target)
                    sol = dict(lift=float(lj), start=sp_name, pre=vp, st=vs, viol=bool(viol), above=bool(above),
                               lens=bool(lens), mind={k: round(v_, 3) for k, v_ in mind.items()})
                    if c['free'] is None:
                        c['free'] = sol
                    if not viol and c['safe'] is None:
                        c['safe'] = sol
                    if not viol and not above and not lens:
                        c['safe_strict'] = sol
                        break
                    why = 'clearance' if viol else 'head'
                if c['safe_strict']:
                    break
            cells[f'{dd:.2f},{aa:.2f}'] = c
            if not c['safe_strict']:
                reasons[why] = reasons.get(why, 0) + 1
    return mode, g, zo, gap, cells, reasons


def collide_main(a):
    import pickle
    P = pickle.load(open(f'{a.out}/raw_ik.pkl', 'rb'))
    raw, dirs = P['raw'], P['dirs']
    jobs = []
    for m in ('front', 'left', 'right'):
        for g in ('top', 'side'):
            raw_mg = {h: out for (mm, gg, h), out in raw.items() if mm == m and gg == g}
            for zo in Z_OBJ:
                for gap in ((0.45, 0.35) if zo >= 1.0 else (None,)):
                    jobs.append((m, g, zo, gap, raw_mg, dirs))
    jobs.sort(key=lambda j: -j[2])                       # slow high-shelf jobs first
    t0 = time.time(); results = {}
    with Pool(a.procs) as pool:
        for k, (m, g, zo, gap, cells, reasons) in enumerate(pool.imap_unordered(eval_job, jobs)):
            results[(m, g, zo, gap)] = (cells, reasons)
            print(f'{k + 1}/{len(jobs)} {m} {g} {zo} {gap} {time.time() - t0:.0f}s', flush=True)
    col = Col()
    summary = {'assumptions': {
        'grasp_z': GRASP_Z, 'pre_grasp_m': PRE, 'margins': {'body_head_m': M_BODY, 'furniture_m': M_ENV},
        'start_poses': {m: {str(z): start_poses(m, z) for z in Z_OBJ} for m in ('front', 'left', 'right')},
        'benign_contact_pairs': sorted(map(list, col.benign)),
        'env': 'floor plane; z<1: table slab, front edge 0.10 m before the object, 0.6 m deep; z>=1: shelf bay 0.3 m deep, 0.8 m wide, side panels, back board, board 0.45 (or 0.35) m above',
        'path': 'start -> retract point 0.20 m behind the pre-grasp (joint move, 24) -> pre-grasp (8) -> grasp (10) and back the same way holding a 6x6x8 cm object; if that collides, the direct joint move start -> pre is tried; penetration >3 mm = collision; near the support the finger tips / hand / wrist camera / object are exempt from the furniture margin; torso margin waived on the 5 samples next to the start pose (kcare give sits 4.7 cm from the torso)',
        'lift_order': 'high: highest lift first; floor: lowest first; else nearest the mode home'}, 'cells': {}}
    for (m, g, zo, gap), (cells, reasons) in results.items():
        n = len(cells)
        def share(key):
            return sum(1 for c in cells.values() if c[key]) / n
        def shift_share(key):
            ok = 0
            for k_, c in cells.items():
                if c[key]:
                    ok += 1; continue
                dd, aa = map(float, k_.split(','))
                for s_ in (0.1, -0.1, 0.2, -0.2):
                    k2 = f'{round(dd - s_, 2):.2f},{aa:.2f}' if m == 'front' else f'{dd:.2f},{round(aa - s_, 2):.2f}'
                    if cells.get(k2, {}).get(key):
                        ok += 1; break
            return ok / n
        strict = [c['safe_strict'] for c in cells.values() if c['safe_strict']]
        anyfree = [c['free'] for c in cells.values() if c['free']]
        key = f'{m}|{g}|{zo}' + (f'|shelf{gap}' if gap else '')
        summary['cells'][key] = dict(
            ik=share('ik'), free=share('free'), safe=share('safe'), safe_strict=share('safe_strict'),
            safe_strict_shift=shift_share('safe_strict'),
            by_d={f'{dd:.1f}': sum(1 for aa in A_GRID if cells[f'{dd:.2f},{aa:.2f}']['safe_strict']) / len(A_GRID) for dd in D_GRID},
            visible_pre=(sum(1 for s_ in strict if s_['pre'][1] and not s_['pre'][2]) / len(strict)) if strict else None,
            visible_start=(sum(1 for s_ in strict if s_['st'][1] and not s_['st'][2]) / len(strict)) if strict else None,
            arm_cover_pre=float(np.mean([s_['pre'][3] for s_ in strict if s_['pre'][3] is not None])) if strict and any(s_['pre'][3] is not None for s_ in strict) else None,
            arm_cover_start=float(np.mean([s_['st'][3] for s_ in strict if s_['st'][3] is not None])) if strict and any(s_['st'][3] is not None for s_ in strict) else None,
            free_above_head=(sum(1 for s_ in anyfree if s_['above']) / len(anyfree)) if anyfree else None,
            free_lens=(sum(1 for s_ in anyfree if s_['lens']) / len(anyfree)) if anyfree else None,
            lift_kcare=sorted({round(s_['lift'] + MOBILE_H, 2) for s_ in strict}),
            d_safe=sorted({float(k_.split(',')[0]) for k_, c in cells.items() if c['safe_strict']}),
            starts={s_: sum(1 for x in strict if x['start'] == s_) for s_ in start_poses(m, zo)},
            fail=reasons)
    json.dump(summary, open(f'{a.out}/summary_collide.json', 'w'), indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for m in ('front', 'left', 'right'):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        for ax, g in zip(axs, ('top', 'side')):
            M = np.array([[summary['cells'][f'{m}|{g}|{zo}' + ('|shelf0.45' if zo >= 1.0 else '')]['by_d'][f'{dd:.1f}'] for dd in D_GRID] for zo in Z_OBJ])
            im = ax.imshow(M, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1,
                           extent=[D_GRID[0] - 0.05, D_GRID[-1] + 0.05, -0.5, len(Z_OBJ) - 0.5])
            ax.set_yticks(range(len(Z_OBJ))); ax.set_yticklabels([f'{z:.2f}' for z in Z_OBJ])
            ax.set_xlabel('distance from base centre along the working side d [m]'); ax.set_ylabel('object support height [m]')
            ax.set_title(f'{m}, {g}: safe (path, margins, head) - shelf for z>=1.1', fontsize=9)
            for i in range(len(Z_OBJ)):
                for j, dd in enumerate(D_GRID):
                    if M[i, j] > 0:
                        ax.text(dd, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=6, color='black' if M[i, j] > 0.6 else 'white')
        fig.colorbar(im, ax=axs, shrink=0.8)
        fig.savefig(f'{a.out}/reach_safe_{m}.png', dpi=110); plt.close(fig)
    print('collide done', time.time() - t0, flush=True)


def _entry():
    if '--collide' in sys.argv:
        ap = argparse.ArgumentParser(); ap.add_argument('--collide', action='store_true')
        ap.add_argument('--procs', type=int, default=16); ap.add_argument('--out', default=f'{SIM}/out/reach')
        collide_main(ap.parse_args())
    else:
        main()


if __name__ == '__main__':
    _entry()
