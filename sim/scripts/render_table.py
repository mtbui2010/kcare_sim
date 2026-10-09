"""Offline head-camera-like view of the test table for scene(s): render_table.py <scene>... -> out/table_<scene>.jpg"""
import os, subprocess, sys
import cv2, mujoco, numpy as np
SIM = os.environ.get('SIM_ROOT', '/sim')
sys.path.insert(0, f'{SIM}/scripts')
import house_scene as hs
tiles = []
for sc in sys.argv[1:]:
    subprocess.run([f'{SIM}/.venv/bin/python', f'{SIM}/scripts/house_scene.py', '--scene', sc], check=True, capture_output=True)
    m = mujoco.MjModel.from_xml_path(f'{SIM}/mjcf/house.xml'); d = mujoco.MjData(m)
    for jn, v in (('base_x', -1.5), ('base_y', 0.0)):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn); d.qpos[m.jnt_qposadr[j]] = v
    for _ in range(600): mujoco.mj_step(m, d)
    tx, ty = hs.TEST_TABLE_AT
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [tx, ty + 0.06, hs.TEST_TABLE[2]]; cam.distance = 0.72; cam.azimuth = 90; cam.elevation = -68
    m.vis.global_.fovy = 51
    r = mujoco.Renderer(m, 480, 848)
    hide = {g for g in range(m.ngeom) if m.geom(g).name == 'ceiling' or m.geom(g).name.startswith('downlight')}
    r.update_scene(d, cam)
    for k in range(r.scene.ngeom):
        if r.scene.geoms[k].objtype == mujoco.mjtObj.mjOBJ_GEOM and r.scene.geoms[k].objid in hide:
            r.scene.geoms[k].rgba[3] = 0
    img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
    cv2.putText(img, sc, (10, 30), 0, 0.9, (0, 0, 255), 2)
    tiles.append(img); r.close()
out = np.vstack([np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.zeros_like(tiles[i])]) for i in range(0, len(tiles), 2)])
cv2.imwrite(f'{SIM}/out/table_photos.jpg', out); print(out.shape)
