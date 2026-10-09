"""Offline views of scenes from where a person would look: render_spots.py out.jpg scene:x,y,z:lx,ly,lz ..."""
import os, subprocess, sys
import cv2, mujoco, numpy as np
SIM = os.environ.get('SIM_ROOT', '/sim')
tiles = []
for arg in sys.argv[2:]:
    sc, eye, at = arg.split(':')
    eye = np.array([float(v) for v in eye.split(',')]); at = np.array([float(v) for v in at.split(',')])
    subprocess.run([f'{SIM}/.venv/bin/python', f'{SIM}/scripts/house_scene.py', '--scene', sc], check=True, capture_output=True)
    m = mujoco.MjModel.from_xml_path(f'{SIM}/mjcf/house.xml'); d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'base_x'); d.qpos[m.jnt_qposadr[j]] = -3.0
    for _ in range(500): mujoco.mj_step(m, d)
    v = at - eye; cam = mujoco.MjvCamera(); cam.lookat[:] = at; cam.distance = float(np.linalg.norm(v))
    cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0]))); cam.elevation = float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))
    m.vis.global_.fovy = 60
    r = mujoco.Renderer(m, 360, 640)
    hide = {g for g in range(m.ngeom) if m.geom(g).name == 'ceiling' or m.geom(g).name.startswith('downlight')}
    r.update_scene(d, cam)
    for k in range(r.scene.ngeom):
        if r.scene.geoms[k].objtype == mujoco.mjtObj.mjOBJ_GEOM and r.scene.geoms[k].objid in hide:
            r.scene.geoms[k].rgba[3] = 0
    img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR); cv2.putText(img, sc, (8, 26), 0, 0.8, (0, 0, 255), 2)
    tiles.append(img); r.close()
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
