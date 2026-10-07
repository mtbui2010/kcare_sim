"""Extent of each camera's housing mesh along its optical axis (+z of the optical frame)."""
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('/sim/mjcf/kaair_slamtec.xml'); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
for cam, housing in [('wrist_rgb', 'camera_link'), ('head_rgb', 'femto_base_link')]:
    c = m.camera(cam).id; b = m.cam_bodyid[c]
    R = d.xmat[b].reshape(3, 3); p = d.xpos[b]           # optical frame of the body the camera sits on
    zs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != m.body(housing).id or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
        loc = (R.T @ (w - p).T).T
        near = loc[np.hypot(loc[:, 0], loc[:, 1]) < 0.01]    # vertices around the optical axis
        zs.append(loc[:, 2].max()); print(cam, housing, 'mesh z range along optical axis', np.round([loc[:, 2].min(), loc[:, 2].max()], 4),
                                          ' near-axis max z', np.round(near[:, 2].max(), 4) if len(near) else None)
