"""Where does kcare's drawer-ready pose put the TCP and where does the head look?"""
import sys, numpy as np
sys.path.insert(0, '/sim/scripts'); import check_sim as cs
ref = cs.UrdfFK('/sim/urdf/raw/kaair_slamtec.urdf')
AD = {'right': [-6.211779573691565, 81.90505354547581, -15.779663096075398, 80.38103343597086, -277.14880451766084, 105.36503896698794, -3.078588880403007],
      'left': [3.834008869823475, 82.74111651120957, 17.51407523013154, 81.91691758406412, -81.43637962500787, 107.12427713112979, 2.9979493191843094]}
RZ = {'left': -70, 'right': 70}
lift = float(np.clip(0.5, 0.56, 1.1) - 0.43)
for mode in ('left', 'right'):
    q = {f'joint{i+1}': np.deg2rad(a) for i, a in enumerate(AD[mode])}
    q.update(lift_joint=lift, head_joint1=np.deg2rad(RZ[mode]), head_joint2=np.deg2rad(-30), virtual_gripper_joint=0.1)
    T = ref.fk(q)
    tcp = T['tool_tcp_link']; cam = T['femto_color_optical_frame']
    print(f'{mode}: lift_joint={lift:.3f}  TCP pos={np.round(tcp[:3,3],3)} tool z-axis={np.round(tcp[:3,2],2)} finger(y)-axis={np.round(tcp[:3,1],2)}')
    print(f'      head cam pos={np.round(cam[:3,3],3)} looks along {np.round(cam[:3,2],2)}   wrist cam looks along {np.round(T["camera_color_optical_frame"][:3,2],2)}')
