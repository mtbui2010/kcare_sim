"""Head camera at the drawer ready pose for several tilts, side by side."""
import math, sys, mujoco, yaml
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/sim/scripts'); import check_sim as cs, drawer_ready as dr
cfg = yaml.safe_load(open('/sim/config/mjcf.yaml')); task = yaml.safe_load(open('/sim/config/task_open_drawer.yaml'))
m = mujoco.MjModel.from_xml_path('/sim/mjcf/scene_open_drawer_left.xml'); d = mujoco.MjData(m)
r = mujoco.Renderer(m, 720, 1280); font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 40)
tiles = []
for tilt in (-30, -40, -50):
    q = dr.ready_q(task, 'left', cfg['home']); q['head_joint2'] = math.radians(tilt)
    cs.set_q(m, d, q, cfg); mujoco.mj_forward(m, d); r.update_scene(d, 'head_rgb')
    im = Image.fromarray(r.render()); ImageDraw.Draw(im).text((20, 15), f'ry = {tilt} deg', fill=(255, 255, 0), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    tiles.append(im.resize((640, 360)))
out = Image.new('RGB', (1920, 360))
for i, t in enumerate(tiles): out.paste(t, (i * 640, 0))
out.save('/sim/out/open_drawer/head_tilt_compare.png'); print('ok')
