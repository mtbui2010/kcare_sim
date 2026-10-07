#!/usr/bin/env python3
"""Grab one frame of each sim camera topic -> /sim/out/house/snap_{head,wrist,overview}.jpg (+ a mosaic).

    source /workspace/ros2/ros2_ws/install/setup.bash; python3 scripts/house_snapshot.py
"""
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

OUT = '/sim/out/house'
TOPICS = {'head': ('/femto/color/image_raw', Image), 'wrist': ('/hand/camera/color/image_raw/compressed', CompressedImage),
          'overview': ('/sim/overview/image_raw/compressed', CompressedImage)}


def main():
    rclpy.init()
    n = Node('house_snapshot')
    got = {}

    def cb(name):
        def f(msg):
            if name in got:
                return
            if isinstance(msg, Image):
                im = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
                got[name] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            else:
                got[name] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        return f
    for k, (t, typ) in TOPICS.items():
        n.create_subscription(typ, t, cb(k), 1)
    t0 = time.time()
    while len(got) < len(TOPICS) and time.time() - t0 < 10:
        rclpy.spin_once(n, timeout_sec=0.2)
    for k, im in got.items():
        cv2.imwrite(f'{OUT}/snap_{k}.jpg', im)
    if len(got) == 3:
        h = cv2.resize(got['head'], (848, 480))
        mos = np.concatenate([np.concatenate([got['overview'], h], 1),
                              np.concatenate([got['wrist'], np.zeros_like(h)], 1)], 0)
        for (x, y), s in (((10, 30), 'overview (not a robot sensor)'), ((858, 30), 'head camera (Femto)'), ((10, 510), 'wrist camera (D405)')):
            cv2.putText(mos, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(mos, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(f'{OUT}/snap.jpg', mos[:, :, :] if mos.shape[0] else mos)
    print('got', sorted(got), '->', OUT)
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
