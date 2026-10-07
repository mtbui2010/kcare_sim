#!/usr/bin/env python3
"""Copy lerobot/smolvla_base to the NAS with input features that match kaair.

    /sim/.venv/bin/python scripts/make_kaair_base.py
      -> /nas/checkpoints/_bases/smolvla_base_kaair   (use as --policy.path)

Why: lerobot 0.4.4 make_policy() fills cfg.input_features from the dataset
only when the pretrained config has none. smolvla_base ships its own (SO-100:
6-D state, camera1..3), so a run started from it kept `observation.state`
[6] and camera3 in its config, and with --policy.empty_cameras=1 fed the model
2 real + 2 black images. The model still saw the full 8-D state (SmolVLA pads
state to max_state_dim=32 and the normalizer uses the dataset's 8-D stats),
but the config was wrong for deployment and the image layout differed from
pretraining (3 slots). Here the declared inputs are exactly what the data
provides: 8-D state, camera1 = head, camera2 = wrist (848x480, renamed at train
time with --rename_map), and empty_cameras=1 keeps 3 image slots. Weights are
untouched: the state projection works on the 32-D padded vector.
"""
import glob, json, os, shutil

SRC = glob.glob('/hf_cache/models--lerobot--smolvla_base/snapshots/*/')[0]
DST = os.path.join(os.environ.get('KAAIR_CKPT', '/nas/checkpoints'), '_bases', 'smolvla_base_kaair')
H, W = 480, 848

os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f.endswith(('.json', '.safetensors')):
        shutil.copyfile(os.path.realpath(os.path.join(SRC, f)), os.path.join(DST, f))   # real files, not HF symlinks

cfg_path = os.path.join(DST, 'config.json')
cfg = json.load(open(cfg_path))
old = {k: v['shape'] for k, v in cfg['input_features'].items()}
cfg['input_features'] = {
    'observation.state': {'type': 'STATE', 'shape': [8]},
    'observation.images.camera1': {'type': 'VISUAL', 'shape': [3, H, W]},
    'observation.images.camera2': {'type': 'VISUAL', 'shape': [3, H, W]},
}
cfg['output_features'] = {'action': {'type': 'ACTION', 'shape': [8]}}
cfg['empty_cameras'] = 1
json.dump(cfg, open(cfg_path, 'w'), indent=4)
print('source   ', SRC)
print('old input', old)
print('new input', {k: v['shape'] for k, v in cfg['input_features'].items()}, 'empty_cameras', cfg['empty_cameras'])
print('->', DST)
