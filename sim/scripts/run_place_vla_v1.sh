#!/bin/bash
# PLACE VLA v1 (2026-10-11): train SmolVLA on kaair_sim_place_house_v1_800 (20-D state = pick's 17 + place target
# x/y/z, 10-D action) on GPU 3, num_workers 6 (the pick v1 run trains on GPU 0 at the same time; A6000 is shared).
# Eval comes later (needs an eval_place_policy.py). Runs on the A6000 host:
#   nohup bash scripts/run_place_vla_v1.sh > out/place_vla_v1.log 2>&1 &
set -u
C=kaair_sim_humble
REPO=kaair_sim_place_house_v1_800
RUN=/nas/checkpoints/smolvla_place_house_v1_${DATE:-$(date +%Y%m%d)}
inc() { docker exec $C bash -c "$1"; }
inc "cd /sim && .venv/bin/python - <<'PY'
import json, os, shutil
src, dst = '/nas/checkpoints/_bases/smolvla_base_kaair', '/nas/checkpoints/_bases/smolvla_base_kaair_place20'
if not os.path.exists(f'{dst}/config.json'):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    c = json.load(open(f'{dst}/config.json'))
    c['input_features']['observation.state']['shape'] = [20]
    c['output_features']['action']['shape'] = [10]
    json.dump(c, open(f'{dst}/config.json', 'w'), indent=4)
print('base', dst)
PY"
echo "[$(date +%T)] training -> $RUN"
inc "cd /sim && CUDA_VISIBLE_DEVICES=3 .venv/bin/lerobot-train \
    --policy.path=/nas/checkpoints/_bases/smolvla_base_kaair_place20 --policy.push_to_hub=false \
    --rename_map='{\"observation.images.head\":\"observation.images.camera1\",\"observation.images.wrist\":\"observation.images.camera2\"}' \
    --dataset.repo_id=$REPO --dataset.root=/nas/datasets/$REPO --dataset.video_backend=pyav \
    --output_dir=$RUN --job_name=smolvla_place_house_v1 \
    --batch_size=64 --steps=${STEPS:-15000} --save_freq=2000 --log_freq=100 --num_workers=6 --wandb.enable=false \
    > out/train_place_v1.log 2>&1"
echo "[$(date +%T)] training ended"; inc "tail -c 400 /sim/out/train_place_v1.log"
