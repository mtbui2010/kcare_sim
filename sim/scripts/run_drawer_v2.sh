#!/bin/bash
# Drawer VLA v2 (2026-10-09): open (head -65, 30 % retry demos) + close data, merged, fine-tuned from officehouse 20261007b
# step 10000 (84 % open in the house), evaluated on open and close separately.
#   docker exec -d kaair_sim_humble bash -c "cd /sim && bash scripts/run_drawer_v2.sh > out/drawer_v2.log 2>&1"
set -e
cd /sim
D=/nas/datasets
BASE=/nas/checkpoints/smolvla_open_drawer_officehouse_20261007b/checkpoints/010000/pretrained_model
RUN=/nas/checkpoints/smolvla_drawer_openclose_v2_20261009
R="--rand-config config/randomization_house.yaml --camera-model"
[ -f $D/kaair_sim_open_drawer_house_v2_500/meta/info.json ] || .venv/bin/python scripts/gen_dataset.py --n 500 --workers 12 --gpu-ids 0,3 --task config/task_open_drawer_vla2.yaml $R --repo-id kaair_sim_open_drawer_house_v2_500 --seed 5000
[ -f $D/kaair_sim_close_drawer_house_v2_300/meta/info.json ] || .venv/bin/python scripts/gen_dataset.py --n 300 --workers 12 --gpu-ids 0,3 --task config/task_close_drawer_vla2.yaml $R --repo-id kaair_sim_close_drawer_house_v2_300 --seed 6000
if [ ! -f $D/kaair_sim_drawer_openclose_v2_800/meta/info.json ]; then
  .venv/bin/python -W ignore - <<EOF
from lerobot.datasets.aggregate import aggregate_datasets
aggregate_datasets(repo_ids=['kaair_sim_open_drawer_house_v2_500', 'kaair_sim_close_drawer_house_v2_300'],
                   aggr_repo_id='kaair_sim_drawer_openclose_v2_800',
                   roots=['$D/kaair_sim_open_drawer_house_v2_500', '$D/kaair_sim_close_drawer_house_v2_300'],
                   aggr_root='$D/kaair_sim_drawer_openclose_v2_800')
EOF
fi
echo "[$(date +%T)] training -> $RUN"
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/lerobot-train --policy.path=$BASE --policy.push_to_hub=false \
    --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --dataset.repo_id=kaair_sim_drawer_openclose_v2_800 --dataset.root=$D/kaair_sim_drawer_openclose_v2_800 \
    --dataset.video_backend=pyav --output_dir=$RUN --job_name=smolvla_drawer_openclose_v2 \
    --batch_size=64 --steps=12000 --save_freq=3000 --log_freq=100 --num_workers=12 --wandb.enable=false \
    > out/train_drawer_v2.log 2>&1 &
sleep 60
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 .venv/bin/python -W ignore scripts/watch_eval.py --run $RUN --n 50 --tag open_v2 \
    --eval-args "--task task_open_drawer_vla2.yaml --rand-config randomization_house.yaml --camera-model" > out/watch_drawer_v2_open.log 2>&1 &
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 .venv/bin/python -W ignore scripts/watch_eval.py --run $RUN --n 50 --tag close_v2 \
    --eval-args "--task task_close_drawer_vla2.yaml --rand-config randomization_house.yaml --camera-model --success-m 0.0" > out/watch_drawer_v2_close.log 2>&1
wait
echo "[$(date +%T)] done"
