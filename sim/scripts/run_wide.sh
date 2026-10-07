#!/bin/bash
# Wide-placement experiment: cabinet placement ±8 cm (config/randomization_wide.yaml)
# -> 500-episode dataset -> SmolVLA 10k steps from smolvla_base_kaair.
# Everything on GPU $GPU (pick a free one: nvidia-smi); data + checkpoints on the NAS.
# Evaluation runs separately (scripts/watch_eval.py) on the SAME 50 scenes as every other run.
#   docker exec -d kaair_sim_humble bash -c "cd /sim && GPU=0 bash scripts/run_wide.sh > /sim/out/run_wide.log 2>&1"
set -e
GPU=${GPU:-0}
REPO=kaair_sim_open_drawer_500_wide
RUN=/nas/checkpoints/smolvla_open_drawer_500wide_kaairbase_$(date +%Y%m%d)
cd /sim
echo "[$(date +%T)] generating $REPO on GPU $GPU"
[ -d /nas/datasets/$REPO ] || .venv/bin/python -W ignore scripts/gen_dataset.py --n 500 --workers 8 --seed 3 \
    --gpu-ids $GPU --rand-config config/randomization_wide.yaml --repo-id $REPO
echo "[$(date +%T)] training -> $RUN"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/lerobot-train \
    --policy.path=/nas/checkpoints/_bases/smolvla_base_kaair --policy.push_to_hub=false \
    --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --dataset.repo_id=$REPO --dataset.root=/nas/datasets/$REPO --dataset.video_backend=pyav \
    --output_dir=$RUN --job_name=smolvla_open_drawer_500wide \
    --batch_size=64 --steps=10000 --save_freq=2500 --log_freq=100 --num_workers=12 --wandb.enable=false
echo "[$(date +%T)] done"
