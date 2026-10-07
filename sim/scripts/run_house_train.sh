#!/bin/bash
# Open drawer in the KETI apartment: co-train SmolVLA on office + house data, evaluate in the house.
#
#   docker exec -d kaair_sim_humble bash -c "cd /sim && bash scripts/run_house_train.sh > out/house_train.log 2>&1"
#
# 1. waits for kaair_sim_open_drawer_house500 (gen_dataset.py, task_open_drawer_house.yaml: house_scene (drawio layout, A_home cabinet in the kitchen) around the
#    A_real cabinet, camera model, the same randomization as real500)
# 2. merges it with the office set kaair_sim_open_drawer_real500 -> kaair_sim_open_drawer_officehouse1000
# 3. baseline: the office-only checkpoint (real500, step 20000) on 50 house scenes      (GPU $EVAL_GPU)
# 4. fine-tunes that checkpoint on the merged set, 10k steps, save every 2500            (GPU $GPU)
# 5. watch_eval.py evaluates every new checkpoint on the SAME 50 house scenes (tag "house")
set -e
GPU=${GPU:-0}
EVAL_GPU=${EVAL_GPU:-3}
D=/nas/datasets
BASE=/nas/checkpoints/smolvla_open_drawer_real500_multidrawer_20260928
RUN=/nas/checkpoints/smolvla_open_drawer_officehouse_20261007b
EVAL_ARGS="--task task_open_drawer_house.yaml --rand-config randomization_house.yaml --camera-model"
cd /sim
echo "[$(date +%T)] waiting for $D/kaair_sim_open_drawer_house500"
until [ -f $D/kaair_sim_open_drawer_house500/meta/info.json ] && ! pgrep -f "[g]en_dataset.py.*house500" >/dev/null; do sleep 30; done

if [ ! -f $D/kaair_sim_open_drawer_officehouse1000/meta/info.json ]; then
  echo "[$(date +%T)] merging office + house"
  .venv/bin/python -W ignore - <<EOF
from lerobot.datasets.aggregate import aggregate_datasets
aggregate_datasets(repo_ids=['kaair_sim_open_drawer_real500', 'kaair_sim_open_drawer_house500'],
                   aggr_repo_id='kaair_sim_open_drawer_officehouse1000',
                   roots=['$D/kaair_sim_open_drawer_real500', '$D/kaair_sim_open_drawer_house500'],
                   aggr_root='$D/kaair_sim_open_drawer_officehouse1000')
EOF
fi

echo "[$(date +%T)] baseline: office-only checkpoint in the house (GPU $EVAL_GPU)"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$EVAL_GPU CUDA_VISIBLE_DEVICES=$EVAL_GPU nohup .venv/bin/python -W ignore \
    scripts/eval_policy.py --ckpt $BASE/checkpoints/020000/pretrained_model --n 50 --videos 3 --tag house $EVAL_ARGS \
    > out/eval_house_baseline.log 2>&1 &

echo "[$(date +%T)] training -> $RUN (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/lerobot-train \
    --policy.path=$BASE/checkpoints/020000/pretrained_model --policy.push_to_hub=false \
    --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --dataset.repo_id=kaair_sim_open_drawer_officehouse1000 --dataset.root=$D/kaair_sim_open_drawer_officehouse1000 \
    --dataset.video_backend=pyav --output_dir=$RUN --job_name=smolvla_open_drawer_officehouse \
    --batch_size=64 --steps=10000 --save_freq=2500 --log_freq=100 --num_workers=12 --wandb.enable=false \
    > out/train_officehouse.log 2>&1 &
sleep 60
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$EVAL_GPU CUDA_VISIBLE_DEVICES=$EVAL_GPU .venv/bin/python -W ignore scripts/watch_eval.py \
    --run $RUN --n 50 --tag house --eval-args "$EVAL_ARGS" > out/watch_officehouse.log 2>&1
echo "[$(date +%T)] done"
