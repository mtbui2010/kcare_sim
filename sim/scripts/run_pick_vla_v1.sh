#!/bin/bash
# PICK phase-A VLA v1 (2026-10-09): wait for the house dataset, train SmolVLA on it (GPU 0), and evaluate every
# checkpoint closed-loop in the house through a second vla-serve container on GPU 3 (port 11451; vla-serve-kaair on
# 11450 is left alone).
#
# Runs on the A6000 HOST (it needs docker for the vla-serve container); everything else runs in kaair_sim_humble:
#   nohup bash ~/trung_workdir/kcare-sim/sim/scripts/run_pick_vla_v1.sh > ~/trung_workdir/kcare-sim/sim/out/pick_vla_v1.log 2>&1 &
#
# Base weights: /nas/checkpoints/_bases/smolvla_base_kaair (what the drawer v1 lineage started from), with its config's
# input / output features rewritten for this dataset (17-D state, 10-D action; SmolVLA pads both to 32, so the weights
# are unchanged) -> _bases/smolvla_base_kaair_pick17. Training normalizes with the dataset's stats.
set -u
C=kaair_sim_humble
REPO=kaair_sim_pick_house_v1_1200
DATE=${DATE:-$(date +%Y%m%d)}
RUN=/nas/checkpoints/smolvla_pick_house_v1_$DATE            # container path; host: /mnt/nas/lerobot_shared/kaair_sim/...
HOST_NAS=/mnt/nas/lerobot_shared/kaair_sim
STEPS=${STEPS:-15000}
SAVE=${SAVE:-2000}
N_EVAL=${N_EVAL:-50}
PORT=11451
SERVE=vla-serve-pick
ts() { date +%T; }
inc() { docker exec $C bash -c "$1"; }

echo "[$(ts)] waiting for /nas/datasets/$REPO (generator log /sim/out/pick_gen_v1.log)"
while true; do
  if inc "test -f /nas/datasets/$REPO/meta/info.json && ! pgrep -f '[g]en_dataset.py.*$REPO' > /dev/null"; then break; fi
  sleep 300
done
inc "tail -n 1 /sim/out/pick_gen_v1.log"

echo "[$(ts)] base weights for 17-D state / 10-D action"
inc "cd /sim && .venv/bin/python - <<'EOF'
import json, os, shutil
src, dst = '/nas/checkpoints/_bases/smolvla_base_kaair', '/nas/checkpoints/_bases/smolvla_base_kaair_pick17'
if not os.path.exists(f'{dst}/config.json'):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    c = json.load(open(f'{dst}/config.json'))
    c['input_features']['observation.state']['shape'] = [17]
    c['output_features']['action']['shape'] = [10]
    json.dump(c, open(f'{dst}/config.json', 'w'), indent=4)
print('base', dst, json.load(open(f'{dst}/config.json'))['input_features']['observation.state'])
EOF"

echo "[$(ts)] training -> $RUN ($STEPS steps, save every $SAVE)"
docker exec -d $C bash -c "cd /sim && CUDA_VISIBLE_DEVICES=0 .venv/bin/lerobot-train \
    --policy.path=/nas/checkpoints/_bases/smolvla_base_kaair_pick17 --policy.push_to_hub=false \
    --rename_map='{\"observation.images.head\":\"observation.images.camera1\",\"observation.images.wrist\":\"observation.images.camera2\"}' \
    --dataset.repo_id=$REPO --dataset.root=/nas/datasets/$REPO --dataset.video_backend=pyav \
    --output_dir=$RUN --job_name=smolvla_pick_house_v1 \
    --batch_size=64 --steps=$STEPS --save_freq=$SAVE --log_freq=100 --num_workers=8 --wandb.enable=false \
    > out/train_pick_v1.log 2>&1"

serve() {     # (re)start the eval server on GPU 3 with one checkpoint
  docker rm -f $SERVE > /dev/null 2>&1
  docker run -d --name $SERVE --gpus '"device=3"' -p 127.0.0.1:$PORT:11450 \
      -v $HOST_NAS/checkpoints:/ckpt -v $HOME/.cache/huggingface:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
      vla-serve:gpu-cu128 serve --host 0.0.0.0 "$1" > /dev/null
  for i in $(seq 120); do curl -s -m 2 127.0.0.1:$PORT/v1/models | grep -q model && return 0; sleep 5; done
  echo "[$(ts)] $SERVE did not come up"; docker logs --tail 20 $SERVE; return 1
}

for s in $(seq $SAVE $SAVE $STEPS); do
  step=$(printf '%06d' $s)
  ck=$RUN/checkpoints/$step/pretrained_model
  echo "[$(ts)] waiting for $step"
  while true; do      # complete = model.safetensors present and the folder unchanged for 2 min
    if inc "test -f $ck/model.safetensors && [ \$(( \$(date +%s) - \$(stat -c %Y \$(ls -t $ck/* | head -1)) )) -gt 120 ]"; then break; fi
    if ! inc "pgrep -f '[l]erobot-train.*smolvla_pick_house_v1' > /dev/null" && ! inc "test -f $ck/model.safetensors"; then
      echo "[$(ts)] training stopped before $step"; inc "tail -n 20 /sim/out/train_pick_v1.log"; break 2
    fi
    sleep 120
  done
  echo "[$(ts)] eval $step on GPU 3 ($N_EVAL held-out house episodes)"
  serve /ckpt/${RUN#/nas/checkpoints/}/checkpoints/$step/pretrained_model || continue
  inc "cd /sim && mkdir -p $RUN/eval && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 OMP_NUM_THREADS=4 .venv/bin/python -W ignore \
       scripts/eval_pick_policy.py --url http://127.0.0.1:$PORT --n $N_EVAL --videos 3 \
       --out $RUN/eval/${step}_pick_results.json > out/eval_pick_v1_$step.log 2>&1; tail -n 3 out/eval_pick_v1_$step.log"
done
docker rm -f $SERVE > /dev/null 2>&1

inc "cd /sim && .venv/bin/python - <<'EOF'
import glob, json, os
rows = []
for f in sorted(glob.glob('$RUN/eval/*_pick_results.json')):
    r = json.load(open(f)); rows.append((os.path.basename(f)[:6], r['success_rate'], r['valid'], r['collisions'], r['base_used'], r['mean_t_success']))
with open('$RUN/eval/summary_pick.md', 'w') as fh:
    fh.write('| step | success | valid | collisions | base used | mean time [s] |\n|---|---|---|---|---|---|\n')
    for s, sr, n, c, b, t in rows:
        fh.write(f'| {s} | {sr * 100:.0f}% | {n} | {c} | {b} | {t} |\n')
print(open('$RUN/eval/summary_pick.md').read())
EOF"
echo "[$(ts)] done"
