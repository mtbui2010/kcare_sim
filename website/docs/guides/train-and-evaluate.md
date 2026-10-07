# Train and evaluate a policy

## Train

SmolVLA from LeRobot, fine-tuned from the office checkpoint on office + house data
([`run_house_train.sh`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/run_house_train.sh) does all steps):

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.path=$BASE/checkpoints/020000/pretrained_model --policy.push_to_hub=false \
  --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --dataset.repo_id=kaair_sim_open_drawer_officehouse1000 --dataset.root=/nas/datasets/kaair_sim_open_drawer_officehouse1000 \
  --batch_size=64 --steps=10000 --save_freq=2500 --wandb.enable=false --output_dir=$RUN
```

```text
INFO ... step:2K smpl:154K ep:457 epch:0.46 loss:0.006 grdn:0.201 lr:8.7e-05 updt_s:1.406 data_s:0.03
```

About 1.4 s per step at batch 64 on one RTX A6000: 10k steps take about 4 hours.

## Evaluate

[`eval_policy.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/eval_policy.py) runs the policy closed-loop on 50 scenes drawn
from fixed seeds (`--seed0 1e9`, far from the data seeds), with the same camera model as the
data. Success: a drawer opened at least 20 cm within 15 s (simulation ground truth).

```bash
python3 scripts/eval_policy.py --ckpt $RUN/checkpoints/002500/pretrained_model --n 50 --tag house \
    --task task_open_drawer_house.yaml --rand-config randomization_house.yaml --camera-model
```

```text
[ 4461s] ep  49 FAIL opened   0.0 cm   success so far 8% (50)
success rate 8.0% over 50 episodes, mean opened 2.9 cm -> .../020000_house_results.json
```

[`watch_eval.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/watch_eval.py) evaluates every checkpoint of a run as soon as it
is saved, and [`eval_clips.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/eval_clips.py) cuts labelled success / failure clips.

![Success rates with 95 % intervals](../assets/img/success_rates.png)

Read the chart with its intervals: 50 scenes give about ±13 points around 50 %. The office
policy's 70 % → 88 % between 5k and 20k steps is suggestive but the intervals still overlap;
70 % vs 68 % is noise. The jump from 8 % to 52 % in the house is far outside the noise.
