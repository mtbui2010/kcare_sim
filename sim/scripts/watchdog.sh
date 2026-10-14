#!/bin/bash
# Memory watchdog for my jobs on the shared A6000: if MemAvailable < 15 % of RAM, kill my lowest-priority job.
# Priority (killed first -> last): proc_sweep / reach_sweep, gen_dataset, eval_pick_policy, lerobot-train.
LOG=~/trung_workdir/kcare-sim/sim/out/watchdog.log
while true; do
  tot=$(awk '/MemTotal/{print $2}' /proc/meminfo); av=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  pct=$(( av * 100 / tot )); ld=$(cut -d' ' -f1 /proc/loadavg)
  if [ "$pct" -lt 15 ]; then
    for pat in '[p]roc_sweep.py' '[r]each_sweep.py' '[g]en_dataset.py' '[e]val_pick_policy.py' '[e]val_policy.py' '[l]erobot-train'; do
      pids=$(pgrep -f "$pat")
      if [ -n "$pids" ]; then echo "$(date '+%F %T') avail ${pct}% load $ld -> kill $pat ($pids)" >> $LOG; kill $pids; break; fi
    done
  fi
  [ $(( $(date +%s) % 1800 )) -lt 30 ] && echo "$(date '+%F %T') avail ${pct}% load $ld" >> $LOG
  sleep 30
done
