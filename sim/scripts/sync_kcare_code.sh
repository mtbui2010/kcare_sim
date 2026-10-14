#!/bin/bash
# HOST side: copy the A6000's own robot_agent / kcare_robot / embodiplan repos into /sim/kcare_code for the sim's
# kcare (run_house.sh start_kcare), then restart it with:  docker exec kaair_sim_humble bash -lc "cd /sim && bash scripts/run_house.sh restart-kcare"
set -e
SRC=${SRC:-$HOME/trung_workdir}
DST=$(cd "$(dirname "$0")/.." && pwd)/kcare_code
mkdir -p $DST
for r in robot_agent kcare_robot embodiplan; do
  rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' $SRC/$r/ $DST/$r/
done
echo "synced $SRC/{robot_agent,kcare_robot,embodiplan} -> $DST"
