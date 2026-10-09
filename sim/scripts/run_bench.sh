#!/bin/bash
# Skill bench batch: baseline (standoff 0.55), repeatability, closer standoff (0.42). Logs in out/bench/*.log
source /workspace/ros2/ros2_ws/install/setup.bash; export ROS_DOMAIN_ID=77
cd /sim
SC="bench_100 bench_101 bench_102 bench_103 bench_104 bench_105 bench_106 bench_107 bench_108 bench_109 bench_110 bench_111"
python3 -u scripts/skill_bench.py --scenes $SC --tasks 5 --out out/bench/base055 > out/bench/base055.log 2>&1
python3 -u scripts/skill_bench.py --scenes bench_100 bench_103 bench_106 --tasks 3 --repeat 3 --seed 7 --out out/bench/repeat > out/bench/repeat.log 2>&1
python3 -u scripts/skill_bench.py --scenes $SC --tasks 5 --standoff 0.42 --out out/bench/near042 > out/bench/near042.log 2>&1
echo done > out/bench/ALL_DONE
