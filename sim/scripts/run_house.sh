#!/bin/bash
# House simulation + kcare_robot backend, inside the kaair_sim_humble container.
#
#   bash scripts/run_house.sh start      # sim bridge (ROS) + kcare_robot API on :8001
#   bash scripts/run_house.sh stop
#   bash scripts/run_house.sh status
#   bash scripts/run_house.sh restart-sim | restart-kcare
#
# From the host:  docker exec kaair_sim_humble bash -lc "cd /sim && bash scripts/run_house.sh start"
# Logs: /sim/out/house/bridge.log, /sim/out/house/kcare.log
# ROS_DOMAIN_ID stays 77 (container default): nothing here can reach the real robot.
source /opt/ros/humble/setup.bash
source /workspace/ros2/ros2_ws/install/setup.bash
cd /sim
LOG=/sim/out/house; mkdir -p $LOG
GPU=${GPU:-0}                     # EGL render device
PORT=${PORT:-8001}                # kcare_sim.aistations.org (cloudflared on this host) -> localhost:8001
KCARE_PY=/sim/.kcare_venv/bin/python

start_sim() {
    pgrep -f "python3 scripts/[h]ouse_bridge.py" >/dev/null && { echo "sim already running"; return; }
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU nohup python3 scripts/house_bridge.py $BRIDGE_ARGS > $LOG/bridge.log 2>&1 &
    for i in $(seq 30); do grep -q "sim ready" $LOG/bridge.log && break; sleep 1; done
    grep -q "sim ready" $LOG/bridge.log && echo "sim ready" || { echo "sim failed:"; tail -20 $LOG/bridge.log; }
}
start_kcare() {
    pgrep -f "[u]vicorn.*--port $PORT" >/dev/null && { echo "kcare already running"; return; }
    [ -x $KCARE_PY ] || { echo "no $KCARE_PY"; return 1; }
    # the real robot's code, read-only from the dev PC (host: ~/.local/bin/kcare_remote_mount.sh, cron)
    [ -d /kcare_mnt/remote_dir/kcare_robot ] || { echo "kcare code not mounted (/kcare_mnt/remote_dir): is the tunnel on .11 up?"; return 1; }
    cd /sim
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 nohup $KCARE_PY -m uvicorn kcare_sim_app:app --app-dir /sim/scripts \
        --host 0.0.0.0 --port $PORT > $LOG/kcare.log 2>&1 &
    for i in $(seq 60); do curl -s -m 1 localhost:$PORT/skills >/dev/null && break; sleep 1; done
    curl -s -m 2 localhost:$PORT/skills >/dev/null || { echo "kcare failed:"; tail -30 $LOG/kcare.log; return 1; }
    # ready = the ROS agents are up and joint states flow (a skill called before that fails at once)
    for i in $(seq 30); do
        curl -s -m 5 -X POST localhost:$PORT/skill/head_state -H "Content-Type: application/json" -d "{}" \
            | grep -q current_ry && { echo "kcare API on :$PORT (robot connected)"; return 0; }
        sleep 2
    done
    echo "kcare API on :$PORT, but no joint states yet — is the sim running?"
}
# stop = SIGTERM, wait up to 10 s, then SIGKILL (a hung skill can keep uvicorn from exiting)
stop_pat() {
    pkill -f "$1" || return 1
    for i in $(seq 20); do pgrep -f "$1" >/dev/null || return 0; sleep 0.5; done
    pkill -9 -f "$1"; sleep 1
}
start_vla() {
    # /drawer/open for kcare's vla_drawer_open: SmolVLA through vla-serve (container vla-serve-kaair, :11450)
    pgrep -f "python3 scripts/[v]la_drawer_node.py" >/dev/null && { echo "vla node already running"; return; }
    curl -s -m 3 localhost:11450/health >/dev/null || { echo "vla-serve not reachable on :11450 (docker start vla-serve-kaair)"; return 1; }
    PYTHONPATH=/sim/third_party:/sim/third_party/vlaclient_deps:$PYTHONPATH nohup python3 scripts/vla_drawer_node.py \
        > $LOG/vla_drawer.log 2>&1 &
    for i in $(seq 20); do grep -q "/drawer/open ready" $LOG/vla_drawer.log && break; sleep 1; done
    grep -q "/drawer/open ready" $LOG/vla_drawer.log && echo "vla node ready" || { echo "vla node failed:"; tail -15 $LOG/vla_drawer.log; }
}
stop_vla()   { stop_pat "python3 scripts/[v]la_drawer_node.py" && echo "vla node stopped"; }
stop_sim()   { stop_pat "python3 scripts/[h]ouse_bridge.py" && echo "sim stopped"; }
stop_kcare() { stop_pat "[u]vicorn.*--port $PORT" && echo "kcare stopped"; }

case "$1" in
  start)         start_sim; start_vla; start_kcare ;;
  stop)          stop_kcare; stop_vla; stop_sim ;;
  restart-vla)   stop_vla; sleep 1; start_vla ;;
  restart-sim)   stop_sim; sleep 2; start_sim ;;
  restart-kcare) stop_kcare; sleep 2; start_kcare ;;
  sim)           start_sim ;;
  status)
    pgrep -f "python3 scripts/[h]ouse_bridge.py" >/dev/null && echo "sim: running" || echo "sim: stopped"
    pgrep -f "python3 scripts/[v]la_drawer_node.py" >/dev/null && echo "vla node: running" || echo "vla node: stopped"
    pgrep -f "[u]vicorn.*--port $PORT" >/dev/null && echo "kcare: running on :$PORT" || echo "kcare: stopped" ;;
  *) echo "usage: $0 start|stop|status|sim|restart-sim|restart-kcare|restart-vla" ;;
esac
