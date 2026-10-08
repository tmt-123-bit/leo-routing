#!/bin/bash
# qaware (queue-aware routing) at 1584: {droptail, purge_edf}.
set -u
PY=/f/leo-venv/Scripts/python.exe
PATCH=/f/LEO研究代码汇总/hypatia_patch
GEN="starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
ROUTES='\\wsl.localhost\Ubuntu-22.04\home\nsuser\hypatia\paper\satellite_networks_state\gen_data\'"$GEN"
HOSTIP=$(wsl -d Ubuntu-22.04 -u nsuser -- bash -lc "ip route show default" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "host ip: $HOSTIP"

run_arm () {
  local QT=$1
  $PY "$PATCH/bridge_server.py" --policy qaware --routes-dir "$ROUTES" \
      --port 7350 > /tmp/bridge_1584_$QT.log 2>&1 &
  local SPID=$!
  sleep 6
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -u nsuser -- \
      bash /mnt/f/LEO研究代码汇总/hypatia_patch/bridge_run_arm.sh qaware $QT $HOSTIP "$GEN"
  kill $SPID 2>/dev/null
  sleep 1
}

run_arm droptail
run_arm purge_edf
echo BRIDGE_1584_DONE
