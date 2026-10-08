#!/bin/bash
# Bridge-routing driver (Windows side): start server per arm, run WSL arm.
set -u
PY=/f/leo-venv/Scripts/python.exe
PATCH=/f/LEO研究代码汇总/hypatia_patch
GEN="knn_4x6_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls"
ROUTES='\\wsl.localhost\Ubuntu-22.04\home\nsuser\hypatia\paper\satellite_networks_state\gen_data\'"$GEN"
HOSTIP=$(wsl -d Ubuntu-22.04 -u nsuser -- bash -lc "ip route show default" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "host ip: $HOSTIP"

run_arm () {
  local POL=$1 QT=$2
  $PY "$PATCH/bridge_server.py" --policy $POL --routes-dir "$ROUTES" \
      --port 7350 > /tmp/bridge_${POL}.log 2>&1 &
  local SPID=$!
  sleep 6
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 -u nsuser -- \
      bash /mnt/f/LEO研究代码汇总/hypatia_patch/bridge_run_arm.sh $POL $QT $HOSTIP
  kill $SPID 2>/dev/null
  sleep 1
}

run_arm qaware droptail
run_arm qaware purge_edf
run_arm mappo0 droptail
run_arm mappo0 purge_edf
echo BRIDGE_DRIVER_DONE
