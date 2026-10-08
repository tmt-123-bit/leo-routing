#!/bin/bash
set -e
SN="$HOME/hypatia/ns3-sat-sim/simulator/contrib/satellite-network"
P="/mnt/f/LEO研究代码汇总/hypatia_patch"
cp "$P"/model/*.h "$P"/model/*.cc "$SN/model/"
cp "$P"/helper/*.h "$P"/helper/*.cc "$SN/helper/"
cp "$P/edit/topology-satellite-network.cc" "$SN/model/"
cp "$P/edit/point-to-point-laser-net-device.cc" "$SN/model/"
cp "$P/edit/arbiter-single-forward-helper.h" "$P/edit/arbiter-single-forward-helper.cc" "$SN/helper/"
cp "$P/edit/wscript" "$SN/"
cp "$P/edit/main_satnet.cc" "$HOME/hypatia/ns3-sat-sim/simulator/scratch/main_satnet/"
cp "$P/custom_pn.py" "$HOME/hypatia/paper/satellite_networks_state/"
ls "$SN/model/" | grep -E 'purge|deadline|oracle'
echo DEPLOY_OK
cd "$HOME/hypatia/ns3-sat-sim/simulator"
./waf build -j8 2>&1 | tail -5
