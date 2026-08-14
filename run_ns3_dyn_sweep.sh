#!/bin/bash
# ns-3 DYNAMIC-topology packet-level validation (frequent_break).
#
# Same path-replay harness as run_ns3_sweep.sh, but links now go DOWN per the
# env's real per-slot break schedule (links_<policy>.csv from
# ns3_trace_extractor.py --scenario frequent_break). A packet hitting a down
# link waits at the node and retries at the next slot boundary; a packet whose
# age reaches its class deadline is dropped. Tests whether MAPPO's env
# advantage (which vanished in the STATIC replay) reappears when the replay
# includes the topology dynamics the policies actually conditioned on.
#
# Usage (from WSL):  bash /mnt/f/leo-routing-preliminary-matlab/run_ns3_dyn_sweep.sh [bw_kbps] [qsize]
set -u
cd /home/nsuser/ns-3.48
IN=/mnt/f/leo-routing-preliminary-matlab/experiments/IEEE-NS3-DYN
BW="${1:-36}"
QSIZE="${2:-8}"
ESLOTS=30                       # env episode_slots (printed by the extractor)
DL="30,12,20"                   # env packet_class_deadlines
OUT="$IN/dyn_sweep_bw${BW}_q${QSIZE}.csv"
echo "policy,slot_sec,bw_kbps,qsize,sent,delivered,delivery_ratio,queue_drops,arrived,deadline_drops,wait_events,stranded,mean_delay_ms,p95_delay_ms,load_imbalance" > "$OUT"
for SS in 1.0 0.5 0.25 0.125; do
  for POL in mappo dijkstra; do
    LINE=$(./ns3 run "scratch/leo-validation --input=$IN/packets_${POL}.csv --output=$IN/dynpkt_${POL}_ss${SS}.csv --policy-name=$POL --bw-kbps=${BW} --qsize-pkts=${QSIZE} --slot-sec=${SS} --episode-slots=${ESLOTS} --deadline-slots=${DL} --links=$IN/links_${POL}.csv" 2>/dev/null | grep '^RESULT,')
    SENT=$(echo "$LINE" | sed -n 's/.*[^_]sent=\([0-9]*\).*/\1/p')
    DEL=$(echo "$LINE" | sed -n 's/.*delivered=\([0-9]*\).*/\1/p')
    RAT=$(echo "$LINE" | sed -n 's/.*delivery_ratio=\([0-9.]*\).*/\1/p')
    QD=$(echo "$LINE" | sed -n 's/.*queue_drops=\([0-9]*\).*/\1/p')
    AR=$(echo "$LINE" | sed -n 's/.*arrived=\([0-9]*\).*/\1/p')
    DD=$(echo "$LINE" | sed -n 's/.*deadline_drops=\([0-9]*\).*/\1/p')
    WE=$(echo "$LINE" | sed -n 's/.*wait_events=\([0-9]*\).*/\1/p')
    ST=$(echo "$LINE" | sed -n 's/.*stranded=\([0-9]*\).*/\1/p')
    MD=$(echo "$LINE" | sed -n 's/.*mean_delay_ms=\([-0-9.]*\).*/\1/p')
    P95=$(echo "$LINE" | sed -n 's/.*p95_delay_ms=\([-0-9.]*\).*/\1/p')
    IMB=$(echo "$LINE" | sed -n 's/.*load_imbalance=\([-0-9.]*\).*/\1/p')
    echo "${POL},${SS},${BW},${QSIZE},${SENT},${DEL},${RAT},${QD},${AR},${DD},${WE},${ST},${MD},${P95},${IMB}" >> "$OUT"
    printf "  %-9s slot=%ss  ratio=%s  arrived=%s  dl_drops=%s  waits=%s  q_drops=%s  p95=%sms\n" \
      "$POL" "$SS" "${RAT}" "${AR}" "${DD}" "${WE}" "${QD}" "${P95}"
  done
done
echo "=> wrote $OUT"
