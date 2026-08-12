#!/bin/bash
# ns-3 packet-level validation: offered-LOAD sweep (via slot-sec compression)
# at a fixed ISL rate. Lower slot-sec => packets injected over a tighter window
# => higher instantaneous offered load => congestion. Mirrors the env load sweep.
# Compares MAPPO vs Dijkstra path-replay under packet-level FIFO/drop-tail.
set -u
cd /home/nsuser/ns-3.48
IN=/mnt/f/leo-routing-preliminary-matlab/experiments/IEEE-NS3
BW="${1:-36}"          # ISL rate kb/s (36 ~= env 3-pkt/slot capacity)
QSIZE="${2:-8}"
OUT="$IN/load_sweep_bw${BW}_q${QSIZE}.csv"
echo "policy,slot_sec,bw_kbps,qsize,sent,delivered,delivery_ratio,queue_drops,mean_delay_ms,p95_delay_ms,load_imbalance" > "$OUT"
for SS in 0.5 0.25 0.125 0.0625; do
  for POL in mappo dijkstra; do
    LINE=$(./ns3 run "scratch/leo-validation --input=$IN/packets_${POL}.csv --output=$IN/nspkt_${POL}_ss${SS}.csv --policy-name=$POL --bw-kbps=${BW} --qsize-pkts=${QSIZE} --slot-sec=${SS}" 2>/dev/null | grep '^RESULT,')
    SENT=$(echo "$LINE" | sed -n 's/.*sent=\([0-9]*\).*/\1/p')
    DEL=$(echo "$LINE" | sed -n 's/.*delivered=\([0-9]*\).*/\1/p')
    RAT=$(echo "$LINE" | sed -n 's/.*delivery_ratio=\([0-9.]*\).*/\1/p')
    QD=$(echo "$LINE" | sed -n 's/.*queue_drops=\([0-9]*\).*/\1/p')
    MD=$(echo "$LINE" | sed -n 's/.*mean_delay_ms=\([-0-9.]*\).*/\1/p')
    P95=$(echo "$LINE" | sed -n 's/.*p95_delay_ms=\([-0-9.]*\).*/\1/p')
    IMB=$(echo "$LINE" | sed -n 's/.*load_imbalance=\([-0-9.]*\).*/\1/p')
    echo "${POL},${SS},${BW},${QSIZE},${SENT},${DEL},${RAT},${QD},${MD},${P95},${IMB}" >> "$OUT"
    printf "  %-9s slot=%.4fs load~%.0fx  ratio=%s  drops=%s  p95=%sms  imb=%s\n" "$POL" "$SS" "$(echo "0.5/$SS"|bc -l)" "${RAT}" "${QD}" "${P95}" "${IMB}"
  done
done
echo "=> wrote $OUT"
