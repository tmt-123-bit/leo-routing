#!/bin/bash
R=~/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline
for ARM in base mech; do
  echo "=== $ARM ==="
  ls "$R/$ARM/logs_ns3/" | head -6
  echo "sent:"; cat "$R/$ARM/logs_ns3/deadline_udp_sent.csv" 2>/dev/null
  echo "packets lines:"; wc -l "$R/$ARM/logs_ns3/deadline_udp_packets.csv" 2>/dev/null
  grep -h "PURGE_TOTAL\|PACKET_MGMT" "$R/$ARM/logs_ns3/"*.txt 2>/dev/null | head -2
done
echo "=== python analysis ==="
python3 - << 'EOF'
import csv, os
root = os.path.expanduser("~/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline")
for arm in ("base", "mech"):
    d = os.path.join(root, arm, "logs_ns3")
    sent = 0
    try:
        with open(os.path.join(d, "deadline_udp_sent.csv")) as f:
            for row in csv.DictReader(f):
                try:
                    sent += int(row["sent_packets"])
                except (TypeError, ValueError):
                    continue  # second appended header block
    except FileNotFoundError:
        print(arm, "no sent log"); continue
    arrived = ontime = 0
    tight_arr = tight_on = loose_arr = loose_on = 0
    try:
        with open(os.path.join(d, "deadline_udp_packets.csv")) as f:
            for row in csv.DictReader(f):
                try:
                    ot = int(row["on_time"]); dln = int(row["deadline_ns"]); snt = int(row["seq_send_ns"])
                except (TypeError, ValueError):
                    continue
                arrived += 1; ontime += ot
                if dln - snt <= 30000000:
                    tight_arr += 1; tight_on += ot
                else:
                    loose_arr += 1; loose_on += ot
    except FileNotFoundError:
        pass
    print(f"{arm}: sent={sent} arrived={arrived} on_time={ontime} "
          f"overall={ontime/sent if sent else 0:.4f} | "
          f"tight(22ms): arr={tight_arr} on={tight_on} | "
          f"loose(60ms): arr={loose_arr} on={loose_on} "
          f"loose_on/sent_loose={loose_on/(sent/2) if sent else 0:.4f}")
EOF
