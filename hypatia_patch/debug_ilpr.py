#!/bin/bash
set -ux
cd $HOME/hypatia/paper/satellite_networks_state
python3 - << 'EOF'
import sys, os
sys.path.insert(0, ".")
src = "gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
state = os.path.join(src, "dynamic_state_100ms_for_10s")
files = sorted(f for f in os.listdir(state) if f.startswith("fstate_"))
print("fstate files found:", len(files), "first:", files[:3], "last:", files[-1])
# load first two and inspect
def load(p):
    t = {}
    n = 0
    for line in open(p):
        parts = line.strip().split(",")
        if len(parts) != 5:
            continue
        a, b, c, d, e = (int(x) for x in parts)
        t.setdefault(a, {})[b] = (c, d, e)
        n += 1
    return t, n
t0, n0 = load(os.path.join(state, files[0]))
print("file", files[0], "lines parsed:", n0, "nodes:", len(t0))
t1, n1 = load(os.path.join(state, files[1]))
print("file", files[1], "lines parsed:", n1, "nodes:", len(t1))
same = diff = 0
for node, tgts in t0.items():
    for tgt, v in tgts.items():
        if t1.get(node, {}).get(tgt) == v:
            same += 1
        else:
            diff += 1
print(f"interval0 vs interval1 next-hops: same={same} diff={diff}")
EOF
