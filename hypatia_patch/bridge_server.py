"""Policy server for the official-Hypatia bridge arbiter (Windows side).

Per 100 ms forwarding interval the ns-3 side reports per-node ISL queue
occupancy; this server computes the next-hop tables and returns deltas.

Policies:
  qaware   queue-aware shortest-path routing (POMAP-style operationalization:
           edge cost = base hops + alpha * normalized queue of the next node)
  mappo0   zero-shot transplant of the trained constrained-MAPPO actor
           (candidate features mapped from Hypatia state; progress-masked so
           every accepted hop strictly decreases hop distance => loop-free)

Loop safety: overrides are only accepted when hops(v, dst) < hops(u, dst)
in the fresh base (official fstate) tables, so installed tables stay
loop-free by construction.
"""
import argparse
import os
import socket
import sys
import threading

N_PLANES, SPP = 4, 6
N_SATS = N_PLANES * SPP
QMAX = 45.0
ALPHA = 2.0
D_REF_MS = 10.0
MAX_STEPS = 30


def torus_dist(a, b):
    p1, s1 = a // SPP, a % SPP
    p2, s2 = b // SPP, b % SPP
    dp = abs(p1 - p2); dp = min(dp, N_PLANES - dp)
    ds = abs(s1 - s2); ds = min(ds, SPP - ds)
    return dp + ds


def progress(u, v, dst):
    return (torus_dist(u, dst) - torus_dist(v, dst)) / float(N_PLANES + SPP)


def coords(node, t_slot):
    plane, pos = node // SPP + 1, node % SPP + 1
    u0 = (plane - 1) / N_PLANES
    w0 = (pos - 1) / SPP
    u = (u0 + t_slot / float(MAX_STEPS * N_PLANES)) % 1.0
    w = (w0 + t_slot / float(MAX_STEPS)) % 1.0
    return u, w


def load_isls(path):
    adj = {}
    for line in open(path):
        a, b = (int(x) for x in line.split())
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def parse_fstate(path, table, edge_if):
    """Apply a (full-first, then delta) fstate file onto table."""
    n = 0
    for line in open(path):
        f = line.strip().split(",")
        if len(f) != 5:
            continue
        node, tgt, nh, mi, ni = (int(x) for x in f)
        table.setdefault(node, {})[tgt] = (nh, mi, ni)
        if nh >= 0:
            edge_if[(node, nh)] = (mi, ni)
        n += 1
    return n


def hop_maps(base, targets, n_nodes):
    """hops[node][target] chasing the base table."""
    hops = {}
    for g in targets:
        h = {g: 0}
        # reverse-BFS would need reverse edges; chasing from each node is
        # fine at these sizes (<= 124 nodes)
        for u in range(n_nodes):
            cur, steps, ok = u, 0, False
            seen = set()
            while cur not in seen:
                if cur == g:
                    ok = True
                    break
                seen.add(cur)
                ent = base.get(cur, {}).get(g)
                if ent is None or ent[0] < 0:
                    break
                cur = ent[0]
                steps += 1
                if steps > 400:
                    break
            if ok:
                h[u] = steps
        hops[g] = h
    return hops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("qaware", "mappo0"), required=True)
    ap.add_argument("--routes-dir", required=True,
                    help="UNC path to the gen_data/<shell> dir (contains isls.txt"
                         " and dynamic_state_*/fstate_*.txt)")
    ap.add_argument("--port", type=int, default=7350)
    ap.add_argument("--checkpoint", default=(
        r"F:\leo-routing-preliminary-matlab\experiments\train-main\checkpoints"
        r"\hotspot_high_load\no_lifetime\seed_1024\*"))
    args = ap.parse_args()

    state_dir = None
    for d in os.listdir(args.routes_dir):
        if d.startswith("dynamic_state_"):
            state_dir = os.path.join(args.routes_dir, d)
            break
    assert state_dir, "no dynamic_state dir"
    adj = load_isls(os.path.join(args.routes_dir, "isls.txt"))

    policy = None
    action_size = feature_dim = None
    if args.policy == "mappo0":
        sys.path.insert(0, r"F:\leo-routing-preliminary-matlab\src")
        import glob as _glob
        from ns3_policy_bridge import Ns3PolicyBridge
        ckpt = sorted(_glob.glob(args.checkpoint + os.sep + "validation_best.pt"))[-1]
        bridge = Ns3PolicyBridge(ckpt, device="cpu")
        policy = bridge.policy
        action_size, feature_dim = bridge.action_size, bridge.feature_dim
        print(f"[server] mappo0 loaded {ckpt} action_size={action_size} "
              f"feature_dim={feature_dim}", flush=True)

    import numpy as np

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"[server] {args.policy} listening on {args.port}", flush=True)
    conn, _ = srv.accept()

    def send(line):
        conn.sendall((line + "\n").encode())

    f = conn.makefile("r")
    send("READY " + args.policy)

    base = {}          # official fstate accumulation: node -> {tgt: (nh,mi,ni)}
    edge_if = {}       # (u, v) -> (my_if, next_if) learned from fstate entries
    installed = {}     # what ns-3 currently has (our mirror)
    targets = None
    fstate_files = sorted(x for x in os.listdir(state_dir) if x.startswith("fstate_"))

    while True:
        line = f.readline()
        if not line:
            break
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "HELLO":
            n_nodes = int(parts[1])
            continue
        if parts[0] != "INTERVAL":
            continue
        t = int(parts[1])
        queues = {}
        while True:
            ql = f.readline().strip().split()
            if ql and ql[0] == "END":
                break
            if ql and ql[0] == "Q":
                queues[int(ql[1])] = int(ql[2])

        # advance the official base table to this interval
        fn = f"fstate_{t}.txt"
        if fn in fstate_files:
            parse_fstate(os.path.join(state_dir, fn), base, edge_if)
        if targets is None:
            tset = set()
            for row in base.values():
                tset.update(row.keys())
            targets = sorted(tset)
        hops = hop_maps(base, targets, n_nodes)

        t_slot = t // 1000000000
        # candidates: (u, g) where u routes via ISL and progress options exist
        cand_map = {}   # (u,g) -> (cands, cur_nh)
        for u in range(N_SATS):
            for g in targets:
                ent = base.get(u, {}).get(g)
                cur = ent[0] if ent else -1
                if cur < 0 or cur >= N_SATS:
                    continue  # not an ISL hop: keep official
                h_u = hops[g].get(u)
                if h_u is None or h_u == 0:
                    continue
                cands = [v for v in sorted(adj.get(u, ()))
                         if hops[g].get(v) is not None
                         and hops[g][v] < h_u
                         and (u, v) in edge_if]
                if cands:
                    cand_map[(u, g)] = (cands, cur)

        overrides = {}   # (u,g) -> chosen nh
        if args.policy == "qaware":
            for (u, g), (cands, cur) in cand_map.items():
                best = min(cands, key=lambda v: hops[g][v]
                           + ALPHA * min(1.0, queues.get(v, 0) / QMAX))
                overrides[(u, g)] = best
        elif cand_map:
            keys = sorted(cand_map.keys())
            obs = np.zeros((len(keys), action_size, feature_dim), dtype=np.float32)
            mask = np.zeros((len(keys), action_size), dtype=bool)
            slots = {}
            uu, uw = 0.0, 0.0
            for i, (u, g) in enumerate(keys):
                cands, cur = cand_map[(u, g)]
                qnorm_u = min(1.0, queues.get(u, 0) / QMAX)
                uu, uw = coords(u, t_slot)
                du, dw = uu, uw   # dst is a GS: no torus coords; reuse u's
                for ci, v in enumerate(cands):
                    slot = ci + 1
                    if slot >= action_size:
                        break
                    vu, vw = coords(v, t_slot)
                    feats = [
                        qnorm_u,
                        min(1.0, queues.get(v, 0) / QMAX),
                        8.0 / D_REF_MS,          # uniform per-hop delay approx
                        1.0, 0.0, 0.995, 1.0, 0.0,
                        progress(u, v, u),       # torus progress toward self=0
                        vu, vw, uu, uw, du, dw,
                        1.0, 0.0,               # hops=0 virtual packet
                        1.0 if v == cur else 0.0,
                        0.0,
                        (t_slot % MAX_STEPS) / float(MAX_STEPS),
                        0.0,                    # waiting ratio
                        1.0, 0.0, 0.0,          # class one-hot (class 0)
                        0.0, 0.0,
                    ]
                    obs[i, slot, :min(26, feature_dim)] = feats[:feature_dim]
                    mask[i, slot] = True
                    slots[(i, slot)] = v
            acts = policy(obs.reshape(len(keys), -1), mask)
            for i, (u, g) in enumerate(keys):
                a = int(acts[i])
                if mask[i, a]:
                    overrides[(u, g)] = slots[(i, a)]

        # emit deltas vs installed mirror (full 5-tuples: ifs from fstate)
        n_sent = 0
        for u in range(n_nodes):
            for g in targets:
                ent = base.get(u, {}).get(g)
                nh, mi, ni = ent if ent else (-1, -1, -1)
                if (u, g) in overrides:
                    v = overrides[(u, g)]
                    if (u, v) in edge_if:
                        mi, ni = edge_if[(u, v)]
                        nh = v
                key = (nh, mi, ni)
                if installed.get(u, {}).get(g) != key:
                    send(f"F {u},{g},{nh},{mi},{ni}")
                    installed.setdefault(u, {})[g] = key
                    n_sent += 1
        send("END")
        print(f"[server] t={t} deltas={n_sent}", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
