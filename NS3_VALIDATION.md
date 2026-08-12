# ns-3 packet-level validation

**Goal.** Test whether MAPPO's congestion-awareness advantage (observed in the
slot-synchronous training env) survives at packet fidelity in a standards-grade
simulator. We replay the *exact* per-packet source-routed paths produced by each
policy in the env through an ns-3.48 point-to-point torus mesh with real FIFO
drop-tail queues, byte-accurate bandwidth, and propagation delay. Both policies
run on the **identical** physical model and traffic, so the only variable is the
path choice.

**Harness.**
- `ns3_trace_extractor.py` — drives 5 workload seeds (21001–21005) of `medium_load`
  through each policy, dumps every packet's full node path (`packet.visited`).
- `ns3_leo_validation.cc` (ns-3.48 scratch scenario) — 24-node 4×6 torus, custom
  `LeoHeader` source-routing, forwarding via non-promiscuous protocol handlers,
  DropTailQueue. Offered load swept by compressing the env slot duration
  (`--slot-sec` ∈ {0.5, 0.25, 0.125, 0.0625} ⇒ 1×/2×/4×/8×), ISL fixed at 36 kb/s
  (~ env 3-pkt/slot capacity), queue 8 packets.

**Figure.** `figures/fig_ns3_validation.{png,pdf}` (also `make_ns3_figure.py`).
- *Left:* delivery vs offered load (ns-3, solid + 95% bootstrap-CI bands over 5
  episodes) and env ground-truth reference (dashed). *Right:* P95 delay vs load.

## The honest result

| metric | env (slot-sync) | ns-3 @4× | ns-3 @8× |
|---|---|---|---|
| MAPPO delivery | **0.783** [0.769, 0.798] | 0.811 [0.776, 0.858] | 0.663 [0.588, 0.753] |
| Dijkstra delivery | 0.719 [0.686, 0.752] | 0.838 [0.799, 0.895] | 0.684 [0.615, 0.774] |
| **ordering** | **MAPPO > Dijkstra (+6.4pp, CI-separated)** | **Dijkstra ≈ MAPPO (overlap)** | **Dijkstra ≈ MAPPO (overlap)** |

**The MAPPO advantage does not transfer to packet-level static topology.** In the
env, MAPPO beats Dijkstra by +6.4 pp with non-overlapping CIs. In ns-3, Dijkstra
edges ahead at every load level; at 4× and 8× the CIs overlap (a statistical tie).
Mean path lengths are nearly identical (MAPPO 2.58, Dijkstra 2.50 hops), so the
two policies choose structurally similar routes here.

## Why (root cause)

The validation is *faithful but adversarial to MAPPO*: it removes exactly the two
mechanisms that give MAPPO its edge in the slot model:

1. **Static topology.** ns-3 replays on a fixed 4×6 torus; ISLs never fail or
   hand off. Dijkstra is *optimal* on a static graph, so there is little room for
   a learned policy to beat it on pure path quality.
2. **Continuous time.** ns-3 injects at `created_slot × slot_sec` in continuous
   seconds, which **desynchronizes** the per-slot link contention that is the
   env's core congestion signal. Packets spread out across 48 links instead of
   colliding in synchronized slots, so congestion is dilute and the load-
   balancing value of MAPPO's multi-path choices largely evaporates.

MAPPO's advantage lives in **slot-synchronized contention on a dynamic topology**
— both absent from this static continuous-time replay.

## What this bounds

This is a **scope-bounding negative result**, reported honestly. It does **not**
refute the env result (which is CI-separated and reproducible). It says: the
congestion-awareness win is a property of the slot-synchronized dynamic regime,
and does not manifest as a delivery-ratio win under static-topology packet-level
replay. The claim in the results package is therefore scoped to the slot model;
extending it to packet-level static topology is **not** supported by this test.

To recover a packet-level win, the ns-3 model would need (a) dynamic topology
(moving ISLs / link failures) and/or (b) slot-synchronized batch injection that
preserves the env's contention structure. Both are out of scope for this package
and left as future work.

## Reproduce

```bash
# 1. extract per-policy paths (env ground truth + paths)
python ns3_trace_extractor.py --checkpoint <no_lifetime>/validation_best.pt \
    --outdir experiments/IEEE-NS3 --workload-seeds 21001,21002,21003,21004,21005

# 2. ns-3 offered-load sweep (WSL, ns-3.48, as nsuser)
bash run_ns3_sweep.sh 36 8

# 3. figure
python make_ns3_figure.py --indir experiments/IEEE-NS3 --outdir figures
```

ns-3.48 is © 2009–2024 UCL / Nsnam, CC-BY-SA 3.0 (https://www.nsnam.org).
