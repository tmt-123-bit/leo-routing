"""Extract per-packet path traces from the LEO env for ns-3 packet-level replay.

For each policy (a trained MAPPO checkpoint + the Dijkstra oracle), run N episodes
on medium_load (no_lifetime), and dump every generated packet's source-routing
path (the sequence of nodes the policy forwarded it through). ns-3 then replays
those EXACT paths through a packet-level model (real FIFO drop-tail queues, byte
bandwidth, propagation delay), so we can test whether the slot-sim's ordering
(MAPPO vs Dijkstra) and fairness survive at packet fidelity.

Path convention: full path = [src] + packet.visited . For env-dropped packets the
path is partial (they never reached dst) -> ns-3 replays the partial path and they
simply don't arrive (consistent with the env). Delivered packets end at dst.

Outputs (args.outdir / experiments/IEEE-NS3/):
  topology.csv            u,v,is_cross,delay_ms  (static 4x6 torus, avg delays)
  packets_<policy>.csv    episode,packet_id,src,dst,class,created_slot,delivered,
                          delivery_slot,hop_count,delay_ms,path,drop_reason
  links_<policy>.csv      episode,slot,u,v  (per-slot DOWN directed links; only
                          non-empty for break scenarios e.g. frequent_break.
                          Reconstructed post-episode via base._build_topology(t),
                          which is deterministic in t given the seed-initialized
                          fault sets -> exactly the availability the policy saw)
  env_summary_<policy>.csv  per-episode slot-sim numbers (ground truth to compare)

main() prints the episode horizon (cfg.episode_slots) and the per-class deadline
slots (cfg.packet_class_deadlines) so the ns-3 run gets matching --episode-slots
/ --deadline-slots arguments.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Callable

import numpy as np

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_evaluation import GlobalDijkstraPolicy, load_checkpoint_policy

N_PLANES, N_SATS_PER_PLANE = 4, 6
TRAFFIC_SLOT_SECONDS = 1.0  # 1 env slot -> 1.0 ns-3 second (tune in the ns-3 model)


def decode(node_id: int) -> tuple[int, int]:
    """(plane, pos), 1-indexed."""
    plane = (node_id - 1) // N_SATS_PER_PLANE + 1
    pos = (node_id - 1) % N_SATS_PER_PLANE + 1
    return plane, pos


def is_cross_plane(u: int, v: int) -> int:
    return 0 if decode(u)[0] == decode(v)[0] else 1


def dump_topology(env, path: Path) -> None:
    """Static undirected torus adjacency with average per-link delay (ms)."""
    seen = set()
    rows = []
    for (u, v) in env.graph.keys():
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        cross = is_cross_plane(u, v)
        rows.append({"u": key[0], "v": key[1], "is_cross": cross,
                     "delay_ms": 12.0 if cross else 8.0})
    rows.sort(key=lambda r: (r["u"], r["v"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["u", "v", "is_cross", "delay_ms"])
        w.writeheader()
        w.writerows(rows)
    return None


def make_wrapper_factory(variant: str, scenario: str) -> Callable[[int], CleanMARLLeoMultiAgentWrapper]:
    """IMPORTANT: both policies must run under the SAME env variant. The action
    masks constrain every policy (Dijkstra picks among mask-feasible actions
    too), so a mask-variant mismatch between MAPPO and its baseline silently
    handicaps one side — e.g. running Dijkstra under variant 'full' applies the
    lifetime mask to it while a no_lifetime MAPPO runs unmasked."""

    ini, exo = MULTIAGENT_LOADS[scenario]
    def make(seed: int) -> CleanMARLLeoMultiAgentWrapper:
        envc = EnvConfig(scenario=SCENARIOS[scenario], seed=seed)
        cfg = MultiAgentConfig(env=envc, initial_packets=ini, exogenous_packets_per_slot=exo,
                               seed=seed, variant=variant)
        return CleanMARLLeoMultiAgentWrapper(cfg=cfg)
    return make


def run_episode(wrapper, policy, seed: int) -> None:
    """Drive one episode to completion with `policy`; leaves env populated."""
    obs, _ = wrapper.reset(seed=seed)
    term = trunc = False
    info = {}
    while not term and not trunc:
        act = policy(obs, wrapper.get_avail_actions())
        obs, _r, term, trunc, info = wrapper.step(act)
        if hasattr(policy, "observe_transition") and callable(policy.observe_transition):
            policy.observe_transition(info)


def dump_policy_traces(policy, policy_name: str, variant: str, scenario: str,
                       workload_seeds: list[int], outdir: Path) -> None:
    fct = make_wrapper_factory(variant, scenario)
    pkt_rows, ep_rows, link_rows = [], [], []
    binder = getattr(policy, "bind", None)
    for ep, wl in enumerate(workload_seeds):
        wrapper = fct(wl)
        if binder is not None:
            binder(wrapper)
        run_episode(wrapper, policy, wl)
        env = wrapper.env
        # Reconstruct the per-slot link-availability schedule the policy actually
        # faced (availability is deterministic in t given the per-seed fault sets;
        # used_rate feeds only features, not `available`). Dump DOWN entries only.
        horizon = env.cfg.episode_slots
        for s in range(1, horizon + 1):
            topo = env.base._build_topology(s)
            for (u, v), link in sorted(topo.items()):
                if not link.available:
                    link_rows.append({"episode": ep, "slot": s, "u": u, "v": v})
        for pid in sorted(env.generated):
            p = env.packets[pid]
            delivered = int(pid in env.delivered)
            dslot = env.delivery_slots.get(pid, -1)
            delay_slots = (dslot - p.created_slot + 1) if delivered else -1
            full_path = list(p.visited)  # visited already starts with src (line 935)
            pkt_rows.append({
                "episode": ep, "packet_id": pid, "src": p.src, "dst": p.dst,
                "traffic_class": p.traffic_class, "created_slot": p.created_slot,
                "delivered": delivered, "delivery_slot": dslot, "hop_count": p.hop_count,
                "delay_slots": delay_slots,
                "delay_ms": round(p.cumulative_link_delay_ms, 2),
                "path": " ".join(map(str, full_path)),
                "drop_reason": env.drop_reasons.get(pid, ""),
            })
        slots = max(1, env.slot - 1)
        delays = [env.delivery_slots[pid] - env.packets[pid].created_slot + 1
                  for pid in env.delivered]
        ep_rows.append({
            "episode": ep, "policy": policy_name, "workload_seed": wl,
            "generated": len(env.generated), "delivered": len(env.delivered),
            "dropped": len(env.dropped),
            "delivery_ratio": round(len(env.delivered) / max(1, len(env.generated)), 4),
            "throughput_per_slot": round(len(env.delivered) / slots, 4),
            "mean_delay_slots": round(float(np.mean(delays)), 3) if delays else 0.0,
            "p95_delay_slots": round(float(np.percentile(delays, 95)), 3) if delays else 0.0,
        })
        wrapper.close()
        print(f"  [{policy_name}] ep{ep} wl={wl}: gen={len(env.generated)} "
              f"del={len(env.delivered)} ratio={len(env.delivered)/max(1,len(env.generated)):.3f}",
              flush=True)

    pk = outdir / f"packets_{policy_name}.csv"
    with open(pk, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pkt_rows[0].keys()))
        w.writeheader(); w.writerows(pkt_rows)
    sk = outdir / f"env_summary_{policy_name}.csv"
    with open(sk, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
        w.writeheader(); w.writerows(ep_rows)
    lk = outdir / f"links_{policy_name}.csv"
    with open(lk, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "slot", "u", "v"])
        w.writeheader(); w.writerows(link_rows)
    print(f"=> wrote {pk} ({len(pkt_rows)} packets) + {sk} ({len(ep_rows)} eps) "
          f"+ {lk} ({len(link_rows)} down-link entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="MAPPO validation_best.pt (no_lifetime)")
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--outdir", type=Path, default=Path("experiments/IEEE-NS3"))
    ap.add_argument("--workload-seeds", default="21001,21002,21003,21004,21005")
    ap.add_argument("--variant", default="no_lifetime",
                    help="env variant for BOTH policies (mask parity)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    wl = [int(s) for s in args.workload_seeds.split(",")]

    # topology (use a throwaway wrapper to access the graph; must reset to populate)
    tw = make_wrapper_factory(args.variant, args.scenario)(wl[0])
    tw.reset(seed=wl[0])
    dump_topology(tw.env, args.outdir / "topology.csv")
    # horizon + per-class deadline slots -> pass to the ns-3 run as
    # --episode-slots / --deadline-slots (must match the env clock exactly)
    print(f"=> episode_slots={tw.env.cfg.episode_slots} "
          f"packet_class_deadlines={tw.env.cfg.packet_class_deadlines}")
    tw.close()
    print(f"=> wrote topology.csv ({sum(1 for _ in open(args.outdir/'topology.csv'))-1} links)")

    # MAPPO (checkpoint of the adopted variant)
    print(f"== MAPPO ({args.checkpoint.name}) ==", flush=True)
    policy, _info = load_checkpoint_policy(args.checkpoint, device=args.device)
    dump_policy_traces(policy, "mappo", args.variant, args.scenario, wl, args.outdir)

    # Dijkstra oracle (SAME variant as MAPPO — mask parity, see factory note)
    print("== Dijkstra ==", flush=True)
    dump_policy_traces(GlobalDijkstraPolicy(), "dijkstra", args.variant, args.scenario, wl, args.outdir)
    print("DONE")


if __name__ == "__main__":
    main()
