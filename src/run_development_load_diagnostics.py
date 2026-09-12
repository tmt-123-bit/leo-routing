"""Frozen-policy development diagnostics; never trains or opens sealed workloads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from mappo_evaluation import (
    GlobalDijkstraPolicy,
    evaluate_policy_with_constraint_metrics,
    load_checkpoint_policy,
)
import run_avoidable_switch_classical_baselines_formal as classical


ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = ROOT / "experiments/avoidable-switch-constraint-formal-v1-r2"
CLASSICAL_ROOT = ROOT / "experiments/avoidable-switch-classical-baselines-formal-v1-r3"
DROP_REASONS = (
    "no_route", "invalid_action", "ttl_exceeded", "queue_overflow",
    "source_queue_overflow", "deadline_exceeded",
)
METHODS = ("qos_only_constrained", "qos_only_baseline", "q_routing",
           "global_dijkstra", "persistent_global_dijkstra")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def choose_persistent_action(old_neighbor, neighbors, mask, fallback):
    if old_neighbor in neighbors:
        action = neighbors.index(old_neighbor) + 1
        if mask[action]:
            return action
    return int(fallback)


class PersistentDijkstraPolicy(GlobalDijkstraPolicy):
    """Keep the cached feasible next hop; otherwise use the existing global oracle."""

    def __call__(self, observation, mask):
        actions = super().__call__(observation, mask)
        for external, sat in enumerate(self.wrapper.external_to_internal):
            obs = self.wrapper._obs[sat - 1]
            if obs["hol_packet_id"] is None:
                continue
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            old = self.wrapper.env.route_cache.get((sat, packet.dst, packet.traffic_class))
            actions[external] = choose_persistent_action(
                old, obs["neighbor_ids"], mask[external], actions[external],
            )
        return actions


class DiagnosticWrapper(CleanMARLLeoMultiAgentWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physical_digest = hashlib.sha256()
        self.decisions = []
        self.accepted_count = 0
        self.blocked_count = 0
        self.no_route_masks = Counter()

    def step(self, actions):
        env = self.env
        # Exclude endogenous load/reliability so this checks exogenous pairing only.
        physical = [(u, v, edge.delay_ms, edge.capacity_mbps, edge.t_rem, edge.available)
                    for (u, v), edge in sorted(env.graph.items())]
        self.physical_digest.update(canonical([env.slot, physical]).encode())
        before = []
        for external, sat in enumerate(self.external_to_internal):
            obs = self._obs[sat - 1]
            if obs["hol_packet_id"] is None:
                continue
            packet = env.packets[obs["hol_packet_id"]]
            old = env.route_cache.get((sat, packet.dst, packet.traffic_class))
            before.append({
                "slot": env.slot, "sat": sat, "packet_id": packet.packet_id,
                "destination": packet.dst, "old_next_hop": old,
                "action": int(actions[external]), "neighbors": obs["neighbor_ids"],
                "mask": obs["action_mask"], "mask_reasons": obs["mask_reasons"],
                "queue": len(env.queues[sat]),
                "neighbor_queues": [len(env.queues[n]) if n else None for n in obs["neighbor_ids"]],
                "neighbor_previous_queues": [env.previous_queue_lengths[n] if n else None
                                             for n in obs["neighbor_ids"]],
                "packet_age_slots": env.slot - packet.created_slot,
                "hop_count": packet.hop_count,
            })
            if not any(obs["action_mask"][1:]):
                self.no_route_masks.update(r for r in obs["mask_reasons"][1:] if r != "padding")
        result = super().step(actions)
        info = result[-1]
        self.accepted_count += len(info["accepted"])
        self.blocked_count += len(info["blocked"])
        for row in before:
            index = row["sat"] - 1
            row.update(
                accepted=row["packet_id"] in info["accepted"],
                blocked=row["packet_id"] in info["blocked"],
                opportunity=bool(info["decision_switch_opportunities"][index]),
                avoidable_switch=bool(info["decision_avoidable_switch_costs"][index]),
                forced_switch=bool(info["decision_forced_switches"][index]),
            )
        self.decisions.extend(before)
        return result


def packet_accounting(env):
    env.validate_invariants()
    if set(env.drop_reasons) != env.dropped:
        raise AssertionError("drop reason coverage differs from dropped packet IDs")
    if set(env.drop_reasons.values()) - set(DROP_REASONS):
        raise AssertionError("new drop reason requires explicit diagnostic schema update")
    queued = env._backlog_ids()
    cutoff = env.slot - 1
    packets = []
    for packet_id in sorted(env.generated):
        packet = env.packets[packet_id]
        status = "delivered" if packet_id in env.delivered else "dropped" if packet_id in env.dropped else "queued"
        packets.append({
            "packet_id": packet_id, "src": packet.src, "dst": packet.dst,
            "traffic_class": packet.traffic_class, "created_slot": packet.created_slot,
            "status": status, "drop_reason": env.drop_reasons.get(packet_id),
            "terminal_slot": env.delivery_slots.get(packet_id, env.drop_slots.get(packet_id)),
            "cutoff_age_slots": cutoff - packet.created_slot + 1,
            "hop_count": packet.hop_count, "owner": packet.owner,
            "repeated_visits": len(packet.visited) - len(set(packet.visited)),
        })
    arrivals = [(p["packet_id"], p["src"], p["dst"], p["traffic_class"], p["created_slot"]) for p in packets]
    counts = Counter(env.drop_reasons.values())
    ages = [cutoff - env.packets[p].created_slot + 1 for p in queued]
    generated_last_slot = sum(p["created_slot"] == cutoff for p in packets)
    return {
        **{f"drop_{reason}": counts[reason] for reason in DROP_REASONS},
        "arrival_sha256": hashlib.sha256(canonical(arrivals).encode()).hexdigest(),
        "backlog_final_slot_arrivals": sum(env.packets[p].created_slot == cutoff for p in queued),
        "generated_final_slot": generated_last_slot,
        "backlog_age_mean": float(np.mean(ages)) if ages else None,
        "backlog_age_max": max(ages) if ages else None,
        "backlog_age_at_least_5": sum(age >= 5 for age in ages),
        "mean_delivered_hops": float(np.mean([env.packets[p].hop_count for p in env.delivered])) if env.delivered else None,
        "packets_with_repeated_visits": sum(p["repeated_visits"] > 0 for p in packets),
    }, packets


def make_wrapper(scenario, load, workload_seed, diagnostic=True):
    initial, _ = MULTIAGENT_LOADS[scenario]
    cfg = MultiAgentConfig(
        env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[scenario]),
        initial_packets=initial, exogenous_packets_per_slot=load,
        seed=workload_seed, variant="qos_only",
    )
    wrapper_type = DiagnosticWrapper if diagnostic else CleanMARLLeoMultiAgentWrapper
    return wrapper_type(scenario=scenario, cfg=cfg)


def run_episode(scenario, load, policy_name, policy, policy_seed, workload_seed):
    wrapper = make_wrapper(scenario, load, workload_seed)
    result = evaluate_policy_with_constraint_metrics(
        scenario, policy_name, policy, policy_seed, [workload_seed],
        wrapper_factory=lambda _: wrapper, variant="qos_only",
    )[0]
    row = asdict(result)
    extra, packets = packet_accounting(wrapper.env)
    row.update(extra)
    row.update(
        load=load, initial_packets=wrapper.env.cfg.initial_packets,
        physical_sha256=wrapper.physical_digest.hexdigest(),
        accepted_forwards=wrapper.accepted_count, blocked_proposals=wrapper.blocked_count,
        no_route_mask_reasons=dict(wrapper.no_route_masks),
    )
    if row["decision_switch_opportunities"] == 0:
        row["decision_avoidable_switch_rate"] = None
    if row["delivered"] == 0:
        row["average_delay_slots"] = row["p95_delay_slots"] = None
    if sum(row[f"drop_{r}"] for r in DROP_REASONS) != row["dropped"]:
        raise AssertionError("drop partition mismatch")
    return row, packets, wrapper.decisions


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["load"], row["policy"])].append(row)
    result = []
    for (scenario, load, policy), items in sorted(groups.items()):
        by_seed = defaultdict(list)
        for row in items:
            by_seed[row["policy_seed"]].append(row)
        seed_rates = []
        per_seed = []
        for seed, values in sorted(by_seed.items()):
            opp = sum(v["decision_switch_opportunities"] for v in values)
            switches = sum(v["decision_avoidable_switches"] for v in values)
            rate = switches / opp if opp else None
            seed_rates.append(rate)
            per_seed.append({"policy_seed": seed, "delivery_ratio": float(np.mean([v["delivery_ratio"] for v in values])),
                             "switches": switches, "opportunities": opp, "rate": rate})
        totals = {field: sum(v[field] for v in items) for field in (
            "generated", "delivered", "dropped", "backlog", "generated_final_slot",
            "backlog_final_slot_arrivals", "decision_avoidable_switches", "decision_switch_opportunities",
            "decision_forced_switches", "accepted_forwards", "blocked_proposals",
            "backlog_age_at_least_5", *(f"drop_{r}" for r in DROP_REASONS),
        )}
        defined = all(rate is not None for rate in seed_rates)
        result.append({
            "scenario": scenario, "load": load, "policy": policy, "episodes": len(items),
            "policy_replicates": len(by_seed), "totals": totals, "per_seed": per_seed,
            "delivery_ratio": float(np.mean([v["delivery_ratio"] for v in per_seed])),
            "drop_rate": totals["dropped"] / totals["generated"],
            "backlog_rate": totals["backlog"] / totals["generated"],
            "decision_switch_rate": float(np.mean(seed_rates)) if defined else None,
            "all_observed_seed_rates_within_budget": all(rate <= 0.12 for rate in seed_rates) if defined else None,
            "descriptive_only": True,
        })
    return result


def plot_summary(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = dict(zip(METHODS, ("Constrained MAPPO", "QoS-only MAPPO", "Q-routing", "Dijkstra", "Persistent Dijkstra")))
    for col, scenario in enumerate(("medium_load", "hotspot_high_load")):
        for method in METHODS:
            points = sorted((r for r in summary if r["scenario"] == scenario and r["policy"] == method), key=lambda r: r["load"])
            if not points:
                continue
            axes[0, col].plot([r["load"] for r in points], [100 * r["delivery_ratio"] for r in points], "o-", label=labels[method])
            axes[1, col].plot([r["load"] for r in points], [100 * r["decision_switch_rate"] if r["decision_switch_rate"] is not None else np.nan for r in points], "o-")
        axes[0, col].set_title(scenario)
        axes[0, col].axhline(95, color="gray", linestyle=":")
        axes[1, col].axhline(12, color="gray", linestyle=":")
        axes[0, col].set_ylim(0, 100)
        axes[0, col].set_ylabel("Delivery (%)")
        axes[1, col].set_ylabel("Avoidable decision switches (%)")
        axes[1, col].set_xlabel("New packets per slot (fixed initial packets)")
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Frozen-policy development diagnostics (not a new sealed test)")
    fig.savefig(output / "load_diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loads", nargs="+", type=int, default=[2, 4, 6, 8, 16])
    parser.add_argument("--policy-seeds", type=int, default=3)
    parser.add_argument("--workloads", type=int, default=10)
    parser.add_argument("--workload-start", type=int, default=910001)
    args = parser.parse_args()
    if not 1 <= args.policy_seeds <= 8 or not 1 <= args.workloads <= 10:
        parser.error("development run supports 1..8 model seeds and 1..10 workloads")
    if args.workload_start < 910001 or args.workload_start + args.workloads > 910101:
        parser.error("only the dedicated 910001..910100 development panel is permitted")
    if len(set(args.loads)) != len(args.loads) or any(not 1 <= n <= 24 for n in args.loads):
        parser.error("unique load points must be between 1 and the 24 available sources")
    if os.environ.get("LEO_REWARD_OVERRIDES"):
        parser.error("unset LEO_REWARD_OVERRIDES for frozen diagnostics")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    method_freeze = read_json(METHOD_ROOT / "training_freeze.json")
    classical_freeze = read_json(CLASSICAL_ROOT / "training_freeze.json")
    classical_spec = read_json(CLASSICAL_ROOT / "preregistration.json")
    jobs = method_freeze["jobs"]
    seed_sets = [
        {entry["job"]["policy_seed"] for entry in jobs.values()
         if entry["job"]["scenario"] == scenario and entry["job"]["arm"] == arm}
        for scenario in ("medium_load", "hotspot_high_load")
        for arm in METHODS[:2]
    ]
    seeds = sorted(set.intersection(*seed_sets))[:args.policy_seeds]
    if len(seeds) != args.policy_seeds:
        raise ValueError("insufficient common frozen training seeds")
    artifacts = []
    for scenario in ("medium_load", "hotspot_high_load"):
        for method in METHODS[:3]:
            for seed in seeds:
                key = f"{scenario}/{method}/seed_{seed}"
                entry = jobs[key]["artifacts"]["selected_checkpoint"] if method != "q_routing" else classical_freeze["jobs"][key]["model"]
                if sha(entry["path"]) != entry["sha256"]:
                    raise ValueError(f"frozen model hash mismatch: {key}")
                artifacts.append({"scenario": scenario, "method": method, "seed": seed, **entry})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = sorted(p for p in (ROOT / "src").glob("*.py") if not p.name.startswith("test_"))
    manifest = {
        "role": "development_diagnostics", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "artifacts": artifacts, "loads": args.loads, "policy_seeds": seeds,
        "workload_seeds": list(range(args.workload_start, args.workload_start + args.workloads)),
        "methods": list(METHODS), "training": False, "sealed_test_access": False,
        "environment_variant": "qos_only", "action_selection": "checkpoint argmax; frozen Q greedy",
        "configs": {s: asdict(make_wrapper(s, args.loads[0], args.workload_start).env.cfg)
                    for s in ("medium_load", "hotspot_high_load")},
        "persistent_rule": "keep cached feasible (node,dst,class) next hop; otherwise global Dijkstra; not ILPR reproduction",
        "switch_aggregation": "ratio of summed decision counts within each model seed, then equal seed mean; zero denominator undefined",
        "trace_sampling": "first workload at every load, method and policy seed",
        "torch": torch.__version__, "numpy": np.__version__,
    }
    write_json(output / "manifest.json", manifest)
    rows, pairings = [], {}
    started = time.monotonic()
    expected = 2 * len(args.loads) * args.workloads * (3 * len(seeds) + 2)
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as episode_file, \
                gzip.open(output / "packets.jsonl.gz", "wt", encoding="utf-8") as packet_file, \
                gzip.open(output / "decisions_sample.jsonl.gz", "wt", encoding="utf-8") as decision_file:
            for scenario in ("medium_load", "hotspot_high_load"):
                for method in METHODS:
                    for seed in seeds if method in METHODS[:3] else [-1]:
                        if method in METHODS[:2]:
                            artifact = next(a for a in artifacts if (a["scenario"], a["method"], a["seed"]) == (scenario, method, seed))
                            policy, _ = load_checkpoint_policy(Path(artifact["path"]), device="cpu")
                        elif method == "q_routing":
                            job = next(j for j in classical.build_training_jobs() if j.scenario == scenario and j.policy_seed == seed)
                            entry = classical_freeze["jobs"][job.job_id]["model"]
                            policy, _ = classical.load_q_model(Path(entry["path"]), job, classical_spec, require_frozen=True)
                        else:
                            policy = GlobalDijkstraPolicy() if method == "global_dijkstra" else PersistentDijkstraPolicy()
                        q_before = hashlib.sha256(policy.q.tobytes()).hexdigest() if method == "q_routing" else None
                        for load in args.loads:
                            for workload in manifest["workload_seeds"]:
                                row, packets, decisions = run_episode(scenario, load, method, policy, seed, workload)
                                key = (scenario, load, workload)
                                signature = (row["arrival_sha256"], row["physical_sha256"])
                                if key in pairings and pairings[key] != signature:
                                    raise AssertionError(f"exogenous event pairing failed: {key}")
                                pairings[key] = signature
                                identity = {k: row[k] for k in ("scenario", "load", "policy", "policy_seed", "workload_seed")}
                                episode_file.write(canonical(row) + "\n")
                                for packet in packets:
                                    packet_file.write(canonical({**identity, **packet}) + "\n")
                                if workload == args.workload_start:
                                    for decision in decisions:
                                        decision_file.write(canonical({**identity, **decision}) + "\n")
                                rows.append(row)
                            episode_file.flush()
                            print(f"{len(rows)}/{expected} {scenario} {method} seed={seed} load={load} elapsed={time.monotonic()-started:.1f}s", flush=True)
                        if q_before is not None and hashlib.sha256(policy.q.tobytes()).hexdigest() != q_before:
                            raise AssertionError("Q-routing changed during frozen evaluation")
        for artifact in artifacts:
            if sha(artifact["path"]) != artifact["sha256"]:
                raise AssertionError("input model changed during diagnostics")
        for relative, digest in manifest["source_sha256"].items():
            if sha(ROOT / relative) != digest:
                raise AssertionError("source changed during diagnostics")
        summary = aggregate(rows)
        write_json(output / "summary.json", summary)
        with (output / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
            fields = ["scenario", "load", "policy", "episodes", "policy_replicates", "delivery_ratio", "drop_rate", "backlog_rate", "decision_switch_rate", "all_observed_seed_rates_within_budget"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
        plot_summary(output, summary)
        write_json(output / "completion.json", {"status": "complete", "episodes": len(rows), "expected": expected,
                   "exogenous_pairings_verified": len(pairings), "input_hashes_unchanged": True,
                   "elapsed_seconds": time.monotonic() - started,
                   "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "completed_episodes": len(rows), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
