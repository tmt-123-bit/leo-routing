"""Single-change development probe: drop certainly-infeasible queued packets.

A packet is certainly infeasible when, even forwarding one hop per slot over the
currently available directed links, it cannot reach its destination by the later
of its class deadline and the episode horizon. Such packets can only end as
``deadline_exceeded`` drops or end-of-episode backlog; removing them at slot
boundaries frees queue positions and head-of-line service turns for packets
that can still be delivered. The purge never removes a packet that any feasible
forwarding schedule could deliver: BFS hop distance is an optimistic lower
bound on remaining slots and one extra slot of safety margin is applied.

The intervention is environment-side and is applied identically to every policy
arm (MAPPO constrained, Q-routing, persistent Dijkstra); no weights, rewards,
masks, admission, or switch accounting change. Arrival and physical event
pairing is asserted between paired base/purge episodes.
"""

import argparse
import csv
import gzip
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import (
    MULTIAGENT_LOADS,
    MultiAgentConfig,
    SynchronousLeoMultiAgentEnv,
)
from mappo_evaluation import evaluate_policy_with_constraint_metrics, load_checkpoint_policy
from run_development_load_diagnostics import (
    DiagnosticWrapper,
    PersistentDijkstraPolicy,
    aggregate,
    canonical,
    packet_accounting,
    read_json,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
import run_avoidable_switch_classical_baselines_formal as classical

LOADS = {"medium_load": 6, "hotspot_high_load": 16}
WORKLOAD_SEEDS = list(range(910041, 910051))
BASE_METHODS = ("qos_only_constrained", "q_routing", "persistent_global_dijkstra")
PURGE_SUFFIX = "_purge"

# packet_accounting treats the drop-reason set as a schema; extending it here is
# the documented way to introduce a reason inside a diagnostic probe.
if "early_infeasible" not in diagnostics.DROP_REASONS:
    diagnostics.DROP_REASONS = diagnostics.DROP_REASONS + ("early_infeasible",)


class PurgeInfeasibleEnv(SynchronousLeoMultiAgentEnv):
    """Remove certainly-undeliverable packets at slot boundaries."""

    drop_reason_name = "early_infeasible"
    safety_margin_slots = 1

    def __init__(self, cfg):
        super().__init__(cfg)
        self.purged_count = 0
        self.purged_by_class = [0, 0, 0]
        self.purged_by_bound = {"deadline": 0, "horizon": 0}

    def observe(self):
        self._purge_infeasible_packets()
        return super().observe()

    def _purge_infeasible_packets(self):
        if self.slot > self.cfg.episode_slots:
            return
        if not any(self.queues.values()):
            return
        self._refresh_graph()
        reverse = defaultdict(list)
        for (u, v), edge in self.graph.items():
            if u >= 1 and v >= 1 and edge.available:
                reverse[v].append(u)
        distances = {}
        for sat in range(1, self.n_agents + 1):
            for packet_id in self.queues[sat]:
                dst = self.packets[packet_id].dst
                if dst not in distances:
                    distances[dst] = self._bfs_distances(dst, reverse)
        for sat in range(1, self.n_agents + 1):
            if not self.queues[sat]:
                continue
            kept = deque()
            for packet_id in self.queues[sat]:
                packet = self.packets[packet_id]
                deadline = self.cfg.packet_class_deadlines[packet.traffic_class]
                deadline_bound = packet.created_slot + deadline - 1
                horizon_bound = self.cfg.episode_slots
                effective_bound = min(deadline_bound, horizon_bound)
                bound_source = "deadline" if deadline_bound <= horizon_bound else "horizon"
                remaining = effective_bound - self.slot
                dist = distances[packet.dst].get(sat)
                if dist is not None and dist > remaining + self.safety_margin_slots:
                    self._drop_packet(packet_id, self.drop_reason_name)
                    self.purged_count += 1
                    self.purged_by_class[packet.traffic_class] += 1
                    self.purged_by_bound[bound_source] += 1
                else:
                    kept.append(packet_id)
            self.queues[sat] = kept

    @staticmethod
    def _bfs_distances(dst: int, reverse):
        dist = {dst: 0}
        queue = deque([dst])
        while queue:
            node = queue.popleft()
            for prev in reverse[node]:
                if prev not in dist:
                    dist[prev] = dist[node] + 1
                    queue.append(prev)
        return dist


class PurgeDiagnosticWrapper(DiagnosticWrapper):
    def __init__(self, *args, purge: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if purge:
            self.env = PurgeInfeasibleEnv(self.env.cfg)


def run_episode(scenario, load, policy_name, policy, policy_seed, workload_seed, purge):
    initial, _ = MULTIAGENT_LOADS[scenario]
    cfg = MultiAgentConfig(
        env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[scenario]),
        initial_packets=initial, exogenous_packets_per_slot=load,
        seed=workload_seed, variant="qos_only",
    )
    wrapper = PurgeDiagnosticWrapper(scenario=scenario, cfg=cfg, purge=purge)
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
        purged_infeasible=int(getattr(wrapper.env, "purged_count", 0)),
    )
    if row["decision_switch_opportunities"] == 0:
        row["decision_avoidable_switch_rate"] = None
    if row["delivered"] == 0:
        row["average_delay_slots"] = row["p95_delay_slots"] = None
    if sum(row[f"drop_{r}"] for r in diagnostics.DROP_REASONS) != row["dropped"]:
        raise AssertionError("drop partition mismatch")
    return row, packets


def assess(rows):
    effects = []
    for scenario in LOADS:
        for method in BASE_METHODS:
            seeds = sorted({r["policy_seed"] for r in rows
                            if r["policy"] == method and r["scenario"] == scenario})
            seed_effects, delay_effects, hop_effects, new_rates = [], [], [], []
            paired_rows = []
            for seed in seeds:
                reference = {r["workload_seed"]: r for r in rows
                             if (r["scenario"], r["policy_seed"], r["policy"]) == (scenario, seed, method)}
                treatment = {r["workload_seed"]: r for r in rows
                             if (r["scenario"], r["policy_seed"], r["policy"]) == (scenario, seed, method + PURGE_SUFFIX)}
                if reference.keys() != treatment.keys():
                    raise AssertionError("unpaired workloads")
                delivery_diff = float(np.mean(
                    [treatment[w]["delivery_ratio"] - reference[w]["delivery_ratio"] for w in reference]))
                opp = sum(r["decision_switch_opportunities"] for r in treatment.values())
                rate = sum(r["decision_avoidable_switches"] for r in treatment.values()) / opp if opp else None
                seed_effects.append(delivery_diff)
                paired_rows.append({
                    "policy_seed": seed, "delivery_diff_pp": 100 * delivery_diff,
                    "treatment_switch_rate": rate,
                    "purged_infeasible": sum(r["purged_infeasible"] for r in treatment.values()),
                })
                if rate is not None:
                    new_rates.append(rate)
                for field, target in (("average_delay_slots", delay_effects),
                                      ("mean_delivered_hops", hop_effects)):
                    pairs = [(reference[w][field], treatment[w][field]) for w in reference
                             if reference[w][field] is not None and treatment[w][field] is not None]
                    if pairs:
                        before = np.mean([b for b, _ in pairs])
                        target.append(float(np.mean([a for _, a in pairs]) / before - 1))
            gates = {
                "every_seed_delivery_nondecreasing": all(d >= -1e-12 for d in seed_effects),
                "mean_delivery_positive": float(np.mean(seed_effects)) > 0,
                "constrained_purge_switch_rate_within_12pct": (
                    max(new_rates) <= 0.12 if method == "qos_only_constrained" else True),
                "mean_success_delay_increase_at_most_5pct": (
                    np.mean(delay_effects) <= 0.05 if delay_effects else True),
                "mean_success_hops_increase_at_most_1pct": (
                    np.mean(hop_effects) <= 0.01 if hop_effects else True),
            }
            effects.append({
                "scenario": scenario, "method": method, "paired_seeds": paired_rows,
                "delivery_diff_pp": 100 * float(np.mean(seed_effects)),
                "mean_success_delay_relative_change": float(np.mean(delay_effects)) if delay_effects else None,
                "mean_success_hops_relative_change": float(np.mean(hop_effects)) if hop_effects else None,
                "gates": {k: bool(v) for k, v in gates.items()},
                "development_screen_pass": bool(all(gates.values())),
                "paper_claim_allowed": False,
            })
    return effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="completed development-load-diagnostics output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    manifest = read_json(source / "manifest.json")
    completion = read_json(source / "completion.json")
    if completion["status"] != "complete" or sha(source / "manifest.json") != completion["output_sha256"]["manifest.json"]:
        raise ValueError("source manifest is not bound to a completed run")
    if set(WORKLOAD_SEEDS) & set(manifest["workload_seeds"]):
        raise ValueError("probe workloads overlap diagnostic development panel")
    if not (910001 <= WORKLOAD_SEEDS[0] and WORKLOAD_SEEDS[-1] <= 910100):
        raise ValueError("probe workloads must stay inside the 910001..910100 development panel")
    artifacts = [a for a in manifest["artifacts"] if a["method"] == "qos_only_constrained"]
    for artifact in artifacts:
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("frozen model changed")
    classical_freeze = read_json(diagnostics.CLASSICAL_ROOT / "training_freeze.json")
    classical_spec = read_json(diagnostics.CLASSICAL_ROOT / "preregistration.json")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "role": "single_change_development_probe",
        "source_manifest_sha256": sha(source / "manifest.json"),
        "runner_sha256": sha(Path(__file__)),
        "policy_seeds": manifest["policy_seeds"],
        "artifacts": artifacts, "workloads": WORKLOAD_SEEDS, "loads": LOADS,
        "intervention": (
            "environment-side purge of certainly-infeasible queued packets at slot "
            "boundaries: BFS hop distance over currently available directed links "
            "versus min(class deadline, episode horizon) remaining slots, one extra "
            "slot of safety margin; applied identically to every policy arm; no "
            "change to weights, masks, rewards, admission, or switch accounting"),
        "training": False, "sealed_test_access": False,
        "gates_predeclared": (
            "per scenario and method: all seed delivery differences >=0; mean >0; "
            "constrained purge arm switch rates <=0.12; seed-mean relative "
            "success-delay increase <=5%; success-hop increase <=1%"),
        "interpretation": (
            "development screening, no formal significance or promotion claim; "
            "purged packets are certain losses reclassified from future "
            "deadline_exceeded drops or end-of-episode backlog"),
    })
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, pairing = [], {}
    started = time.monotonic()
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as episode_file, \
                gzip.open(output / "packets.jsonl.gz", "wt", encoding="utf-8") as packet_file:
            for scenario, load in LOADS.items():
                for method in BASE_METHODS:
                    for seed in manifest["policy_seeds"] if method != "persistent_global_dijkstra" else [-1]:
                        if method == "qos_only_constrained":
                            artifact = next(a for a in artifacts if (a["scenario"], a["seed"]) == (scenario, seed))
                            policy, _ = load_checkpoint_policy(Path(artifact["path"]), device="cpu")
                        elif method == "q_routing":
                            job = next(j for j in classical.build_training_jobs()
                                       if j.scenario == scenario and j.policy_seed == seed)
                            entry = classical_freeze["jobs"][job.job_id]["model"]
                            policy, _ = classical.load_q_model(Path(entry["path"]), job, classical_spec, require_frozen=True)
                        else:
                            policy = PersistentDijkstraPolicy()
                        for purge in (False, True):
                            label = method + (PURGE_SUFFIX if purge else "")
                            for workload in WORKLOAD_SEEDS:
                                row, packets = run_episode(scenario, load, label, policy, seed, workload, purge)
                                key = (scenario, workload)
                                events = (row["arrival_sha256"], row["physical_sha256"])
                                if key in pairing:
                                    if pairing[key]["events"] != events:
                                        raise AssertionError("exogenous pairing changed")
                                    pairing[key]["arms"] += 1
                                else:
                                    pairing[key] = {"events": events, "arms": 1}
                                episode_file.write(canonical(row) + "\n")
                                identity = {k: row[k] for k in ("scenario", "load", "policy", "policy_seed", "workload_seed")}
                                for packet in packets:
                                    packet_file.write(canonical({**identity, **packet}) + "\n")
                                rows.append(row)
                            episode_file.flush()
                            print(f"{len(rows)} {scenario} {label} seed={seed} elapsed={time.monotonic()-started:.1f}s", flush=True)
        for artifact in artifacts:
            if sha(artifact["path"]) != artifact["sha256"]:
                raise AssertionError("frozen input changed")
        if any(entry["arms"] != 14 for entry in pairing.values()):
            raise AssertionError("unexpected arm coverage in pairing")
        summaries, effects = aggregate(rows), assess(rows)
        write_json(output / "summary.json", summaries)
        write_json(output / "paired_effects.json", effects)
        with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
            fields = ["scenario", "policy", "delivery_ratio", "drop_rate", "backlog_rate", "decision_switch_rate"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
                   "exogenous_pairings_verified": len(pairing), "elapsed_seconds": time.monotonic() - started,
                   "development_screen_pass": all(e["development_screen_pass"] for e in effects),
                   "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
        print(canonical(effects), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error), "episodes": len(rows)})
        raise


if __name__ == "__main__":
    main()
