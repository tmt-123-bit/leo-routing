"""Paired development probes of local budgeting and frozen-policy decision splits."""

import argparse
import gzip
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from local_switch_budget_policy import LocalSwitchBudgetPolicy
from decision_split_policy import DecisionSplitPolicy
from mappo_evaluation import load_checkpoint_policy
from run_development_load_diagnostics import (
    ROOT, METHOD_ROOT, CLASSICAL_ROOT, PersistentDijkstraPolicy, aggregate,
    canonical, classical, read_json, run_episode, sha, write_json,
)
from run_direct_delivery_controls import DirectDeliveryControlPolicy, crossed_interval, matrix, row_index
from run_direct_delivery_probe import LOADS

SEEDS = [179055553, 183895110, 310818925, 515636025, 626669596, 746965870, 985998595, 2136406109]
METHODS = {
    "original_constrained": ("qos_only_constrained", "original"),
    "direct_constrained": ("qos_only_constrained", "direct"),
    "direct_qos": ("qos_only_baseline", "direct"),
    "budget_qos": ("qos_only_baseline", "budget"),
    "budget_q": ("q_routing", "budget"),
    "direct_persistent": ("persistent_global_dijkstra", "direct"),
}
SPLIT_METHODS = {
    "original_constrained": ("qos_only_constrained", "original"),
    "direct_constrained": ("qos_only_constrained", "direct"),
    "direct_qos": ("qos_only_baseline", "direct"),
    "direct_q": ("q_routing", "direct"),
    "split_qos": ("qos_only_constrained", "split_qos"),
    "split_q": ("qos_only_constrained", "split_q"),
    "direct_persistent": ("persistent_global_dijkstra", "direct"),
}


def assess(rows, seeds, workloads, candidates=("budget_qos", "budget_q")):
    index = row_index(rows)
    results = []
    for scenario in LOADS:
        for candidate in candidates:
            for reference in ("original_constrained", "direct_constrained", "direct_persistent"):
                get = lambda name, field: matrix(index, scenario, name, seeds, workloads, field)
                generated = get(candidate, "generated")
                if not np.array_equal(generated, get(reference, "generated")):
                    raise AssertionError("unequal packet denominators")
                effect = 100 * (get(candidate, "delivered") - get(reference, "delivered")) / generated
                if not np.all(generated == generated[0, 0]):
                    raise AssertionError("this fixed-load screen expects equal episode packet counts")
                seed_packet_differences = (get(candidate, "delivered") - get(reference, "delivered")).sum(axis=1)
                seed_effect = 100 * seed_packet_differences / generated.sum(axis=1)
                opportunities = get(candidate, "decision_switch_opportunities").sum(axis=1)
                if np.any(opportunities == 0):
                    raise AssertionError("undefined switch rate")
                rates = get(candidate, "decision_avoidable_switches").sum(axis=1) / opportunities
                relative = {f: float(np.mean(get(candidate, f).mean(axis=1) / get(reference, f).mean(axis=1) - 1))
                            for f in ("average_delay_slots", "mean_delivered_hops")}
                checks = {
                    "every_seed_delivery_nondecreasing": bool(np.all(seed_packet_differences >= 0)),
                    "mean_delivery_positive": bool(seed_packet_differences.sum() > 0),
                    "every_seed_rate_within_12pct": bool(np.all(rates <= .12)),
                    "mean_success_delay_increase_at_most_5pct": relative["average_delay_slots"] <= .05,
                    "mean_success_hops_increase_at_most_1pct": relative["mean_delivered_hops"] <= .01,
                }
                results.append(dict(scenario=scenario, candidate=candidate, reference=reference,
                                    effect_pp=float(seed_effect.mean()), per_seed_effect_pp=seed_effect.tolist(),
                                    per_seed_delivered_packet_difference=seed_packet_differences.astype(int).tolist(),
                                    descriptive_crossed_interval_pp=crossed_interval(effect, scenario+candidate+reference),
                                    seed_rates=rates.tolist(), relative_changes=relative, checks=checks,
                                    pass_all=all(checks.values()), formal_claim_allowed=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanism", choices=("local_budget", "decision_split", "pareto_split", "supported_split"), default="local_budget")
    parser.add_argument("--replicate-from", type=Path)
    parser.add_argument("--control-followup-from", type=Path)
    args = parser.parse_args()
    supported = args.mechanism == "supported_split"
    split = args.mechanism in ("decision_split", "pareto_split", "supported_split")
    guarded = args.mechanism in ("pareto_split", "supported_split")
    methods = SPLIT_METHODS if split else METHODS
    candidates = ("split_qos", "split_q") if split else ("budget_qos", "budget_q")
    seeds, workloads = SEEDS[:3], list(range(920011, 920021) if split else range(920001, 920011))
    if guarded:
        workloads = list(range(920041, 920051))
    if supported:
        seeds, workloads = SEEDS, list(range(920101, 920111))
    parent = None
    parent_path = args.replicate_from or args.control_followup_from
    primary_candidate = candidates[0]
    if supported:
        primary_candidate = "split_q"
    if args.replicate_from and args.control_followup_from:
        raise ValueError("select one parent mode")
    if args.control_followup_from:
        if args.mechanism != "pareto_split":
            raise ValueError("control follow-up is reserved for the guarded decision split")
        parent = read_json(parent_path / "completion.json")
        if parent["status"] != "complete":
            raise ValueError("incomplete control screen")
        for path, digest in parent["output_sha256"].items():
            if sha(parent_path / path) != digest:
                raise ValueError("screen artifact changed")
        parent_manifest = read_json(parent_path / "manifest.json")
        if parent_manifest["mechanism"] != "pareto_split":
            raise ValueError("unexpected control mechanism")
        # The runner now records this separately selected exploratory follow-up;
        # policy and simulation implementations must still match the screen.
        for path, digest in parent_manifest["source_sha256"].items():
            if Path(path).name != Path(__file__).name and sha(ROOT / path) != digest:
                raise ValueError("screen implementation changed: " + path)
        primary_candidate = "split_q"
        parent_gates = [r for r in read_json(parent_path / "development_gates.json")
                        if r["candidate"] == primary_candidate and r["reference"] != "direct_persistent"]
        if len(parent_gates) != 4 or not all(r["pass_all"] for r in parent_gates):
            raise ValueError("Q-assisted control does not meet all retained gates")
        seeds, workloads = SEEDS, list(range(920051, 920071))
    if args.replicate_from:
        if not split:
            raise ValueError("no replication panel reserved for the failed local-budget candidate")
        parent = read_json(args.replicate_from / "completion.json")
        if parent["status"] != "complete" or not parent["candidate_eligible_for_replication"]:
            raise ValueError("candidate did not pass its frozen screening gates")
        for path, digest in parent["output_sha256"].items():
            if sha(args.replicate_from / path) != digest:
                raise ValueError("screen artifact changed")
        parent_manifest = read_json(args.replicate_from / "manifest.json")
        if parent_manifest["mechanism"] != args.mechanism:
            raise ValueError("mechanism differs from screen")
        for path, digest in parent_manifest["source_sha256"].items():
            if sha(ROOT / path) != digest:
                raise ValueError("source changed after screen")
        seeds, workloads = SEEDS, list(range(920111, 920131) if supported else range(920051, 920071) if guarded else range(920021, 920041))
    expected_episodes = 2 * (len(seeds) * (len(methods)-1) + 1) * len(workloads)
    method_freeze = read_json(METHOD_ROOT / "training_freeze.json")
    q_freeze = read_json(CLASSICAL_ROOT / "training_freeze.json")
    spec = read_json(CLASSICAL_ROOT / "preregistration.json")
    artifacts = {}
    for scenario in LOADS:
        for base in ("qos_only_constrained", "qos_only_baseline", "q_routing"):
            for seed in seeds:
                key = f"{scenario}/{base}/seed_{seed}"
                artifact = q_freeze["jobs"][key]["model"] if base == "q_routing" else method_freeze["jobs"][key]["artifacts"]["selected_checkpoint"]
                if sha(artifact["path"]) != artifact["sha256"]:
                    raise AssertionError("checkpoint hash mismatch")
                artifacts[key] = artifact
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / "src").glob("*.py"))
    (output / "source_snapshot").mkdir()
    for path in sources:
        shutil.copyfile(path, output / "source_snapshot" / path.name)
    manifest = dict(role="exploratory_q_assisted_control_followup" if args.control_followup_from else "development_only_frozen_policy_replication" if parent else "development_only_frozen_policy_screen", seeds=seeds,
                    mechanism=args.mechanism, workloads=workloads, methods=methods, artifacts=artifacts,
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    budget="original 12% aggregate evaluation budget; split controller has no hard guarantee" if split else "per satellite prefix: 25 * switches <= 3 * opportunities; zero initial credit; direct overrides subject to same cap",
                    selection="No tuning on this panel. Expand only if the QoS-based candidate passes retained gates against both original and direct constrained in both scenarios. Q and persistent controls do not establish MAPPO innovation.",
                    gates="every seed delivery difference >=0, mean >0, every seed switch rate <=12%, seed-mean relative successful delay <=5% and hops <=1%",
                    expected_episodes=expected_episodes, training=False, sealed_test_access=False,
                    parent_completion_sha256=sha(parent_path / "completion.json") if parent else None,
                    two_actor_cost="split candidates evaluate two frozen policies per slot; additional model memory and inference must be reported" if split else None)
    manifest["pareto_guard"] = "Only replace a non-opportunity choice if observed candidate queue is no higher, progress is no lower, and at least one is strictly better. No fitted thresholds or scenario switches." if guarded else None
    manifest["primary_candidate"] = primary_candidate
    manifest["max_logit_gap"] = float(np.log(2)) if supported else None
    if supported:
        manifest["selection"] = "Fixed Q-assisted candidate with the same local Pareto guard and an additional actor logit gap <=ln(2), meaning proposed action probability >=half the actor's preferred action. No confidence-calibration claim. Screen all eight seeds without fitting this threshold; retain all existing gates against original and direct constrained. Only expand this candidate if every gate passes."
    if args.control_followup_from:
        manifest["selection"] = "New exploratory hypothesis selected after inspecting the Q-assisted control. The original QoS screen remains failed. Freeze the unchanged Q-assisted mechanism for all 8 seeds and fresh workloads; retain every delivery, budget, delay and hop gate against original and direct constrained. No pure-MAPPO or confirmatory claim. No selection among the other variants in this follow-up."
    write_json(output / "manifest.json", manifest)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, pairings, started = [], {}, time.monotonic()
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream, gzip.open(output / "packets.jsonl.gz", "wt", encoding="utf-8") as packets_stream:
            for scenario, load in LOADS.items():
                for name, (base, mode) in methods.items():
                    for seed in [-1] if base == "persistent_global_dijkstra" else seeds:
                        if seed == -1:
                            policy = PersistentDijkstraPolicy()
                        else:
                            path = Path(artifacts[f"{scenario}/{base}/seed_{seed}"]["path"])
                            if base == "q_routing":
                                job = next(j for j in classical.build_training_jobs() if (j.scenario, j.policy_seed) == (scenario, seed))
                                policy, _ = classical.load_q_model(path, job, spec, require_frozen=True)
                            else:
                                policy, _ = load_checkpoint_policy(path, device="cpu")
                        if mode == "direct":
                            policy = DirectDeliveryControlPolicy(policy)
                        elif mode == "budget":
                            policy = LocalSwitchBudgetPolicy(policy)
                        elif mode.startswith("split_"):
                            route_base = "qos_only_baseline" if mode == "split_qos" else "q_routing"
                            route_path = Path(artifacts[f"{scenario}/{route_base}/seed_{seed}"]["path"])
                            if route_base == "q_routing":
                                job = next(j for j in classical.build_training_jobs() if (j.scenario, j.policy_seed) == (scenario, seed))
                                routing, _ = classical.load_q_model(route_path, job, spec, require_frozen=True)
                            else:
                                routing, _ = load_checkpoint_policy(route_path, device="cpu")
                            policy = DecisionSplitPolicy(policy, routing, pareto_guard=guarded,
                                                         max_logit_gap=float(np.log(2)) if supported else None)
                        for workload in workloads:
                            row, packets, _ = run_episode(scenario, load, name, policy, seed, workload)
                            if mode == "budget":
                                if int(policy.opportunities.sum()) != row["decision_switch_opportunities"] or int(policy.switches.sum()) != row["decision_avoidable_switches"]:
                                    raise AssertionError("local counters disagree with environment")
                                if np.any(25 * policy.switches > 3 * policy.opportunities):
                                    raise AssertionError("local prefix budget exceeded")
                                row.update(local_opportunities=policy.opportunities.tolist(), local_switches=policy.switches.tolist(), denied_switches=policy.denied)
                            if mode.startswith("split_"):
                                row["split_counts"] = dict(policy.counts)
                                if policy.counts["opportunity"] != row["decision_switch_opportunities"]:
                                    raise AssertionError("split eligibility disagrees with environment")
                            signature = (row["arrival_sha256"], row["physical_sha256"])
                            key = scenario, workload
                            if key in pairings and pairings[key] != signature:
                                raise AssertionError("external event mismatch")
                            pairings[key] = signature
                            rows.append(row)
                            stream.write(canonical(row) + "\n")
                            identity = {k: row[k] for k in ("scenario", "policy", "policy_seed", "workload_seed")}
                            for packet in packets:
                                packets_stream.write(canonical({**identity, **packet}) + "\n")
                        stream.flush()
                        print(f"{len(rows)}/{expected_episodes} {scenario} {name} seed={seed} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if len(rows) != manifest["expected_episodes"]:
            raise AssertionError("incomplete panel")
        for artifact in artifacts.values():
            if sha(artifact["path"]) != artifact["sha256"]:
                raise AssertionError("model modified")
        for path, digest in manifest["source_sha256"].items():
            if sha(ROOT / path) != digest:
                raise AssertionError("source modified")
        results = assess(rows, seeds, workloads, candidates)
        write_json(output / "summary.json", aggregate(rows))
        write_json(output / "development_gates.json", results)
        eligible = all(r["pass_all"] for r in results if r["candidate"] == primary_candidate and r["reference"] != "direct_persistent")
        write_json(output / "completion.json", dict(status="complete", episodes=len(rows), elapsed_seconds=time.monotonic()-started,
                   candidate_eligible_for_replication=eligible, input_hashes_unchanged=True,
                   output_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}))
        print(canonical(results))
    except Exception as error:
        write_json(output / "failure.json", dict(error=repr(error), completed_episodes=len(rows)))
        raise


if __name__ == "__main__":
    main()
