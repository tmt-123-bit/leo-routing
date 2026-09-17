"""Development probe: stack the remaining no-retraining headroom.

Arms: ``base``; ``purge1`` (infeasibility purge, safety margin 1); ``purge0``
(safety margin 0, still provably only removing packets whose hop distance
exceeds their remaining deliverable slots); ``purge0_direct`` (purge0 plus the
direct-delivery action override for MAPPO arms). Methods: constrained MAPPO and
Q-routing. Arrival and physical event pairing is asserted between paired arms.
"""

import argparse
import csv
import gzip
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from mappo_evaluation import evaluate_policy_with_constraint_metrics, load_checkpoint_policy
from run_development_load_diagnostics import (
    DiagnosticWrapper,
    aggregate,
    canonical,
    packet_accounting,
    read_json,
    sha,
    write_json,
)
import run_development_load_diagnostics as diagnostics
import run_avoidable_switch_classical_baselines_formal as classical
from run_infeasible_drop_probe import PurgeInfeasibleEnv
from run_direct_delivery_probe import DirectDeliveryPolicy

LOADS = {"medium_load": 6, "hotspot_high_load": 16}
WORKLOAD_SEEDS = list(range(910071, 910081))
METHODS = ("qos_only_constrained", "q_routing")
ENVS = ("base", "purge1", "purge0")
EXTRA_MAPPO_ARMS = ("purge0_direct",)

if "early_infeasible" not in diagnostics.DROP_REASONS:
    diagnostics.DROP_REASONS = diagnostics.DROP_REASONS + ("early_infeasible",)


class PurgeMarginZeroEnv(PurgeInfeasibleEnv):
    safety_margin_slots = 0


class StackedDiagnosticWrapper(DiagnosticWrapper):
    def __init__(self, *args, arm: str = "base", **kwargs):
        super().__init__(*args, **kwargs)
        if arm == "base":
            return
        env_cls = PurgeInfeasibleEnv if arm == "purge1" else PurgeMarginZeroEnv
        self.env = env_cls(self.env.cfg)


def run_episode(scenario, load, policy_name, policy, policy_seed, workload_seed, arm):
    initial, _ = MULTIAGENT_LOADS[scenario]
    cfg = MultiAgentConfig(
        env=EnvConfig(seed=workload_seed, scenario=SCENARIOS[scenario]),
        initial_packets=initial, exogenous_packets_per_slot=load,
        seed=workload_seed, variant="qos_only",
    )
    wrapper = StackedDiagnosticWrapper(scenario=scenario, cfg=cfg, arm=arm)
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
        for method in METHODS:
            seeds = sorted({r["policy_seed"] for r in rows
                            if r["policy"] == method and r["scenario"] == scenario})
            treatments = list(ENVS[1:]) + (list(EXTRA_MAPPO_ARMS) if method == "qos_only_constrained" else [])
            for treatment in treatments:
                seed_effects, new_rates = [], []
                for seed in seeds:
                    reference = {r["workload_seed"]: r for r in rows
                                 if (r["scenario"], r["policy_seed"], r["policy"]) == (scenario, seed, method)}
                    treat = {r["workload_seed"]: r for r in rows
                             if (r["scenario"], r["policy_seed"], r["policy"]) == (scenario, seed, method + "_" + treatment)}
                    if reference.keys() != treat.keys():
                        raise AssertionError("unpaired workloads")
                    seed_effects.append(float(np.mean(
                        [treat[w]["delivery_ratio"] - reference[w]["delivery_ratio"] for w in reference])))
                    opp = sum(r["decision_switch_opportunities"] for r in treat.values())
                    if opp:
                        new_rates.append(sum(r["decision_avoidable_switches"] for r in treat.values()) / opp)
                effects.append({
                    "scenario": scenario, "method": method, "treatment": treatment,
                    "per_seed_pp": [100 * d for d in seed_effects],
                    "delivery_diff_pp": 100 * float(np.mean(seed_effects)),
                    "max_treatment_switch_rate": max(new_rates) if new_rates else None,
                })
    return effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    manifest = read_json(source / "manifest.json")
    completion = read_json(source / "completion.json")
    if completion["status"] != "complete" or sha(source / "manifest.json") != completion["output_sha256"]["manifest.json"]:
        raise ValueError("source manifest is not bound to a completed run")
    if set(WORKLOAD_SEEDS) & set(manifest["workload_seeds"]):
        raise ValueError("probe workloads overlap diagnostic development panel")
    artifacts = [a for a in manifest["artifacts"] if a["method"] == "qos_only_constrained"]
    for artifact in artifacts:
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("frozen model changed")
    classical_freeze = read_json(diagnostics.CLASSICAL_ROOT / "training_freeze.json")
    classical_spec = read_json(diagnostics.CLASSICAL_ROOT / "preregistration.json")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "role": "development_probe_stacked",
        "source_manifest_sha256": sha(source / "manifest.json"),
        "runner_sha256": sha(Path(__file__)),
        "policy_seeds": manifest["policy_seeds"],
        "artifacts": artifacts, "workloads": WORKLOAD_SEEDS, "loads": LOADS,
        "arms": ENVS + EXTRA_MAPPO_ARMS, "training": False, "sealed_test_access": False,
        "interpretation": "development screening of stacked no-retraining headroom; no formal claim",
    })
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, pairing = [], {}
    started = time.monotonic()
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as episode_file:
            for scenario, load in LOADS.items():
                for method in METHODS:
                    for seed in manifest["policy_seeds"]:
                        if method == "qos_only_constrained":
                            artifact = next(a for a in artifacts if (a["scenario"], a["seed"]) == (scenario, seed))
                            base_policy, _ = load_checkpoint_policy(Path(artifact["path"]), device="cpu")
                        else:
                            job = next(j for j in classical.build_training_jobs()
                                       if j.scenario == scenario and j.policy_seed == seed)
                            entry = classical_freeze["jobs"][job.job_id]["model"]
                            base_policy, _ = classical.load_q_model(Path(entry["path"]), job, classical_spec, require_frozen=True)
                        arms = list(ENVS) + (list(EXTRA_MAPPO_ARMS) if method == "qos_only_constrained" else [])
                        for arm in arms:
                            label = method if arm == "base" else method + "_" + arm
                            policy = base_policy
                            if arm == "purge0_direct":
                                policy = DirectDeliveryPolicy(base_policy)
                            for workload in WORKLOAD_SEEDS:
                                row, _ = run_episode(scenario, load, label, policy, seed, workload, arm)
                                key = (scenario, workload)
                                events = (row["arrival_sha256"], row["physical_sha256"])
                                if key in pairing:
                                    if pairing[key]["events"] != events:
                                        raise AssertionError("exogenous pairing changed")
                                    pairing[key]["arms"] += 1
                                else:
                                    pairing[key] = {"events": events, "arms": 1}
                                episode_file.write(canonical(row) + "\n")
                                rows.append(row)
                            episode_file.flush()
                            print(f"{len(rows)} {scenario} {label} seed={seed} elapsed={time.monotonic()-started:.1f}s", flush=True)
        for artifact in artifacts:
            if sha(artifact["path"]) != artifact["sha256"]:
                raise AssertionError("frozen input changed")
        if any(entry["arms"] != 21 for entry in pairing.values()):
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
                   "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
        print(canonical(effects), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error), "episodes": len(rows)})
        raise


if __name__ == "__main__":
    main()
