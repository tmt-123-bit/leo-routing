"""Single-change development probe: deliver directly to a feasible destination."""

import argparse
import copy
import csv
import gzip
from pathlib import Path
import time

import numpy as np
import torch

from mappo_evaluation import load_checkpoint_policy
from run_development_load_diagnostics import (
    PersistentDijkstraPolicy, aggregate, canonical, read_json, run_episode, sha, write_json,
)

LOADS = {"medium_load": 6, "hotspot_high_load": 16}


def direct_action(destination, neighbors, mask, original):
    if destination in neighbors:
        action = neighbors.index(destination) + 1
        if mask[action]:
            return action
    return int(original)


class DirectDeliveryPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.checkpoint_schema = copy.deepcopy(policy.checkpoint_schema)
        self.switch_constraint_spec = copy.deepcopy(getattr(policy, "switch_constraint_spec", None))
        self.wrapper = None
        self.overrides = 0

    def bind(self, wrapper):
        self.wrapper = wrapper
        self.overrides = 0

    def __call__(self, observation, mask):
        actions = np.array(self.policy(observation, mask), dtype=np.int64, copy=True)
        for external, sat in enumerate(self.wrapper.external_to_internal):
            obs = self.wrapper._obs[sat - 1]
            if obs["hol_packet_id"] is None:
                continue
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            original = actions[external]
            actions[external] = direct_action(packet.dst, obs["neighbor_ids"], mask[external], original)
            self.overrides += actions[external] != original
        return actions


def assess(rows):
    effects = []
    for scenario in LOADS:
        seeds = sorted({r["policy_seed"] for r in rows if r["policy"] == "qos_only_constrained" and r["scenario"] == scenario})
        seed_effects, delay_effects, hop_effects, new_rates = [], [], [], []
        paired_rows = []
        for seed in seeds:
            reference = {r["workload_seed"]: r for r in rows if (r["scenario"], r["policy_seed"], r["policy"]) == (scenario, seed, "qos_only_constrained")}
            treatment = {r["workload_seed"]: r for r in rows if (r["scenario"], r["policy_seed"], r["policy"]) == (scenario, seed, "direct_delivery_constrained")}
            if reference.keys() != treatment.keys():
                raise AssertionError("unpaired workloads")
            delivery_diff = float(np.mean([treatment[w]["delivery_ratio"] - reference[w]["delivery_ratio"] for w in reference]))
            opp = sum(r["decision_switch_opportunities"] for r in treatment.values())
            rate = sum(r["decision_avoidable_switches"] for r in treatment.values()) / opp if opp else None
            if rate is None:
                raise AssertionError("undefined budget rate")
            seed_effects.append(delivery_diff)
            new_rates.append(rate)
            for field, target in (("average_delay_slots", delay_effects), ("mean_delivered_hops", hop_effects)):
                before = np.mean([r[field] for r in reference.values()])
                after = np.mean([r[field] for r in treatment.values()])
                target.append(float(after / before - 1))
            paired_rows.append({"policy_seed": seed, "delivery_diff_pp": 100 * delivery_diff,
                                "treatment_switch_rate": rate, "overrides": sum(r["direct_overrides"] for r in treatment.values())})
        gates = {
            "every_seed_delivery_nondecreasing": all(d >= 0 for d in seed_effects),
            "mean_delivery_positive": np.mean(seed_effects) > 0,
            "every_seed_switch_rate_within_12pct": max(new_rates) <= 0.12,
            "mean_success_delay_increase_at_most_5pct": np.mean(delay_effects) <= 0.05,
            "mean_success_hops_increase_at_most_1pct": np.mean(hop_effects) <= 0.01,
        }
        effects.append({"scenario": scenario, "paired_seeds": paired_rows,
                        "delivery_diff_pp": 100 * float(np.mean(seed_effects)),
                        "mean_success_delay_relative_change": float(np.mean(delay_effects)),
                        "mean_success_hops_relative_change": float(np.mean(hop_effects)),
                        "gates": {k: bool(v) for k, v in gates.items()},
                        "development_screen_pass": bool(all(gates.values())),
                        "paper_claim_allowed": False})
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
    workload_seeds = list(range(910011, 910021))
    if set(workload_seeds) & set(manifest["workload_seeds"]):
        raise ValueError("probe workloads overlap diagnostic development panel")
    artifacts = [a for a in manifest["artifacts"] if a["method"] == "qos_only_constrained"]
    for artifact in artifacts:
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("frozen model changed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "role": "single_change_development_probe", "source_manifest_sha256": sha(source / "manifest.json"),
        "runner_sha256": sha(Path(__file__)), "policy_seeds": manifest["policy_seeds"],
        "artifacts": artifacts, "workloads": workload_seeds, "loads": LOADS,
        "intervention": "override only when destination is in the original feasible action mask; no change to weights, masks, rewards, queue service, or switch accounting",
        "training": False, "sealed_test_access": False,
        "gates_predeclared": "per scenario: all seed delivery differences >=0; mean >0; all seed switch rates <=0.12; seed-mean relative success-delay increase <=5%; success-hop increase <=1%",
        "interpretation": "development screening, no formal significance or promotion claim; conditional success-delay bias remains possible",
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
                for method in ("qos_only_constrained", "direct_delivery_constrained", "persistent_global_dijkstra"):
                    for seed in manifest["policy_seeds"] if method != "persistent_global_dijkstra" else [-1]:
                        if seed == -1:
                            policy = PersistentDijkstraPolicy()
                        else:
                            artifact = next(a for a in artifacts if (a["scenario"], a["seed"]) == (scenario, seed))
                            policy, _ = load_checkpoint_policy(Path(artifact["path"]), device="cpu")
                            if method == "direct_delivery_constrained":
                                policy = DirectDeliveryPolicy(policy)
                        for workload in workload_seeds:
                            row, packets, _ = run_episode(scenario, load, method, policy, seed, workload)
                            row["direct_overrides"] = int(getattr(policy, "overrides", 0))
                            key = (scenario, workload)
                            events = (row["arrival_sha256"], row["physical_sha256"])
                            if key in pairing and pairing[key] != events:
                                raise AssertionError("exogenous pairing changed")
                            pairing[key] = events
                            episode_file.write(canonical(row) + "\n")
                            identity = {k: row[k] for k in ("scenario", "load", "policy", "policy_seed", "workload_seed")}
                            for packet in packets:
                                packet_file.write(canonical({**identity, **packet}) + "\n")
                            rows.append(row)
                        episode_file.flush()
                        print(f"{len(rows)}/140 {scenario} {method} seed={seed} elapsed={time.monotonic()-started:.1f}s", flush=True)
        for artifact in artifacts:
            if sha(artifact["path"]) != artifact["sha256"]:
                raise AssertionError("frozen input changed")
        summaries, effects = aggregate(rows), assess(rows)
        write_json(output / "summary.json", summaries)
        write_json(output / "paired_effects.json", effects)
        with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
            fields = ["scenario", "policy", "delivery_ratio", "drop_rate", "backlog_rate", "decision_switch_rate"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
                   "exogenous_pairings_verified": len(pairing), "elapsed_seconds": time.monotonic()-started,
                   "development_screen_pass": all(e["development_screen_pass"] for e in effects),
                   "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
        print(canonical(effects), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"error": repr(error), "episodes": len(rows)})
        raise


if __name__ == "__main__":
    main()
