"""Frozen-actor diagnostics of switch surrogates, occupancy, and shared gradients."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from mappo_design import SharedCandidateActor, avoidable_switch_decisions
from mappo_evaluation import load_checkpoint_policy
from run_development_load_diagnostics import (
    ROOT, METHOD_ROOT, canonical, read_json, run_episode, sha, write_json,
)
from run_direct_delivery_probe import LOADS
from run_local_switch_budget_probe import SEEDS
from switch_surrogate_diagnostics import gradient_probe, summarize_decisions


WORKLOADS = list(range(930001, 930011))
RESERVED_REPLICATION = list(range(930101, 930121))
BASES = ("qos_only_constrained", "qos_only_baseline")
MODES = ("greedy", "sampled")


def audit_seed_metadata():
    """Check recorded seed values and ranges without opening sealed outcomes."""
    command = [
        "rg", "--files", "--hidden", "--no-ignore", "experiments",
        "-g", "*manifest*.json", "-g", "*registration*.json", "-g", "*freeze*.json",
        "-g", "!**/source_snapshot/**", "-g", "!**/*sealed*/**",
    ]
    paths = subprocess.check_output(command, cwd=ROOT, text=True).splitlines()
    requested = set(WORKLOADS + RESERVED_REPLICATION)
    hashes = {}

    def check(value, location):
        if isinstance(value, dict):
            for key, child in value.items():
                check(child, location + "/" + key)
        elif isinstance(value, list):
            if ("range" in location.lower() and len(value) == 2
                    and all(type(v) is int for v in value)):
                if any(value[0] <= seed <= value[1] for seed in requested):
                    raise ValueError("reserved workload range overlaps " + location)
            for index, child in enumerate(value):
                check(child, location + "/" + str(index))
        elif type(value) is int and value in requested:
            raise ValueError("workload seed already reserved/exposed in " + location)

    for relative in sorted(paths):
        path = ROOT / relative
        check(read_json(path), relative)
        hashes[relative] = sha(path)
    return hashes


def reconstruct_actor(checkpoint):
    spec = checkpoint["candidate_actor_spec"]
    accepted = inspect.signature(SharedCandidateActor).parameters
    kwargs = {key: value for key, value in spec.items() if key in accepted}
    if kwargs.get("route_hysteresis_residual_parameterization"):
        raise ValueError("this diagnostic is frozen for actors without route residuals")
    kwargs.update(
        candidate_feature_dim=int(checkpoint["candidate_feature_dim"]),
        hidden_dim=int(checkpoint["args"]["actor_hidden_dim"]),
        num_layers=int(checkpoint["args"]["actor_num_layers"]),
    )
    actor = SharedCandidateActor(**kwargs).eval()
    prefix = "shared_candidate_actor."
    actor.load_state_dict({key[len(prefix):]: value for key, value in checkpoint["actor"].items()
                           if key.startswith(prefix)})
    return actor


class RecordingPolicy:
    def __init__(self, policy, mode, rng_seed):
        self.policy = policy
        self.mode = mode
        self.checkpoint_schema = policy.checkpoint_schema
        self.switch_constraint_spec = policy.switch_constraint_spec
        self.generator = torch.Generator().manual_seed(rng_seed)
        self.observations, self.masks, self.logits, self.actions = [], [], [], []
        self.pending = None
        self.wrapper = None

    def bind(self, wrapper):
        self.wrapper = wrapper

    def __call__(self, observation, mask):
        logits = self.policy.action_logits(observation, mask)
        if self.mode == "greedy":
            actions = logits.argmax(axis=-1)
            if not np.array_equal(actions, self.policy(observation, mask)):
                raise AssertionError("logit interface changed greedy policy actions")
        else:
            actions = torch.multinomial(torch.softmax(torch.from_numpy(logits), dim=-1),
                                        1, generator=self.generator).squeeze(-1).numpy()
        self.observations.append(np.array(observation, copy=True))
        self.masks.append(np.asarray(mask, dtype=bool).copy())
        self.logits.append(logits.copy())
        self.actions.append(actions.copy())
        candidates = torch.from_numpy(np.asarray(observation, dtype=np.float32)).reshape(
            len(mask), mask.shape[-1], self.checkpoint_schema["candidate_feature_dim"])
        self.pending = avoidable_switch_decisions(
            candidates, torch.from_numpy(np.asarray(mask, dtype=bool)),
            torch.from_numpy(actions), self.wrapper.get_route_switch_feature_index())
        return actions

    def observe_transition(self, info):
        costs, opportunities = self.pending
        order = np.asarray(self.wrapper.external_to_internal) - 1
        for field, expected in (("decision_avoidable_switch_costs", costs),
                                ("decision_switch_opportunities", opportunities)):
            if not np.array_equal(np.asarray(info[field])[order], expected.numpy()):
                raise AssertionError("helper/environment decision ledger mismatch: " + field)

    def arrays(self):
        return {"observations": np.concatenate(self.observations),
                "masks": np.concatenate(self.masks), "logits": np.concatenate(self.logits),
                "actions": np.concatenate(self.actions)}


def grouped_episode_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["policy"], row["policy_seed"])].append(row)
    result = []
    for (scenario, policy, seed), values in sorted(groups.items()):
        opportunities = sum(v["decision_switch_opportunities"] for v in values)
        switches = sum(v["decision_avoidable_switches"] for v in values)
        result.append(dict(scenario=scenario, policy=policy, policy_seed=seed,
                           generated=sum(v["generated"] for v in values),
                           delivered=sum(v["delivered"] for v in values),
                           opportunities=opportunities, switches=switches,
                           switch_rate=switches / opportunities if opportunities else None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; do not rerun or overwrite frozen diagnostics")
    metadata_hashes = audit_seed_metadata()
    freeze = read_json(METHOD_ROOT / "training_freeze.json")
    artifacts = {}
    for scenario in LOADS:
        for seed in SEEDS:
            for base in BASES:
                key = f"{scenario}/{base}/seed_{seed}"
                artifact = freeze["jobs"][key]["artifacts"]["selected_checkpoint"]
                if sha(artifact["path"]) != artifact["sha256"]:
                    raise AssertionError("input model changed: " + key)
                artifacts[key] = artifact
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    sources = sorted((ROOT / "src").glob("*.py"))
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    manifest = dict(
        role="development_only_switch_surrogate_diagnostic", workloads=WORKLOADS,
        reserved_replication_workloads=RESERVED_REPLICATION, policy_seeds=SEEDS,
        scenarios=LOADS, bases=BASES, occupancy_modes=MODES, artifacts=artifacts,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        seed_metadata_sha256=metadata_hashes, metadata_files_scanned=len(metadata_hashes),
        sealed_test_access=False, training=False, candidate_selection=False,
        expected_episodes=len(LOADS) * len(BASES) * len(MODES) * len(SEEDS) * len(WORKLOADS),
        gradient_probe=dict(relative_parameter_l2_step=0.001, step_rule="negative unit surrogate gradient times max(parameter_l2, 1)",
                            purpose="local sensitivity only; no PPO/QoS gradient or training outcome claim"),
        pairing="same exogenous arrivals, topology and faults; endogenous queues may differ",
        fixed_next_step="Use diagnostic evidence to choose one training change; never select a routing candidate on this panel.",
        retained_gates=dict(switch_budget=0.12, per_seed_delivery_difference_minimum=0,
                            mean_delivery_difference_positive=True, mean_relative_delay_maximum=0.05,
                            mean_relative_hops_maximum=0.01, independent_replication_required=True),
    )
    write_json(output / "manifest.json", manifest)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    rows, diagnostics, pairings = [], [], {}
    (output / "occupancy").mkdir()
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for scenario, load in LOADS.items():
                for seed in SEEDS:
                    policies, actors, checkpoints = {}, {}, {}
                    for base in BASES:
                        key = f"{scenario}/{base}/seed_{seed}"
                        policies[base], checkpoints[base] = load_checkpoint_policy(Path(artifacts[key]["path"]), device="cpu")
                        actors[base] = reconstruct_actor(checkpoints[base])
                    for base in BASES:
                        for mode in MODES:
                            rng_seed = int.from_bytes(hashlib.sha256(canonical([scenario, seed, base, mode]).encode()).digest()[:8], "big") % (2**63)
                            recorder = RecordingPolicy(policies[base], mode, rng_seed)
                            for workload in WORKLOADS:
                                row, _, _ = run_episode(scenario, load, base + "_" + mode, recorder, seed, workload)
                                pair = (row["arrival_sha256"], row["physical_sha256"])
                                pairing_key = (scenario, workload)
                                if pairings.setdefault(pairing_key, pair) != pair:
                                    raise AssertionError("exogenous pairing mismatch")
                                rows.append(row)
                                stream.write(canonical(row) + "\n")
                                stream.flush()
                            arrays = recorder.arrays()
                            occupancy_name = f"{scenario}__{seed}__{base}__{mode}"
                            np.savez_compressed(output / "occupancy" / (occupancy_name + ".npz"), **arrays)
                            candidates = torch.from_numpy(arrays["observations"]).reshape(
                                len(arrays["masks"]), arrays["masks"].shape[-1],
                                policies[base].checkpoint_schema["candidate_feature_dim"])
                            mask = torch.from_numpy(arrays["masks"])
                            for scored_base in BASES:
                                with torch.no_grad():
                                    logits = actors[scored_base](candidates, mask)
                                if scored_base == base and not torch.allclose(logits, torch.from_numpy(arrays["logits"]), rtol=1e-5, atol=2e-6):
                                    raise AssertionError("reconstructed actor logits do not reproduce inference")
                                diag = dict(scenario=scenario, policy_seed=seed, occupancy_policy=base,
                                            occupancy_mode=mode, scored_policy=scored_base,
                                            decisions=summarize_decisions(logits, candidates, mask))
                                if scored_base == base:
                                    multiplier = (checkpoints[base].get("switch_constraint_state") or {}).get("multiplier", 0.0)
                                    diag["gradient_probe"] = gradient_probe(actors[base], candidates, mask,
                                                                          relative_step=0.001, dual_multiplier=multiplier)
                                diagnostics.append(diag)
                            write_json(output / "progress.json", dict(episodes=len(rows), expected_episodes=manifest["expected_episodes"],
                                                                       last_occupancy=occupancy_name, elapsed_seconds=time.monotonic()-started))
                            print(f"{len(rows)}/{manifest['expected_episodes']} {occupancy_name}", flush=True)
        if len(rows) != manifest["expected_episodes"]:
            raise AssertionError("episode count mismatch")
        if not all(sha(item["path"]) == item["sha256"] for item in artifacts.values()):
            raise AssertionError("input model changed during diagnostic")
        write_json(output / "diagnostics.json", diagnostics)
        write_json(output / "episode_summary.json", grouped_episode_metrics(rows))
        write_json(output / "completion.json", dict(
            status="complete", episodes=len(rows), elapsed_seconds=time.monotonic()-started,
            input_hashes_unchanged=True, exogenous_pairing_verified=True,
            helper_environment_ledgers_verified=True, candidate_eligible_for_replication=False,
            formal_claim_allowed=False, reserved_replication_accessed=False,
            output_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        ))
    except Exception as error:
        write_json(output / "completion.json", dict(status="failed", episodes=len(rows),
                   elapsed_seconds=time.monotonic()-started, error=repr(error)))
        raise


if __name__ == "__main__":
    main()
