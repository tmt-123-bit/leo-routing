"""Eight-seed development replication with symmetric direct-delivery controls."""

import argparse
import copy
import csv
import gzip
import hashlib
import itertools
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from mappo_evaluation import load_checkpoint_policy
from run_direct_delivery_probe import DirectDeliveryPolicy, LOADS
from run_development_load_diagnostics import (
    ROOT, METHOD_ROOT, CLASSICAL_ROOT, PersistentDijkstraPolicy, aggregate,
    canonical, classical, read_json, run_episode, sha, write_json,
)

BASES = ("qos_only_constrained", "qos_only_baseline", "q_routing", "persistent_global_dijkstra")
WORKLOADS = tuple(range(910021, 910041))
BOOTSTRAPS = 5000


class DirectDeliveryControlPolicy(DirectDeliveryPolicy):
    """Use the original intervention unchanged, with binding for frozen baselines."""

    def __init__(self, policy):
        if getattr(policy, "training", False):
            raise ValueError("direct-delivery controls require a frozen policy")
        self.policy = policy
        self.wrapper = None
        self.overrides = 0
        for name in ("checkpoint_schema", "switch_constraint_spec"):
            if hasattr(policy, name):
                setattr(self, name, copy.deepcopy(getattr(policy, name)))

    def bind(self, wrapper):
        super().bind(wrapper)
        binder = getattr(self.policy, "bind", None)
        if binder is not None:
            binder(wrapper)


def row_index(rows):
    index = {}
    for row in rows:
        key = (row["scenario"], row["policy"], row["policy_seed"], row["workload_seed"])
        if key in index:
            raise ValueError("duplicate episode identity")
        index[key] = row
    return index


def matrix(index, scenario, method, seeds, workloads, field):
    available = {key[2] for key in index if key[:2] == (scenario, method)}
    actual_seeds = [-1] * len(seeds) if available == {-1} else seeds
    try:
        return np.asarray([[index[(scenario, method, seed, workload)][field]
                            for workload in workloads] for seed in actual_seeds], dtype=float)
    except KeyError as error:
        raise ValueError("incomplete paired episode panel") from error


def crossed_interval(values, tag):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big"))
    n_seed, n_work = values.shape
    draws = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        si = rng.integers(0, n_seed, n_seed)
        wi = rng.integers(0, n_work, n_work)
        draws[i] = values[np.ix_(si, wi)].mean()
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def exact_sign_flip(values):
    if len(values) <= 1:
        return None
    observed = abs(float(np.mean(values)))
    extreme = sum(abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12
                  for signs in itertools.product((-1, 1), repeat=len(values)))
    return extreme / 2 ** len(values)


def contrast(index, scenario, terms, seeds, tag):
    generated = None
    differences = np.zeros((len(seeds), len(WORKLOADS)))
    for method, coefficient in terms:
        counts = matrix(index, scenario, method, seeds, WORKLOADS, "generated")
        if generated is None:
            generated = counts
        if not np.array_equal(generated, counts):
            raise ValueError("generated packet denominators differ across methods")
        differences += coefficient * matrix(index, scenario, method, seeds, WORKLOADS, "delivered")
    effects = differences / generated * 100
    return {"scenario": scenario, "contrast": tag, "terms": terms,
            "mean_delivery_effect_pp": float(effects.mean()),
            "crossed_95pct_interval_pp": crossed_interval(effects, scenario + tag),
            "per_seed_effect_pp": [float(v) for v in effects.mean(axis=1)],
            "positive_seeds": int((differences.sum(axis=1) > 0).sum()),
            "negative_seeds": int((differences.sum(axis=1) < 0).sum()),
            "unchanged_seeds": int((differences.sum(axis=1) == 0).sum()),
            "raw_seed_sign_flip_p": exact_sign_flip(effects.mean(axis=1)),
            "formal_claim_allowed": False}


def analyze(rows, seeds):
    index, effects, gates = row_index(rows), [], []
    for scenario in LOADS:
        for base in BASES:
            effect = contrast(index, scenario, [("direct_" + base, 1), (base, -1)],
                              seeds if base != BASES[-1] else [-1], "direct_minus_original_" + base)
            effects.append(effect)
        for reference in ("qos_only_baseline", "q_routing", "persistent_global_dijkstra"):
            effects.append(contrast(index, scenario,
                                    [("direct_qos_only_constrained", 1), ("direct_" + reference, -1)],
                                    seeds, "direct_constrained_minus_direct_" + reference))
        effects.append(contrast(index, scenario, [("direct_qos_only_constrained", 1), ("qos_only_constrained", -1),
                                                  ("direct_qos_only_baseline", -1), ("qos_only_baseline", 1)],
                                seeds, "constraint_vs_qos_difference_in_direct_rule_effect"))
        selected = next(e for e in effects if e["scenario"] == scenario and e["contrast"] == "direct_minus_original_qos_only_constrained")
        costs = matrix(index, scenario, "direct_qos_only_constrained", seeds, WORKLOADS, "decision_avoidable_switches")
        opportunities = matrix(index, scenario, "direct_qos_only_constrained", seeds, WORKLOADS, "decision_switch_opportunities")
        if np.any(opportunities.sum(axis=1) == 0):
            raise ValueError("undefined per-seed constraint rate")
        rates = costs.sum(axis=1) / opportunities.sum(axis=1)
        relative = {}
        for field in ("average_delay_slots", "mean_delivered_hops"):
            before = matrix(index, scenario, "qos_only_constrained", seeds, WORKLOADS, field).mean(axis=1)
            after = matrix(index, scenario, "direct_qos_only_constrained", seeds, WORKLOADS, field).mean(axis=1)
            relative[field] = float((after / before - 1).mean())
        checks = {"every_seed_delivery_nondecreasing": selected["negative_seeds"] == 0,
                  "mean_delivery_positive": selected["mean_delivery_effect_pp"] > 0,
                  "every_seed_rate_within_12pct": bool(np.all(rates <= 0.12)),
                  "mean_success_delay_increase_at_most_5pct": relative["average_delay_slots"] <= 0.05,
                  "mean_success_hops_increase_at_most_1pct": relative["mean_delivered_hops"] <= 0.01}
        gates.append({"scenario": scenario, "checks": checks, "development_screen_pass": all(checks.values()),
                      "seed_switch_rates": [float(v) for v in rates], "relative_changes": relative})
    return effects, gates


def export_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    method_freeze = read_json(METHOD_ROOT / "training_freeze.json")
    q_freeze = read_json(CLASSICAL_ROOT / "training_freeze.json")
    q_spec = read_json(CLASSICAL_ROOT / "preregistration.json")
    seed_sets = [{v["job"]["policy_seed"] for v in method_freeze["jobs"].values()
                  if v["job"]["scenario"] == scenario and v["job"]["arm"] == arm}
                 for scenario in LOADS for arm in BASES[:2]]
    seeds = sorted(set.intersection(*seed_sets))
    if len(seeds) != 8:
        raise ValueError("expected all eight common frozen model seeds")
    artifacts = []
    for scenario in LOADS:
        for method in BASES[:3]:
            for seed in seeds:
                key = f"{scenario}/{method}/seed_{seed}"
                entry = q_freeze["jobs"][key]["model"] if method == "q_routing" else method_freeze["jobs"][key]["artifacts"]["selected_checkpoint"]
                if sha(entry["path"]) != entry["sha256"]:
                    raise ValueError("frozen model hash mismatch: " + key)
                artifacts.append({"scenario": scenario, "method": method, "seed": seed, **entry})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = sorted(p for p in (ROOT / "src").glob("*.py") if not p.name.startswith("test_"))
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    manifest = {
        "role": "eight_seed_development_replication_with_symmetric_controls", "policy_seeds": seeds,
        "workload_seeds": WORKLOADS, "scenarios_and_loads": LOADS, "base_methods": BASES,
        "variants": ["unchanged", "direct_destination_if_original_mask_allows"],
        "training": False, "sealed_test_access": False, "artifacts": artifacts,
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "parent_probe_completion_sha256": sha(ROOT / "experiments/direct-delivery-probe-20260909-v1/completion.json"),
        "gates": "retain the prior development gates without changes: every constrained seed delivery difference >=0; mean >0; every seed decision switch rate <=12%; mean relative success-delay increase <=5%; success-hop increase <=1%",
        "statistics": "5000 crossed seed/workload bootstrap draws; paired packet-count contrasts; exact seed sign flips where n_seed>1; descriptive intervals and unadjusted p values, no confirmatory claims",
        "deterministic_baseline": "one persistent Dijkstra per workload, not eight independent models; same direct rule applied",
        "expected_episodes": 2000,
    }
    write_json(output / "manifest.json", manifest)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, pairings, started = [], {}, time.monotonic()
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as episodes_file, \
                gzip.open(output / "packets.jsonl.gz", "wt", encoding="utf-8") as packets_file:
            for scenario, load in LOADS.items():
                for base in BASES:
                    for seed in seeds if base != BASES[-1] else [-1]:
                        for direct in (False, True):
                            name = "direct_" + base if direct else base
                            if base == BASES[-1]:
                                policy = PersistentDijkstraPolicy()
                            else:
                                artifact = next(a for a in artifacts if (a["scenario"], a["method"], a["seed"]) == (scenario, base, seed))
                                if base == "q_routing":
                                    job = next(j for j in classical.build_training_jobs() if (j.scenario, j.policy_seed) == (scenario, seed))
                                    policy, _ = classical.load_q_model(Path(artifact["path"]), job, q_spec, require_frozen=True)
                                else:
                                    policy, _ = load_checkpoint_policy(Path(artifact["path"]), device="cpu")
                            q_policy = policy if base == "q_routing" else None
                            q_before = hashlib.sha256(q_policy.q.tobytes()).hexdigest() if q_policy is not None else None
                            if direct:
                                policy = DirectDeliveryControlPolicy(policy)
                            for workload in WORKLOADS:
                                row, packets, _ = run_episode(scenario, load, name, policy, seed, workload)
                                row["direct_overrides"] = int(getattr(policy, "overrides", 0))
                                key = (scenario, workload)
                                signature = (row["arrival_sha256"], row["physical_sha256"])
                                if key in pairings and pairings[key] != signature:
                                    raise AssertionError("external workload pairing differs")
                                pairings[key] = signature
                                rows.append(row)
                                episodes_file.write(canonical(row) + "\n")
                                identity = {k: row[k] for k in ("scenario", "load", "policy", "policy_seed", "workload_seed")}
                                for packet in packets:
                                    packets_file.write(canonical({**identity, **packet}) + "\n")
                            episodes_file.flush()
                            if q_policy is not None and hashlib.sha256(q_policy.q.tobytes()).hexdigest() != q_before:
                                raise AssertionError("Q policy changed during evaluation")
                            print(f"{len(rows)}/2000 {scenario} {name} seed={seed} elapsed={time.monotonic()-started:.1f}s", flush=True)
        if len(rows) != manifest["expected_episodes"]:
            raise AssertionError("incomplete episode panel")
        for artifact in artifacts:
            if sha(artifact["path"]) != artifact["sha256"]:
                raise AssertionError("frozen checkpoint changed")
        for path, digest in manifest["source_sha256"].items():
            if sha(ROOT / path) != digest:
                raise AssertionError("source changed during run")
        summaries, (effects, gates) = aggregate(rows), analyze(rows, seeds)
        write_json(output / "summary.json", summaries)
        write_json(output / "paired_effects.json", effects)
        write_json(output / "development_gates.json", gates)
        export_csv(output / "summary.csv", summaries, ["scenario", "policy", "episodes", "policy_replicates", "delivery_ratio", "decision_switch_rate", "drop_rate", "backlog_rate"])
        export_csv(output / "paired_effects.csv", effects, ["scenario", "contrast", "mean_delivery_effect_pp", "crossed_95pct_interval_pp", "positive_seeds", "negative_seeds", "unchanged_seeds", "raw_seed_sign_flip_p"])
        write_json(output / "completion.json", {"status": "complete", "episodes": len(rows),
                   "exogenous_pairings_verified": len(pairings), "elapsed_seconds": time.monotonic()-started,
                   "input_hashes_unchanged": True, "development_screen_pass": all(g["development_screen_pass"] for g in gates),
                   "output_sha256": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
        print(canonical(gates))
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "completed_episodes": len(rows), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
