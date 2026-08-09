"""Zero-shot routing replay on a frozen SGP4/TLE topology snapshot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from hypatia_topology_provider_stub import HypatiaTopologyProvider
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig
from mappo_evaluation import (
    evaluate_policy,
    heuristic_policy,
    load_checkpoint_policy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/TLE-REPLAY"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provider = HypatiaTopologyProvider.from_csv(args.links)

    def wrapper_factory(seed: int):
        env_cfg = EnvConfig(
            seed=seed,
            scenario=SCENARIOS["medium_load"],
            topology_provider=provider,
        )
        cfg = MultiAgentConfig(
            env=env_cfg,
            initial_packets=12,
            exogenous_packets_per_slot=6,
            seed=seed,
        )
        return CleanMARLLeoMultiAgentWrapper(cfg=cfg, seed=seed)

    policies = {
        "delay_only": (heuristic_policy("delay_only"), -1),
        "full_heuristic": (heuristic_policy("full_heuristic"), -1),
        "random": (heuristic_policy("random", seed=1234), -1),
    }
    checkpoint_metadata = None
    if args.checkpoint:
        policy, checkpoint_metadata = load_checkpoint_policy(args.checkpoint)
        policies["mappo_zero_shot"] = (
            policy,
            int(checkpoint_metadata["args"]["seed"]),
        )

    workload_seeds = range(12001, 12001 + args.episodes)
    rows = []
    for name, (policy, policy_seed) in policies.items():
        rows.extend(
            evaluate_policy(
                "tle_starlink_24",
                name,
                policy,
                policy_seed,
                workload_seeds,
                wrapper_factory=wrapper_factory,
            )
        )
    dictionaries = [asdict(row) for row in rows]
    with (args.output / "episode_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)

    summary = []
    for policy in policies:
        selected = [row for row in rows if row.policy == policy]
        summary.append(
            {
                "policy": policy,
                "episodes": len(selected),
                "delivery_ratio_mean": float(
                    np.mean([row.delivery_ratio for row in selected])
                ),
                "drop_rate_mean": float(np.mean([row.drop_rate for row in selected])),
                "average_delay_slots_mean": float(
                    np.mean([row.average_delay_slots for row in selected])
                ),
                "p95_delay_slots_mean": float(
                    np.mean([row.p95_delay_slots for row in selected])
                ),
                "throughput_mean": float(
                    np.mean([row.throughput_packets_per_slot for row in selected])
                ),
            }
        )
    with (args.output / "aggregate_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "TLE-REPLAY",
                "links": str(args.links.resolve()),
                "episodes": args.episodes,
                "workload_seed_range": [12001, 12000 + args.episodes],
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "interpretation": (
                    "MAPPO is evaluated zero-shot when a checkpoint is supplied; "
                    "this is a topology-transfer test, not retraining on TLE data."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"TLE replay complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
