"""Train and evaluate MAPPO directly on a frozen TLE topology sequence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import stats

from cleanmarl_tle_wrapper import CleanMARLTLEWrapper
from mappo_evaluation import (
    GlobalDijkstraPolicy,
    evaluate_policy,
    heuristic_policy,
    load_checkpoint_policy,
)
from run_exp004_mappo import METRICS, aggregate_rows


SEEDS = [7, 42, 1024]


def find_checkpoint(root: Path) -> Path | None:
    candidates = sorted(root.glob("*/validation_best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_fingerprint(args, config: dict, seed: int) -> dict:
    code_paths = {
        "tle_entrypoint": args.project / "cleanmarl_mappo_tle.py",
        "tle_wrapper": args.project / "cleanmarl_tle_wrapper.py",
        "multiagent_wrapper": args.project / "cleanmarl_leo_multiagent_wrapper.py",
        "environment": args.project / "leo_multiagent_env.py",
        "design": args.project / "mappo_design.py",
    }
    return {
        "topology_csv": str(args.links.resolve()),
        "topology_sha256": sha256(args.links),
        "config": config,
        "seed": seed,
        "code_sha256": {
            name: sha256(path.resolve()) for name, path in code_paths.items()
        },
    }


def paired_tle_tests(rows) -> list[dict]:
    output = []
    for topology in sorted({row.scenario for row in rows}):
        for baseline in ["global_dijkstra", "full_heuristic"]:
            for metric in METRICS:
                learned = {}
                reference = {}
                for row in rows:
                    if row.scenario != topology:
                        continue
                    target = learned if row.policy == "mappo_tle_trained" else reference
                    if row.policy in {"mappo_tle_trained", baseline}:
                        target.setdefault(row.workload_seed, []).append(
                            getattr(row, metric)
                        )
                common = sorted(set(learned) & set(reference))
                learned_values = np.asarray(
                    [np.mean(learned[seed]) for seed in common], dtype=float
                )
                reference_values = np.asarray(
                    [np.mean(reference[seed]) for seed in common], dtype=float
                )
                differences = learned_values - reference_values
                try:
                    test = stats.wilcoxon(differences)
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                output.append(
                    {
                        "topology": topology,
                        "baseline": baseline,
                        "metric": metric,
                        "paired_workloads": len(common),
                        "mappo_mean": float(learned_values.mean()),
                        "baseline_mean": float(reference_values.mean()),
                        "mean_difference": float(differences.mean()),
                        "median_difference": float(np.median(differences)),
                        "wilcoxon_statistic": statistic,
                        "raw_p_value": p_value,
                    }
                )
    ordered = sorted(range(len(output)), key=lambda index: output[index]["raw_p_value"])
    adjusted = [1.0] * len(output)
    running = 1.0
    for rank in range(len(output), 0, -1):
        index = ordered[rank - 1]
        running = min(
            running,
            output[index]["raw_p_value"] * len(output) / rank,
        )
        adjusted[index] = min(1.0, running)
    for row, value in zip(output, adjusted):
        row["benjamini_hochberg_p"] = value
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--train-topology-name", default="tle_starlink_24")
    parser.add_argument("--cross-links", type=Path)
    parser.add_argument("--cross-topology-name", default="tle_oneweb_24")
    parser.add_argument("--output", type=Path, default=Path("experiments/TLE-TRAINED"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = (
        {
            "timesteps": 20000,
            "validation_episodes": 50,
            "test_episodes": 50,
            "eval_every": 20,
            "batch_size": 4,
        }
        if args.mode == "full"
        else {
            "timesteps": 600,
            "validation_episodes": 20,
            "test_episodes": 20,
            "eval_every": 4,
            "batch_size": 2,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)

    def train(seed: int) -> Path:
        root = args.output / "checkpoints" / f"seed_{seed}"
        root.mkdir(parents=True, exist_ok=True)
        fingerprint = training_fingerprint(args, config, seed)
        fingerprint_path = root / "code_fingerprint.json"
        existing = find_checkpoint(root)
        fingerprint_matches = (
            fingerprint_path.exists()
            and json.loads(fingerprint_path.read_text(encoding="utf-8"))
            == fingerprint
        )
        if existing is not None and fingerprint_matches:
            return existing
        command = [
            sys.executable,
            str(args.project / "cleanmarl_mappo_tle.py"),
            "--env-type", "leo_tle",
            "--env-name", "medium_load",
            "--leo-project-path", str(args.project),
            "--leo-topology-csv", str(args.links.resolve()),
            "--seed", str(seed),
            "--batch-size", str(config["batch_size"]),
            "--total-timesteps", str(config["timesteps"]),
            "--epochs", "3",
            "--num-minibatches", "4",
            "--eval-steps", str(config["eval_every"]),
            "--num-eval-ep", str(config["validation_episodes"]),
            "--save-every-steps", "5000",
            "--checkpoint-dir", str(root.resolve()),
            "--run-tag", "TLE-TRAINED",
            "--device", args.device,
        ]
        completed = subprocess.run(
            command,
            cwd=args.project,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        (root / "trainer.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(f"TLE training failed for seed {seed}")
        selected = find_checkpoint(root)
        if selected is None:
            raise RuntimeError(f"no TLE validation checkpoint for seed {seed}")
        fingerprint_path.write_text(
            json.dumps(fingerprint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return selected

    checkpoints = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = {executor.submit(train, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            checkpoints[seed] = future.result()
            print(f"trained TLE seed {seed}", flush=True)

    workload_seeds = range(14001, 14001 + config["test_episodes"])
    rows = []

    def evaluate_topology(topology_name: str, links: Path) -> None:
        def wrapper_factory(seed: int):
            return CleanMARLTLEWrapper(
                topology_csv=links,
                scenario="medium_load",
                seed=seed,
            )

        for seed, checkpoint in sorted(checkpoints.items()):
            policy, _ = load_checkpoint_policy(checkpoint, device=args.device)
            rows.extend(
                evaluate_policy(
                    topology_name,
                    "mappo_tle_trained",
                    policy,
                    seed,
                    workload_seeds,
                    wrapper_factory=wrapper_factory,
                )
            )
        for name, policy in [
            ("global_dijkstra", GlobalDijkstraPolicy()),
            ("full_heuristic", heuristic_policy("full_heuristic")),
        ]:
            rows.extend(
                evaluate_policy(
                    topology_name,
                    name,
                    policy,
                    -1,
                    workload_seeds,
                    wrapper_factory=wrapper_factory,
                )
            )

    evaluate_topology(args.train_topology_name, args.links)
    if args.cross_links is not None:
        evaluate_topology(args.cross_topology_name, args.cross_links)
    dictionaries = [asdict(row) for row in rows]
    write_rows(args.output / "episode_metrics.csv", dictionaries)
    write_rows(args.output / "aggregate_metrics.csv", aggregate_rows(rows))
    write_rows(args.output / "paired_tests.csv", paired_tle_tests(rows))
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "TLE-trained-MAPPO",
                "mode": args.mode,
                "links": str(args.links.resolve()),
                "links_sha256": hashlib.sha256(args.links.read_bytes()).hexdigest(),
                "train_topology_name": args.train_topology_name,
                "cross_links": (
                    str(args.cross_links.resolve()) if args.cross_links else None
                ),
                "cross_links_sha256": (
                    hashlib.sha256(args.cross_links.read_bytes()).hexdigest()
                    if args.cross_links
                    else None
                ),
                "cross_topology_name": (
                    args.cross_topology_name if args.cross_links else None
                ),
                "cross_validation": (
                    "zero-shot: Starlink-trained checkpoints are evaluated on the "
                    "cross topology without fine-tuning"
                    if args.cross_links
                    else None
                ),
                "config": config,
                "policy_seeds": SEEDS,
                "test_seed_range": [14001, 14000 + config["test_episodes"]],
                "checkpoints": {str(seed): str(path) for seed, path in checkpoints.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"TLE training experiment complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
