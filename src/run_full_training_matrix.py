"""Run the 5-scenario x 3-seed full MAPPO training matrix with resume support."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace

from run_exp004_mappo import ALL_SCENARIOS, POLICY_SEEDS, mode_config, train_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/EXP-004-FULL"))
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--scenarios", nargs="*", default=ALL_SCENARIOS)
    args = parser.parse_args()
    config = mode_config("full")
    train_args = SimpleNamespace(
        output=args.output,
        cleanmarl=args.cleanmarl,
        project=args.project,
        skip_training=False,
        device=args.device,
    )
    jobs = [
        (scenario, seed) for scenario in args.scenarios for seed in POLICY_SEEDS
    ]
    checkpoints = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        future_to_job = {
            executor.submit(
                train_one, train_args, config, scenario, seed, "full", []
            ): (scenario, seed)
            for scenario, seed in jobs
        }
        for future in as_completed(future_to_job):
            scenario, seed = future_to_job[future]
            key = f"{scenario}/seed_{seed}"
            try:
                checkpoints[key] = str(future.result())
                print(f"completed {key}", flush=True)
            except Exception as error:
                failures[key] = repr(error)
                print(f"failed {key}: {error}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "EXP-004-FULL-training-matrix",
        "config": config,
        "scenarios": args.scenarios,
        "policy_seeds": POLICY_SEEDS,
        "max_parallel": args.max_parallel,
        "checkpoints": checkpoints,
        "failures": failures,
    }
    (args.output / "training_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"{len(failures)} training jobs failed")
    print(f"Full training matrix complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
