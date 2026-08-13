"""Train a disjoint ablation subset without writing evaluation summaries."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace

from run_ablation_experiments import SCENARIOS, VARIANTS, mode_config, overrides_for
from run_exp004_mappo import POLICY_SEEDS, train_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=POLICY_SEEDS, default=POLICY_SEEDS
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--manifest-name", required=True)
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
        (scenario, variant, seed)
        for scenario in args.scenarios
        for variant in args.variants
        for seed in args.seeds
    ]

    def train(job):
        scenario, variant, seed = job
        checkpoint = train_one(
            train_args,
            config,
            scenario,
            seed,
            variant=variant,
            trainer_overrides=overrides_for(variant),
        )
        return f"{scenario}/{variant}/seed_{seed}", str(checkpoint)

    checkpoints = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        futures = {executor.submit(train, job): job for job in jobs}
        for future in as_completed(futures):
            scenario, variant, seed = futures[future]
            key = f"{scenario}/{variant}/seed_{seed}"
            try:
                result_key, checkpoint = future.result()
                checkpoints[result_key] = checkpoint
                print(f"completed {result_key}", flush=True)
            except Exception as error:
                failures[key] = repr(error)
                print(f"failed {key}: {error}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "controlled_ablation_training_shard",
        "config": config,
        "scenarios": args.scenarios,
        "variants": args.variants,
        "policy_seeds": args.seeds,
        "checkpoints": checkpoints,
        "failures": failures,
    }
    path = args.output / f"training_shard_{args.manifest_name}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} shard jobs failed")
    print(f"Ablation training shard complete: {path.resolve()}")


if __name__ == "__main__":
    main()
