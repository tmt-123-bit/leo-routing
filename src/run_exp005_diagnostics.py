"""EXP-005: reproducible diagnostics for EXP-004 training runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp004", type=Path, default=Path("experiments/EXP-004"))
    parser.add_argument("--output", type=Path, default=Path("experiments/EXP-005"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    update_rows = []
    validation_rows = []
    run_rows = []

    manifest_path = args.exp004 / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dirs = sorted(
            {Path(path).parent for path in manifest.get("checkpoints", {}).values()}
        )
        metrics_paths = [run_dir / "training_metrics.jsonl" for run_dir in run_dirs]
    else:
        metrics_paths = sorted(
            args.exp004.glob("checkpoints/**/training_metrics.jsonl")
        )
    for metrics_path in metrics_paths:
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        run_dir = metrics_path.parent
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        updates = [row for row in records if row["record_type"] == "training_update"]
        validations = [row for row in records if row["record_type"] == "validation"]
        run_id = run_dir.name
        for row in updates:
            update_rows.append({"run_id": run_id, **row})
        for row in validations:
            validation_rows.append({"run_id": run_id, **row})

        tail_count = max(1, int(np.ceil(0.1 * len(updates))))
        tail = updates[-tail_count:]
        entropy_tail = np.mean([row["normalized_entropy"] for row in tail])
        actor_grad_tail = np.mean([row["actor_gradient_pre_clip"] for row in tail])
        critic_grad_tail = np.mean([row["critic_gradient_pre_clip"] for row in tail])
        selected = run_dir / "validation_best.pt"
        final = run_dir / "final.pt"
        validation_tuples = {
            (
                round(row["delivery_ratio"], 10),
                round(row["mean_reward"], 10),
                round(row["drop_rate"], 10),
                round(row["average_delay_slots"], 10),
            )
            for row in validations
        }
        run_rows.append(
            {
                "run_id": run_id,
                "scenario": config["env_name"],
                "policy_seed": config["seed"],
                "training_updates": len(updates),
                "validation_points": len(validations),
                "unique_validation_tuples": len(validation_tuples),
                "tail_normalized_entropy": float(entropy_tail),
                "tail_actor_gradient_pre_clip": float(actor_grad_tail),
                "tail_critic_gradient_pre_clip": float(critic_grad_tail),
                "entropy_below_0_01": bool(entropy_tail < 0.01),
                "actor_gradient_below_0_01": bool(actor_grad_tail < 0.01),
                "critic_clipping_active": bool(
                    any(
                        row["critic_gradient_pre_clip"]
                        > row["critic_gradient_post_clip"] + 1e-8
                        for row in updates
                    )
                ),
                "selected_checkpoint_sha256": sha256(selected) if selected.exists() else "",
                "final_checkpoint_sha256": sha256(final) if final.exists() else "",
                "selected_differs_from_final": bool(
                    selected.exists()
                    and final.exists()
                    and sha256(selected) != sha256(final)
                ),
            }
        )

    write_csv(args.output / "training_updates.csv", update_rows)
    write_csv(args.output / "validation_trace.csv", validation_rows)
    write_csv(args.output / "run_diagnostics.csv", run_rows)
    summary = {
        "experiment": "EXP-005",
        "source": str(args.exp004.resolve()),
        "runs": len(run_rows),
        "entropy_collapse_runs": sum(row["entropy_below_0_01"] for row in run_rows),
        "low_actor_gradient_runs": sum(
            row["actor_gradient_below_0_01"] for row in run_rows
        ),
        "critic_clipping_runs": sum(row["critic_clipping_active"] for row in run_rows),
        "validation_indistinguishable_runs": sum(
            row["unique_validation_tuples"] <= 1 for row in run_rows
        ),
        "threshold_note": (
            "0.01 thresholds are diagnostic flags inherited from the prior audit; "
            "they are not universal convergence criteria."
        ),
    }
    (args.output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"EXP-005 complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
