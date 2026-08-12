"""Constellation-scale experiment for the LEO MAPPO routing policy.

Two modes:
  --mode transfer  (CPU, inference-only): load checkpoints trained at a SOURCE
                   scale (e.g. 24-sat) and evaluate each policy seed's actor
                   ZERO-SHOT on every target scale, plus Dijkstra/heuristic
                   oracles at each scale. Produces the transfer matrix.
  --mode train     (GPU): train MAPPO at each scale (flat critic, no_lifetime),
                   then eval + baselines. [Stubbed — implement when GPU free.]

Key design decisions (informed by the scale-collapse diagnostic):
  * The shared candidate actor is permutation-equivariant and neighbor-relative,
    so its observation (182-dim) is identical at every constellation size ->
    a checkpoint trained at scale A deploys at scale B with no architecture
    change. CTDE means only the (scale-invariant) actor is needed at execution.
  * Per-satellite offered load is held CONSTANT across scales (exogenous and
    initial packets scale with n_sats/24), so each satellite sees the same load.
  * Episode length scales with the constellation diameter (~sqrt(n_sats)),
    because at fixed 30 slots, packets cannot cross a large diameter and the
    delivery ratio collapses as an episode-length artifact (diagnostic showed
    ~72% of packets merely stuck in flight at 110 sats, not dropped).
  * Throughput (delivered/slot) is reported alongside delivery: it is the
    scale-invariant comparison metric (the delivery RATIO drifts with diameter;
    the MAPPO/Dijkstra throughput RATIO does not).

Outputs (experiments/IEEE-SCALE[-<tag>]/):
  transfer_matrix.csv  -- per (source_scale, target_scale, policy, seed) metrics
  aggregate_scale.csv  -- per (target_scale, policy) bootstrap means + 95% CI
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig, MULTIAGENT_LOADS
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_evaluation import (
    EpisodeMetrics,
    GlobalDijkstraPolicy,
    evaluate_policy,
    heuristic_policy,
    load_checkpoint_policy,
)

# (n_planes, sats_per_plane, label). 24 = the training/headline scale.
DEFAULT_SCALES = [
    (4, 6, "n24"),
    (6, 11, "n66"),
    (10, 11, "n110"),
    (6, 22, "n132"),
]
DEFAULT_SOURCE = (4, 6, "n24")  # where the transfer-source checkpoints were trained
BASE_N = 24
BASE_STEPS = 30


def scaled_workload(npl: int, spp: int, scenario: str, seed: int, variant: str):
    """Build a wrapper at this scale with constant per-sat load + diameter-scaled
    episode length. Returns CleanMARLLeoMultiAgentWrapper."""
    n = npl * spp
    scale = n / BASE_N
    base_init, base_exo = MULTIAGENT_LOADS[scenario]  # (initial, exogenous/slot) at 24
    ini = max(1, int(round(base_init * scale)))
    exo = max(1, int(round(base_exo * scale)))
    # episode length ~ diameter growth (sqrt of node count), floored at BASE_STEPS
    max_steps = max(BASE_STEPS, int(round(BASE_STEPS * math.sqrt(scale))))
    envc = EnvConfig(
        n_planes=npl,
        sats_per_plane=spp,
        scenario=SCENARIOS[scenario],
        seed=seed,
        max_steps=max_steps,
    )
    cfg = MultiAgentConfig(
        env=envc,
        initial_packets=ini,
        exogenous_packets_per_slot=exo,
        seed=seed,
        variant=variant,
    )
    return CleanMARLLeoMultiAgentWrapper(cfg=cfg)


def wrapper_factory(npl: int, spp: int, scenario: str, variant: str):
    def make(seed: int) -> CleanMARLLeoMultiAgentWrapper:
        return scaled_workload(npl, spp, scenario, seed, variant)
    return make


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bootstrap_ci(values: list[float], seeds=(4001,)):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seeds[0])
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(5000)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def find_source_checkpoints(source_dir: Path, scenario: str) -> list[Tuple[int, Path]]:
    """Locate validation_best.pt for each policy seed under source_dir/checkpoints/<scenario>/.

    Handles both checkpoint layouts produced by run_exp004_mappo.py:
      full variant:        checkpoints/<scenario>/seed_X/<rundir>/validation_best.pt
      non-full variant:    checkpoints/<scenario>/<variant>/seed_X/<rundir>/validation_best.pt
    """
    out = []
    root = source_dir / "checkpoints" / scenario
    seed_dirs = sorted(root.glob("seed_*")) + sorted(root.glob("*/seed_*"))
    seen = set()
    for seed_dir in seed_dirs:
        if seed_dir in seen:
            continue
        seen.add(seed_dir)
        try:
            seed = int(seed_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        cks = sorted(seed_dir.glob("*/validation_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cks:
            out.append((seed, cks[0]))
    return out


def aggregate_scale(rows: list[EpisodeMetrics], scales_meta) -> list[dict]:
    metrics = [
        "delivery_ratio", "drop_rate", "throughput_packets_per_slot",
        "average_delay_slots", "p95_delay_slots", "mean_queue_packets",
    ]
    out = []
    for (npl, spp, label) in scales_meta:
        for policy in sorted({r.policy for r in rows if r.scenario == label}):
            sel = [r for r in rows if r.scenario == label and r.policy == policy]
            if not sel:
                continue
            for m in metrics:
                vals = [float(getattr(r, m)) for r in sel]
                mean, lo, hi = bootstrap_ci(vals)
                out.append({
                    "target_scale": label, "n_sats": npl * spp, "policy": policy,
                    "metric": m, "n": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi,
                })
    return out


def run_transfer(args):
    scales = DEFAULT_SCALES
    scenario = args.scenario
    device = args.device
    wseeds = list(range(args.workload_seed_start, args.workload_seed_start + args.workload_seeds))
    sources = find_source_checkpoints(args.source_dir, scenario)
    if not sources:
        raise FileNotFoundError(f"no checkpoints under {args.source_dir}/checkpoints/{scenario}/seed_*")
    print(f"transfer mode: {len(sources)} source seeds from {args.source_dir} ({scenario}), "
          f"-> {len(scales)} scales, {len(wseeds)} workload seeds each, device={device}")

    all_rows: list[EpisodeMetrics] = []
    # Load each source actor once
    actors = []
    for seed, ckpt in sources:
        policy, info = load_checkpoint_policy(ckpt, device=device)
        variant = info.get("args", {}).get("leo_variant", "full") if isinstance(info, dict) else "full"
        actors.append((seed, policy, variant))
        print(f"  loaded source seed={seed} <- {ckpt.name} (variant={variant})")

    for (npl, spp, label) in scales:
        n = npl * spp
        print(f"\n=== target scale {label} (n={n}, pl={npl}x{spp}) ===")
        # zero-shot transfer: every source actor on this scale
        for seed, policy, variant in actors:
            rows = evaluate_policy(
                label, f"mappo_transfer_n24", policy, seed, wseeds,
                wrapper_factory=wrapper_factory(npl, spp, scenario, variant), variant=variant,
            )
            all_rows.extend(rows)
        # oracles / baselines at this scale (variant-agnostic)
        fct = wrapper_factory(npl, spp, scenario, "full")
        all_rows.extend(evaluate_policy(label, "global_dijkstra", GlobalDijkstraPolicy(), -1, wseeds, wrapper_factory=fct))
        all_rows.extend(evaluate_policy(label, "full_heuristic", heuristic_policy("full_heuristic", seed=1234), -1, wseeds, wrapper_factory=fct))
        # quick interim readout
        def mean(pol, m="delivery_ratio"):
            xs = [float(getattr(r, m)) for r in all_rows if r.scenario == label and r.policy == pol]
            return float(np.mean(xs)) if xs else float("nan")
        def mthrough(pol):
            xs = [float(r.throughput_packets_per_slot) for r in all_rows if r.scenario == label and r.policy == pol]
            return float(np.mean(xs)) if xs else float("nan")
        print(f"  delivery: MAPPO-transfer={mean('mappo_transfer_n24'):.3f}  "
              f"Dijkstra={mean('global_dijkstra'):.3f}  heuristic={mean('full_heuristic'):.3f}")
        print(f"  thru/slot: MAPPO={mthrough('mappo_transfer_n24'):.2f}  "
              f"Dijkstra={mthrough('global_dijkstra'):.2f}  ratio={mthrough('mappo_transfer_n24')/max(1e-9,mthrough('global_dijkstra')):.3f}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "transfer_matrix.csv", [asdict(r) for r in all_rows])
    write_csv(out / "aggregate_scale.csv", aggregate_scale(all_rows, scales))
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "constellation_scale_transfer", "mode": "transfer",
        "source_dir": str(args.source_dir), "source_scale": "n24",
        "scales": [{"label": l, "n_planes": p, "sats_per_plane": s, "n_sats": p * s} for p, s, l in scales],
        "scenario": scenario, "policy_seeds": [s for s, _ in sources],
        "workload_seeds": wseeds,
        "workload": "constant per-sat load (proportional exogenous), episode~sqrt(n) diameter",
    }, indent=2), encoding="utf-8")
    print(f"\n=> wrote {out/'transfer_matrix.csv'} + {out/'aggregate_scale.csv'}")


def run_train(args):
    raise NotImplementedError(
        "train mode: train MAPPO at each scale (flat critic, no_lifetime) on GPU. "
        "Implement when GPU is free — reuses train_one with a scaled cfg + --leo-n-planes/--leo-sats-per-plane "
        "overrides threaded into cleanmarl/mappo.py."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["transfer", "train"], default="transfer")
    p.add_argument("--source-dir", type=Path, default=Path("experiments/IEEE-REPRO-CHECK"))
    p.add_argument("--output", type=Path, default=Path("experiments/IEEE-SCALE"))
    p.add_argument("--scenario", default="medium_load")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workload-seed-start", type=int, default=16001)
    p.add_argument("--workload-seeds", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "transfer":
        run_transfer(args)
    else:
        run_train(args)
