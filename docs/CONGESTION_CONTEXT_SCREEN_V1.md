# Congestion Context Screen v1

Status: frozen exploratory development screen. This protocol is not a
confirmatory experiment and its results cannot be used as final IEEE evidence.

## Objective

Test whether `with_congestion_context` is worth a later, separately frozen
50k head-to-head against `proposed`. The candidate appends queue trend and a
current-graph two-hop continuation-headroom proxy. It is not added to the
existing 540-job ablation matrix.

## Fixed design

- Scenarios: `medium_load`, `hotspot_high_load`.
- Variants: `proposed`, `with_congestion_context`.
- Training budget: 20,000 environment steps for every cell.
- Policy seeds: four SHA-256-derived seeds from namespace
  `CONGESTION-CONTEXT-SCREEN-v1-policy-seed-` and counters 0 through 3.
- Training workloads: the trainer's existing `9001..9200` panel.
- Validation workloads: `32001..32020`, used only for checkpoint selection.
- Screening workloads: `33001..33050`, evaluated once after all 16 selected
  checkpoints and their hashes are frozen.
- PPO settings are identical between variants: batch size 4, 3 epochs, 4
  minibatches, validation every 20 rollouts, and checkpointing every 2,500
  steps.
- No performance-based early termination and no failed-seed replacement.

The four policy seeds are:

```text
1710210210
2078783072
1047581915
1245825580
```

They are computed as:

```text
int.from_bytes(SHA256(namespace + counter)[0:4], "big") & 0x7fffffff
```

## Analysis

The fixed contrast is `with_congestion_context - proposed`. Every scenario
and metric must contain a complete matched 4 policy-seed x 50 workload matrix.
Report the crossed pigeonhole-bootstrap interval and the exact two-sided
policy-seed sign-flip test implemented in `hierarchical_statistics.py`.

With four policy seeds, the smallest attainable two-sided exact p-value is
0.125. P-values and intervals are uncertainty descriptions only; advancement
is based on the effect-size and consistency gates below.

Primary metric: delivery ratio in both scenarios.

Predeclared cost diagnostics: average delay, p95 delay, routing switches,
control overhead, and traffic-class delivery ratios. Episode reward is not
evidence of routing superiority.

## Decision rule

Promote only when all conditions hold:

1. Hotspot delivery improves by at least 0.010 absolute and at least 3 of 4
   policy-seed means improve.
2. Medium-load delivery is no worse than -0.003 absolute, and at least 3 of 4
   policy-seed differences exceed -0.010.
3. Neither scenario has a greater than 10% relative regression in average
   delay, p95 delay, routing switches, or control overhead.
4. Neither scenario loses more than 0.020 absolute delivery in any traffic
   class.

Reject when hotspot delivery is non-positive, medium-load delivery is below
-0.010, hotspot direction is supported by at most 2 of 4 seeds, medium-load
non-inferiority is supported by fewer than 3 of 4 seeds, or a predeclared cost
gate fails. All other complete outcomes are `inconclusive`.

An inconclusive screen may receive at most four additional fresh policy seeds
under a new frozen extension. Thresholds and the `33001..33050` panel cannot
be changed or reused to tune the method after results are viewed.

## Integrity and recovery

The runner writes an immutable spec hash, per-job training/evaluation states,
selected checkpoint and artifact hashes, an immutable training freeze, and
one evaluation shard per job. Completed cells are reused only after every hash
and schema check passes. An interrupted cell restarts from zero because the
current trainer does not support intra-run checkpoint continuation.

The output directory must remain separate from `experiments/ablation-50k-v2`.
The full 800-row evaluation table is required before a decision is emitted.
