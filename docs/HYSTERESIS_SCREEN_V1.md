# Actor-Score Hysteresis Screen v1

Status: frozen exploratory optimization screen. This protocol is not a
confirmatory experiment and cannot support a final IEEE claim.

## Objective

Stabilize the already-trained `with_congestion_context` checkpoints without
using the viewed `33001..33050` congestion-screen workloads for tuning. The
controller gives a fixed `beta` preference to a feasible cached next hop. A new
next hop is selected only when its raw actor logit advantage over the
cached-route logit is at least `beta`. `beta=0` exactly reproduces raw greedy
selection, including its tie behavior.

Candidate feature 17 is the existing `is_route_switch` indicator. Before a
cache exists, every feasible candidate receives the same bonus and the action
ordering is unchanged. When the cached next hop is infeasible, it receives no
bonus and is never made feasible by the controller.

No policy is retrained. All checkpoints and their SHA-256 hashes come from the
immutable training freeze of `CONGESTION-CONTEXT-SCREEN-v1`.

## Fixed tuning stage

- Scenarios: `medium_load`, `hotspot_high_load`.
- Policy seeds: the same four independently trained seeds as the source screen.
- Tuning workloads: `34001..34020`.
- Fixed candidate bonuses: `0.02, 0.05, 0.10, 0.20, 0.40` actor-logit units.
- References evaluated on the same paired grid: the original `proposed`
  checkpoint and raw `with_congestion_context` checkpoint (`beta=0`).
- No seed replacement, early termination, or performance-based grid extension.

A candidate is eligible only if all of these gates hold on the complete tuning
grid:

1. Mean routing switches are no greater than `1.05 * proposed` in both
   scenarios.
2. Hotspot delivery is no more than `0.002` absolute below raw congestion
   context.
3. Medium-load delivery is no more than `0.003` absolute below `proposed`.

The smallest eligible beta is selected. If none is eligible, the optimization
is rejected and the final workload panel remains unopened. The selected beta,
source checkpoint freeze, complete tuning table, selection table, and hashes
are written to an immutable selection freeze before final evaluation.

## Fixed final stage

- Final workloads: `35001..35050`, not used in tuning.
- Policies: `proposed`, raw congestion context, and congestion context with the
  single frozen beta.
- Complete grid: `2 scenarios x 3 policies x 4 policy seeds x 50 workloads =
  1,200` episodes.
- Primary comparisons: stabilized minus proposed, raw context minus proposed,
  and stabilized minus raw context.
- Uncertainty: crossed policy-seed/workload bootstrap intervals and exact
  policy-seed sign-flip tests, with the policy seed as the algorithmic
  replication unit.

The optimization is accepted for later method development only when the same
three tuning gates hold on the untouched final panel. Advancement to a new
separately frozen training experiment additionally requires the original
congestion-screen effect gate: hotspot delivery at least `+0.010` over
`proposed`, at least three of four hotspot policy-seed means positive,
medium-load delivery no worse than `-0.003`, no greater than 10% regression in
predeclared costs, and no traffic-class delivery loss greater than `0.020`.

Whatever the outcome, this two-stage 20k checkpoint analysis remains
exploratory. A successful result is a design-selection result, not final paper
evidence.

## Integrity and recovery

The runner verifies the source v1 spec, manifest, training freeze, checkpoint
paths, checkpoint hashes, and checkpoint schemas before evaluation. It writes
an immutable protocol spec, hash-bound per-job shards, a tuning selection
freeze before opening the final panel, invocation records, and a final artifact
manifest. Completed shards are reused only after schema, grid, metadata, and
SHA-256 validation.
