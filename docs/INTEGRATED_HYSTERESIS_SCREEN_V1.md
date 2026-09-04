# Integrated Hysteresis Screen v1

## Status and purpose

This protocol defines an **exploratory engineering screen**. It is not a
confirmatory experiment, it cannot support a paper claim, and every generated
artifact must record `paper_claim_allowed=false`.

The screen asks whether the actor-score hysteresis selected by the prior
post-hoc study remains useful when the same fixed bias is present during MAPPO
rollout collection, PPO log-probability recomputation, entropy calculation,
validation, checkpoint selection, and final greedy evaluation.

## Frozen policy change

Only eight new policies are trained. Every new policy uses the canonical
`with_congestion_context` environment and the following actor contract:

- candidate actor type: `shared_candidate_actor`;
- candidate feature dimension: `28`;
- route-switch feature index: `17`;
- additive cached-route logit bias: `beta=0.20`;
- candidate actor schema version: `1`.

The actor applies the bias after learned scoring and before the availability
mask. The same actor path must be used for sampled rollout actions, stored
rollout log probabilities, PPO current log probabilities, normalized entropy,
validation actions, and checkpoint evaluation. Evaluation must recover beta and
the switch-feature index from the checkpoint. A positive-beta checkpoint with
missing or inconsistent actor metadata is invalid.

## Training grid

- Scenarios: `medium_load`, `hotspot_high_load`.
- Policy seeds: `1710210210`, `2078783072`, `1047581915`, `1245825580`.
- New training jobs: `2 scenarios x 4 policy seeds = 8`.
- Budget: `20,000` environment steps per job.
- Batch size: `4`.
- Training workloads: `9001..9200`.
- Validation workloads: `32001..32020`.
- Validation frequency: every `20` rollout batches.
- Periodic checkpoint interval: `2,500` environment steps.

The validation panel is deliberately reused from the source congestion-context
screen. This makes checkpoint selection comparable to the frozen raw-context
checkpoints and isolates the effect of integrating beta during learning. It also
makes the study adaptive and exploratory; validation results cannot be treated
as fresh evidence.

## Frozen reference source

The source is
`experiments/archive/congestion-context-screen-20k-v1`. Before training or
evaluation, the runner must validate its self-hashed spec and training freeze,
the manifest links, the exact 16-checkpoint grid, every selected checkpoint
path, and every checkpoint SHA-256 digest.

The source supplies three evaluation arms without new training:

1. Frozen proposed checkpoints, evaluated without hysteresis.
2. Frozen congestion-context checkpoints, evaluated without hysteresis.
3. The same frozen congestion-context checkpoints with post-hoc actor-score
   hysteresis at `beta=0.20`.

The fourth arm is the newly trained integrated-hysteresis policy. Thus the final
grid contains `2 scenarios x 4 arms x 4 policy seeds = 32` independently
recoverable evaluation shards.

## Fresh test panel

All four arms are evaluated on workloads `37001..37050`, producing `1,600`
episode rows. This panel is disjoint from:

- source screening workloads `33001..33050`;
- hysteresis tuning workloads `34001..34020`;
- post-hoc hysteresis final workloads `35001..35050`.

The runner must create and self-hash the complete eight-checkpoint integrated
training freeze before evaluating any `37001..37050` workload. The test panel
must not be used to adjust beta, training settings, policy seeds, gates, or code.
Any such adjustment requires another newly frozen panel.

## Statistics and contrasts

The unit structure is crossed: four policy seeds by fifty workload seeds.
Uncertainty uses the repository's crossed bootstrap with `5,000` resamples.
Direction consistency and exact sign-flip inference operate on the four policy
seed means. With four seeds, the minimum attainable two-sided exact p-value is
`0.125`; p-values are descriptive and are not promotion gates.

The frozen contrasts are:

- raw context minus proposed;
- post-hoc hysteresis minus proposed;
- integrated hysteresis minus proposed;
- integrated hysteresis minus raw context;
- integrated hysteresis minus post-hoc hysteresis.

The hard decision gates for integrated hysteresis are:

- hotspot delivery difference versus proposed `>= +0.010`, with all four
  policy-seed differences positive;
- hotspot delivery difference versus post-hoc hysteresis `>= +0.005`;
- medium-load delivery difference versus proposed `>= -0.003`, with at least
  three policy-seed differences `>= -0.010`;
- mean switch ratio versus proposed `<= 1.02` in both scenarios; in each
  scenario at least three seed ratios must be `<= 1.05`, and none may exceed
  `1.15`;
- mean switch ratio versus raw context `<= 0.90` in both scenarios, with all
  four policy seeds improving in each scenario;
- medium-load mean average-delay ratio versus proposed `<= 1.01`; at least
  three seed ratios must be `<= 1.02`, and none may exceed `1.03`;
- medium-load mean P95-delay ratio versus proposed `<= 1.02`;
- medium-load class-2 delivery difference versus proposed `>= -0.002`; at
  least three seed differences must be `>= -0.010`, and none may be below
  `-0.015`;
- no other lower-is-better cost regression above `10%` versus proposed;
- no other traffic-class delivery regression above `0.020` versus proposed.

Passing every gate permits only a separately frozen 50k experiment. Failure of
any gate yields `do_not_advance`.

## Integrity and recovery

The immutable screen spec records the complete grids, source freeze, code
fingerprints, actor contract, gates, and paths and carries a canonical JSON
self-hash. The integrated training freeze contains exactly eight audited
selected checkpoints, their artifact hashes, the actor contract, its source
freeze link, and its own self-hash.

Each training and evaluation job has a status record, attempt history, lock,
and hashed artifact metadata. A completed training job is reused only after a
full checkpoint audit. A completed evaluation shard is reused only when its
metadata, checkpoint hash, workload grid, CSV schema, row count, and CSV hash
all match. Interrupted training can resume at job granularity; an interrupted
individual MAPPO job restarts from the beginning.

The final manifest records every evaluation CSV and metadata hash, all merged
artifact hashes and row counts, both training-freeze hashes, the statistical
contract, the decision, and a canonical self-hash.

## Commands

From the repository root with the project environment:

```powershell
F:\leo-venv\Scripts\python.exe src\run_integrated_hysteresis_screen.py --dry-run
F:\leo-venv\Scripts\python.exe src\run_integrated_hysteresis_screen.py --train-only --device cuda --max-parallel 2
F:\leo-venv\Scripts\python.exe src\run_integrated_hysteresis_screen.py --evaluate-only --device cuda --max-parallel 2
```

Omitting the mode flag performs training, freezes all eight new checkpoints,
and then evaluates all four arms. `--dry-run` audits the source and runtime and
prints the frozen plan without creating the output directory.
