# Decoupled Adaptive Hysteresis Preflight v2

## Status

This protocol is a short, diagnostic engineering preflight. It is exploratory,
not confirmatory, and cannot support a paper claim or a promotion decision. Its
purpose is to check whether the v2 mechanism behaves in the intended direction
before a separately frozen, adequately replicated experiment is designed.

## Frozen mechanism

The candidate actor uses schema version `2` and mode `decoupled_adaptive`.
Candidate feature `17` is zeroed before learned encoding, pooling, and scoring,
so PPO cannot learn a switch-feature coefficient that cancels the fixed output
prior. The original feature `17` is then used to apply the cached-route bonus.

- Base hysteresis beta: `0.20`, carried forward from the v1 freeze.
- Urgency feature index: `20`.
- Urgency relief: `0.50`.
- Class-2 feature index: `23`.
- Class-2 relief: `0.0`.
- Candidate feature schema ID: `leo_multi_candidate_features_v1_dim_28`.
- Candidate feature schema SHA-256:
  `c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503`.

The urgency value is a mechanism-level, pre-test choice. At maximum normalized
urgency, the effective beta remains `0.10`, preserving half of the stability
prior while permitting more responsiveness. It was not chosen using this
preflight's validation or test outcomes and is not declared a final tuned
hyperparameter. Class-2 relief is frozen at zero because enabling it after
viewing v1 test results would be post-hoc tuning.

The schema ID and hash are obtained from the canonical
`with_congestion_context` wrapper. The runner compares the live wrapper contract
to these frozen literals before training and requires exact equality in every
schema-v2 checkpoint, run manifest, training freeze, and evaluation contract.

## Short training grid

- Scenarios: `medium_load`, `hotspot_high_load`.
- Policy seeds: `1710210210`, `2078783072`.
- Training jobs: `2 scenarios x 2 policy seeds = 4`.
- Budget: `10,000` environment steps per job.
- Training workload seeds: `9001..9200`.
- Batch size: `4`.
- Validation every `10` rollout batches.
- Periodic checkpoint interval: `2,500` environment steps.

The two policy seeds are retained from the frozen source so all four evaluation
arms can be paired. Two seeds are insufficient for a paper-level algorithmic
claim; this reduction is solely for a faster mechanism check.

## Validation-only checkpoint selection

Checkpoint selection uses only the fresh validation panel `41001..41010`.
After all validation candidates have been evaluated:

1. Find the maximum validation delivery ratio.
2. Retain candidates no more than `0.003` below that maximum.
3. Within those candidates, find the maximum class-2 delivery ratio and retain
   candidates no more than `0.010` below it.
4. Select the candidate with the fewest routing switches, followed by frozen
   deterministic tie-breaks.

Every candidate checkpoint and the final selection record are retained. The
test panel is not consulted during selection.

## Fresh diagnostic test panel

All four frozen arms are evaluated on `42001..42025`. The validation and test
panels are mutually disjoint and disjoint from exposed panels `32001..32020`,
`33001..33050`, `34001..34020`, `35001..35050`, and `37001..37050`.

The evaluation grid has `2 scenarios x 4 arms x 2 policy seeds = 16` shards and
`400` episode rows. The arms are the frozen proposed policy, frozen raw-context
policy, frozen post-hoc beta-0.20 policy, and newly trained schema-v2 policy.

## Statistics and reporting limit

Crossed policy-seed by workload-seed summaries and crossed bootstrap intervals
are retained for diagnostics. Exact sign-flip p-values are raw and exploratory.
No Holm adjustment and no Benjamini-Hochberg adjustment is applied or claimed.
With only two policy seeds, inferential resolution is especially weak.

Every paired-effect CSV row is labeled
`diagnostic_decoupled_hysteresis_preflight_v2`, carries
`inferential_status=raw_exploratory_only` and `confirmatory=false`, and exposes
only `raw_p_value`. The runner rejects rows containing Holm, BH, adjusted,
multiplicity, or Wilcoxon p-value fields before writing the CSV.

The final decision label is always
`diagnostic_only_no_promotion_decision`. Any formal follow-up requires a new
protocol, more independent policy seeds, a newly frozen test panel, and a
statistical plan that is implemented consistently in both CSV output and the
manifest.
