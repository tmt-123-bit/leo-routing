# Switch-Regularized MAPPO Preflight v3

## Status

This is a short diagnostic engineering preflight. It is exploratory, not
confirmatory, and cannot support a paper claim or a promotion decision. The v2
test panel was used only to identify the failure mechanism: most excess route
switches occurred while the cached next hop remained feasible. No v3 outcome
was available when this protocol and its fresh panels were frozen.

## Frozen mechanism

The candidate actor retains the schema-v2 decoupled adaptive hysteresis path:
feature `17` is removed from learned encoding and a fixed beta-`0.20` cached
route prior is applied after scoring. Schema version `3` adds a training-only
regularizer for avoidable switch probability.

For each active decision, the regularizer is eligible only when the cached next
hop and at least one switching alternative are both feasible. First-use states,
forced reroutes, masked actions, padding, and inactive agents are excluded. For
eligible decisions:

```text
actor_loss = ppo_actor_loss
             + 0.10 * sum(policy_probability[switching actions])
```

The coefficient `0.10` is a mechanism-level pre-test choice, not the result of a
coefficient sweep and not a final tuned hyperparameter. It is frozen before the
validation panel `43001..43010` and test panel `44001..44025` are evaluated.

Other frozen fields remain:

- Hysteresis beta: `0.20`.
- Urgency feature index and relief: `20`, `0.50`.
- Class-2 feature index and relief: `23`, `0.0`.
- Candidate feature schema ID: `leo_multi_candidate_features_v1_dim_28`.
- Candidate feature schema SHA-256:
  `c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503`.

Every accepted transmission is also classified as an avoidable switch, forced
switch, retained opportunity, or a decision without a switch opportunity.
Existing `routing_switches` semantics are unchanged.

## Short training grid

- Scenarios: `medium_load`, `hotspot_high_load`.
- Policy seeds: `1710210210`, `2078783072`.
- Training jobs: `2 scenarios x 2 policy seeds = 4`.
- Budget: `6,000` environment steps per job.
- Training workload seeds: `9001..9200`.
- Batch size: `4`.
- Validation every `5` rollout batches.
- Periodic checkpoint interval: `1,500` environment steps.

Two policy seeds are insufficient for a paper-level claim. This reduced grid is
only a fast mechanism check.

## Validation-only checkpoint selection

Checkpoint selection uses only `43001..43010`:

1. Keep candidates no more than `0.003` below the best delivery ratio.
2. Within that set, keep candidates no more than `0.010` below the best class-2
   delivery ratio.
3. Minimize the micro-aggregated avoidable-switch rate.
4. Break ties by total routing switches and the frozen deterministic metrics.

All validation candidates and the final selection record are retained. The test
panel is never consulted during checkpoint selection.

Each validation record stores both per-episode means and exact panel totals for
routing, avoidable, and forced switches. The avoidable-switch rate is audited as
`avoidable_switches_total / max(1, switch_opportunities)` before checkpoint
promotion. The frozen selection specification is embedded in every candidate,
selected, and final checkpoint as well as the training metrics and run manifest.

## Fresh diagnostic test panel

The four arms are evaluated on `44001..44025`: frozen proposed, frozen raw
congestion-context, frozen post-hoc beta-`0.20`, and newly trained schema-v3
MAPPO. This produces `16` shards and `400` episode rows. The fresh validation
and test panels are disjoint from training and all previously exposed panels,
including v2 panels `41001..41010` and `42001..42025`.

Primary diagnostics compare schema-v3 MAPPO with both proposed and raw-context
policies. The raw-context contrast tests whether the regularizer removes the
extra switching introduced by congestion context without discarding its useful
delivery behavior.

## Statistics and reporting limit

Crossed policy-seed by workload-seed summaries and bootstrap intervals are
descriptive only. Exact sign-flip p-values are raw; no Holm or Benjamini-Hochberg
adjustment is applied or claimed. With two policy seeds, inferential resolution
is too weak for a paper-level conclusion.

Every paired row is labeled `diagnostic_switch_regularized_preflight_v3`, has
`inferential_status=raw_exploratory_only` and `confirmatory=false`, and exposes
only `raw_p_value`. The decision is always
`diagnostic_only_no_promotion_decision`. A formal follow-up requires more policy
seeds, a new frozen test panel, and a separate confirmatory protocol.
