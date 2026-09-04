# Avoidable-Switch Classical Baselines Formal Protocol v1

## Purpose and timing

This protocol freezes the classical comparison layer before any workload in
the formal train, checkpoint-selection, independent-gate, or sealed-test panel
is instantiated. It complements, but does not modify,
`AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1.md`.

This document does not authorize the sealed test panel. Q-routing training and
all baseline artifacts must be complete and hashed before a later one-time
test authorization can be considered. No result from MAPPO formal validation
may change the definitions below.

## Scenarios, environment, and seeds

The baseline layer is limited to the same two synthetic scenarios:

```text
medium_load
hotspot_high_load
```

All classical methods use the canonical `qos_only` wrapper. For these methods,
the switch-reward flag does not alter physical dynamics; using `qos_only`
keeps reward reporting aligned with the primary MAPPO arms.

The eight stochastic policy identities are exactly the SHA-256-derived MAPPO
formal policy seeds:

```text
179055553
626669596
746965870
183895110
310818925
2136406109
985998595
515636025
```

Workload roles remain:

```text
training:             76001..76200
checkpoint selection: 77001..77010  (MAPPO only)
independent gate:      77011..77020
sealed test:           78001..78050  (metadata only; access forbidden here)
```

## Q-routing

Q-routing is retrained independently for every scenario and policy seed,
creating `2 x 8 = 16` models. Historical tables or Q arrays cannot be reused.

The frozen implementation is `mappo_evaluation.QRoutingPolicy` with:

```text
n_nodes = 24
alpha = 0.3
epsilon during training = 0.1
epsilon after freeze = 0.0
Q initialization = 10.0 in float32 for every node/destination/neighbor cell
training episodes = 500
performance early stopping = disabled
```

Training episode `e`, zero based, uses:

```text
workload_seed = 76001 + (e mod 200)
```

Thus all 200 train workloads are traversed twice in their fixed ascending
order and `76001..76100` are traversed once more. The policy seed initializes
the policy-local Python RNG used for epsilon exploration. A feasible action is
otherwise selected by the smallest tabular Q plus the frozen action-selection
candidate cost:

```text
selection candidate cost = normalized edge delay
                         + normalized neighbor queue occupancy
                         + current candidate-link load rho
```

These are respectively `candidate_features[2]`, `candidate_features[1]`, and
`candidate_features[4]` in the frozen implementation. Exact ties retain
deterministic candidate order. At every active decision with more than one
feasible candidate, exploration is an independent Bernoulli draw with
probability `0.1`; on an exploration draw, the policy chooses uniformly from
the feasible candidate list using its policy-local Python RNG.

After the environment step, every pre-contention pending
`(satellite, destination, neighbor)` proposal created by that decision branch
is updated. The update is not conditioned on whether contention ultimately
accepted the proposal. Its bootstrap minimum ranges over every neighbor
returned by the frozen base-topology `_neighbors(neighbor)` method and is not
filtered by the next decision's instantaneous feasibility mask:

```text
transition immediate = normalized edge delay
                     + normalized neighbor queue occupancy
target = transition immediate
       + min_next_neighbor Q[neighbor,destination,next_neighbor]
Q <- (1-alpha) Q + alpha target
```

The current-link load term is part of action selection but not the transition
update target. This asymmetry is an explicit property of the frozen
`mappo_evaluation.QRoutingPolicy`; the runner must bind and report both formulas
separately rather than silently adapting the policy.

There is no discount factor, replay buffer, target network, validation
selection, hyperparameter search, or continuation after episode 500. The
final Q array, dtype, shape, hyperparameters, policy RNG seed, train workload
order, source fingerprint, and model hash are persisted. Training may be
rerun after infrastructure failure only with identical inputs; replacement
seeds and performance-based retries are forbidden.

The independent gate panel may be used once to verify finite execution and to
produce descriptive rows. Q-routing performance is not a gate for the new
method and cannot tune either method.

## OSPF-ECMP

`mappo_evaluation.OspfEcmpPolicy` is evaluated without training. At each
decision it computes the minimum current incident-edge delay plus the current
shortest remaining delay to the destination. Actions within `1e-6` of the
minimum form the ECMP set.

Because ECMP tie selection is stochastic, it is evaluated once for every
scenario, workload, and each of the same eight policy seeds. The policy-local
Python RNG is initialized by that seed before each complete policy replicate;
it is not reset between workloads within the replicate. These eight ECMP
replicates are stochastic routing replicates, not independently trained models,
and must be labeled accordingly.

## Global Dijkstra

`mappo_evaluation.GlobalDijkstraPolicy` is evaluated without training. It is
deterministic for a fixed workload, so it is evaluated exactly once per
scenario and workload with sentinel policy seed `-1`. Its rows must never be
duplicated eight times or treated as eight independent algorithm replications.

## Metrics and future sealed comparison

Every evaluation uses the same structured episode evaluator as the formal
MAPPO study and records, at minimum:

```text
generated, delivered, dropped, backlog
delivery ratio, throughput, mean and p95 delay
mean and maximum queue occupancy
accepted total/avoidable/forced cached-next-hop switches
accepted switch opportunities
pre-contention decision avoidable cost/opportunity/rate
pre-contention forced decisions and forced constraint cost (=0)
```

Rates are reconstructed from integer counts. Accepted episode switch totals
and pre-contention decision-level rates are different estimands and remain in
separate columns.

The independent-gate summary is descriptive. For Q-routing and OSPF-ECMP,
delivery is first averaged over the 10 workloads within each policy or routing
identity and then averaged with equal weight over the eight identities. The
decision avoidable-switch rate is first reconstructed within each identity as
`sum(cost)/sum(opportunity)` and is then averaged with equal identity weight.
Pooled count ratios are retained only as integrity diagnostics. If any identity
has zero total opportunity, its rate is recorded as undefined, the affected
identity is listed, and the equal-identity rate summary is also undefined.
Global Dijkstra remains a single deterministic workload-paired reference and
is never expanded into eight identities for this summary.

For a future sealed analysis, Q-routing comparisons are paired on identical
policy-seed and workload identities. OSPF-ECMP comparisons pair the eight
stochastic RNG replicates and workloads but must not call them trained seeds.
Global Dijkstra is a deterministic workload-paired reference and is not used
to inflate the independent seed count.

The primary future confirmatory family remains the constrained-minus-QoS-only
MAPPO endpoints defined by the formal method protocol. The following are a
separate classical-comparison family and cannot rescue a failed primary
claim:

```text
constrained MAPPO minus Q-routing delivery, per scenario
constrained MAPPO minus Q-routing decision avoidable rate, per scenario
```

The four Q-routing comparison p-values use policy-seed exact tests with Holm
FWER control. Crossed bootstrap intervals resample policy-seed and workload
axes. Comparisons to OSPF-ECMP and Global Dijkstra are secondary; all raw
methods and both scenarios are reported even when unfavorable.

## Integrity and claim limits

The classical runner must have no sealed-test execution path. Before a later
test authorization, it must freeze its source and dependency fingerprint, all
16 Q-routing arrays and manifests, its complete gate rows, and a zero-access
audit. The future authorization must bind these exact hashes together with the
48 MAPPO selected checkpoint hashes.

Each Q-routing training job and each independent-gate replicate permits at
most two recorded attempts, meaning one initial attempt and one infrastructure
retry with identical inputs. Attempt status, checkpoint/model state, RNG state,
row shards, and final manifests are hash-bound. A completed artifact is loaded
read-only and fully audited; it is not silently repaired on re-entry.

Retry eligibility is fail-closed and recorded in the attempt status. Automatic
retry and retry after process re-entry are allowed only for an orphaned
`running` attempt or for a recognized infrastructure exception: Python
`TimeoutError`, Python `ConnectionError`, or `OSError` carrying one of the
frozen errno names below.

```text
EAGAIN EBUSY ECONNABORTED ECONNRESET EHOSTUNREACH EINTR EIO
ENETDOWN ENETRESET ENETUNREACH EPIPE ESTALE ETIMEDOUT
```

Configuration, schema, source-fingerprint, hash, audit, scientific-computation,
and other runtime failures are recorded with `retry_authorized=false` and fail
immediately. A later invocation must not consume the second attempt unless the
prior status contains a recognized retryable category and
`retry_authorized=true`. Keyboard interruption and unbound/manual fail-fast
cancellation are not retryable. A fail-fast cancellation caused by another
parallel job is retryable only when the cancellation status is explicitly
bound to that job's recognized infrastructure exception; cancellation caused
by a configuration, audit, scientific, or other runtime failure remains
non-retryable.

Even if every later test succeeds, claims are limited to `medium_load` and
`hotspot_high_load` in the audited 24-satellite synthetic simulator. This
protocol does not support broad failure robustness, five-scenario
generalization, real-orbit validation, CPO guarantees, or comparison with
unimplemented published systems. Old ICC figures use a different switch
estimand and cannot serve as the new study's result table.
