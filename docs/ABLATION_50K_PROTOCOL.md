# Frozen Protocol: 50k-Step MAPPO Ablation Study

**Protocol ID:** `ABLATION-50K-v2.2`
**Status:** frozen before any confirmatory test outcome is inspected
**Freeze date:** 2026-08-31
**Scope:** method ablation only; runtime estimates in Appendix A are not scientific results

This document is the analysis and execution contract for the full-budget
ablation. It supersedes the earlier draft with eight policy seeds. The method,
contrasts, endpoints, seed sets, checkpoint rule, exclusions, and inferential
procedures below must not be changed in response to test-set results.

## 1. Machine-auditable constants

```json
{
  "protocol_id": "ABLATION-50K-v2.2",
  "status": "FROZEN_PRE_CONFIRMATORY",
  "method_reference": "proposed",
  "method_reference_level": "L0",
  "environment_config_count": 8,
  "trainer_package_count": 1,
  "config_count": 9,
  "configs": [
    "proposed",
    "no_queue",
    "no_credit",
    "no_packet_context",
    "flat_critic",
    "no_ppo_protection",
    "with_lifetime_feature",
    "with_lifetime_reward",
    "with_hard_lifetime_mask"
  ],
  "scenarios": [
    "low_load",
    "medium_load",
    "hotspot_high_load",
    "frequent_break",
    "fault_links"
  ],
  "policy_seeds": [2009387241, 688652842, 1824192069, 1446495455, 140273932, 1016147766, 885581732, 247192783, 1367579272, 1996870291, 646895287, 489261598],
  "train_workload_seeds": {"start": 9001, "end": 9200, "count": 200},
  "validation_workload_seeds": {"start": 10001, "end": 10050, "count": 50},
  "test_workload_seeds": {"start": 31001, "end": 31050, "count": 50},
  "total_timesteps_argument": 50000,
  "validation_episodes_per_selection": 50,
  "test_episodes_per_job": 50,
  "planned_training_jobs": 540,
  "planned_test_episode_rows": 27000,
  "planned_contrasts_per_scenario": 8,
  "planned_contrasts": [
    "remove_queue_mechanism_package",
    "remove_centered_local_credit",
    "remove_packet_context",
    "replace_graph_critic_with_flat_critic",
    "remove_ppo_protection_package",
    "add_lifetime_feature",
    "add_lifetime_reward",
    "add_hard_lifetime_mask"
  ],
  "primary_confirmatory_tests": 40,
  "primary_endpoint": "delivery_ratio",
  "secondary_endpoints": [
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "routing_switches",
    "global_control_overhead_ratio"
  ],
  "alpha": 0.05,
  "confidence_level": 0.95,
  "crossed_bootstrap_resamples": 5000,
  "bootstrap_rng_seed_base": 18000,
  "historical_checkpoint_reuse_count": 0,
  "automatic_retries_after_initial_attempt": 1,
  "hash_algorithm": "SHA-256"
}
```

The runner must reject any mismatch between these constants, its matrix spec,
and the values actually passed to the trainer or evaluator.

## 2. Frozen method configurations

`proposed` is **L0**, the adopted method. Dynamic link failures remain part of
the physical environment, but predictive remaining-lifetime information is not
used as an actor/critic feature, reward term, or hard feasibility mask. Physical
queue capacity and physical link failure behavior remain enforced in every
configuration.

The eight environment configurations are defined by the following Boolean
flags. `1` means enabled and `0` means disabled.

| Configuration | Lifetime feature | Lifetime reward | Hard lifetime mask | Queue features | Queue reward | Centered local credit | Packet context | Graph critic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `proposed` (L0) | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| `with_lifetime_feature` (L1) | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| `with_lifetime_reward` (L2) | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| `with_hard_lifetime_mask` (L3) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `no_queue` | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| `no_credit` | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |
| `no_packet_context` | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| `flat_critic` | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |

`no_queue` removes the learned queue observations and queue-shaped reward
together; it does not remove the physical queue-capacity constraint. It is
therefore a queue-mechanism package, not a claim about either subcomponent in
isolation.

The ninth configuration, `no_ppo_protection`, uses the exact `proposed` L0
environment and changes only this trainer package:

| Trainer setting | `proposed` | `no_ppo_protection` |
|---|---:|---:|
| Actor gradient-norm clipping | `1.0` | disabled (`0`) |
| Target-KL epoch stopping | `0.02` | disabled (`0`) |
| Advantage normalization | enabled | disabled |

Critic gradient clipping, return normalization, and all other trainer settings
remain unchanged. These three removals form one **compound package**. This
comparison is not one-factor-at-a-time (OFAT) and may support only a statement
about their joint effect; it cannot identify any one protection's contribution.

Legacy labels are not reportable method names. In particular, historical
`no_lifetime` maps to `proposed`, and historical `full` maps to
`with_hard_lifetime_mask`.

## 3. Predeclared contrasts

Every contrast is `treatment - reference`, evaluated separately in each of the
five scenarios. The lifetime contrasts are adjacent steps of the cumulative
L0-to-L3 ladder; no non-adjacent lifetime comparison is confirmatory.

| Contrast ID | Family | Reference | Treatment | Frozen interpretation |
|---|---|---|---|---|
| `remove_queue_mechanism_package` | component | `proposed` | `no_queue` | Remove queue features and queue reward jointly |
| `remove_centered_local_credit` | component | `proposed` | `no_credit` | Remove centered local credit only |
| `remove_packet_context` | component | `proposed` | `no_packet_context` | Remove packet context only |
| `replace_graph_critic_with_flat_critic` | component | `proposed` | `flat_critic` | Replace graph critic only |
| `remove_ppo_protection_package` | trainer package | `proposed` | `no_ppo_protection` | Jointly remove the three specified PPO protections |
| `add_lifetime_feature` | adjacent lifetime | `proposed` (L0) | `with_lifetime_feature` (L1) | Add lifetime feature only |
| `add_lifetime_reward` | adjacent lifetime | `with_lifetime_feature` (L1) | `with_lifetime_reward` (L2) | Add lifetime reward only |
| `add_hard_lifetime_mask` | adjacent lifetime | `with_lifetime_reward` (L2) | `with_hard_lifetime_mask` (L3) | Add hard lifetime mask only |

A negative removal contrast supports the utility of the removed component or
package; lifetime contrast signs are interpreted directly. Any other contrast
must be labelled post hoc/exploratory and cannot replace a planned contrast.

## 4. Design, budgets, and seed separation

The confirmatory grid is
`9 configurations x 5 scenarios x 12 policy seeds = 540` independently trained
jobs. Each job is evaluated on the same 50 held-out workloads, producing
`540 x 50 = 27,000` test rows.

The fixed policy seeds are:

`2009387241, 688652842, 1824192069, 1446495455, 140273932, 1016147766,`
`885581732, 247192783, 1367579272, 1996870291, 646895287, 489261598`.

All 12 are new algorithm-replication identities: none appears in any historical
result CSV available at protocol freeze time. They are the first 12 distinct,
nonzero values obtained in counter order from
`SHA256("ABLATION-50K-v2.2-policy-seed-" + decimal_counter)`, taking the first
four digest bytes as an unsigned big-endian integer and clearing the high bit.
Candidates found in the identity-only historical seed inventory would be
rejected. No historical or confirmatory performance value was read or used to
generate or select this panel. This identity-only rule and the sample size were
fixed before confirmatory training. With eight seeds, a two-sided exact sign-flip test has
minimum attainable nonzero `p = 2 / 2^8 = 0.0078125`, which cannot cross the
first Holm threshold `0.05 / 40 = 0.00125`. With 12 seeds the minimum is
`2 / 2^12 = 0.00048828125`, providing basic resolution without guaranteeing
significance.

The workload partitions are fixed and pairwise disjoint:

| Phase | Workload seeds | Permitted use |
|---|---|---|
| Training | `9001..9200` (200) | Cycled during optimization only |
| Validation | `10001..10050` (50) | Checkpoint selection only |
| Test | `31001..31050` (50) | Final locked evaluation only |

The test panel was selected using an identity-only scan of all historical CSV
artifacts, without reading outcome values. The previously proposed
`13001..13050` panel was rejected before any confirmatory training because it
already appears in the legacy 5k ablation. No seed in `31001..31050` appears in
any historical experiment artifact available at protocol freeze time.

No test workload may be used for training, hyperparameter choice, checkpoint
selection, debugging, failure triage, or method revision. All configurations
within a scenario use identical policy-seed and workload-seed panels (common
random numbers). Policy seeds initialize the trainer's Python, NumPy, and
PyTorch random streams; workload seeds control episode exogenous randomness.

Every job receives `total_timesteps=50000`, `batch_size=4`, three PPO epochs,
four minibatches, validation every 40 rollout updates, and periodic saves every
5,000 environment steps. Training is never stopped early for performance. The
trainer completes the rollout batch in which the counter first reaches or
exceeds 50,000; the exact terminal counter must be recorded, and this identical
rule applies to all configurations.

## 5. Checkpoint selection and test lock

At each scheduled validation, the current policy is evaluated on all 50 fixed
validation workloads. The selected checkpoint maximizes the following
lexicographic tuple:

```text
(mean delivery_ratio, mean episode_reward, -mean drop_rate, -mean average_delay_slots)
```

Exact ties retain the earliest checkpoint. Selection uses no test observation.
The run still trains through the full budget after an earlier checkpoint becomes
best. A job without a valid `validation_best` checkpoint is failed.

Execution is two-phase:

1. Complete training and validation selection for all 540 jobs, then freeze a
   checkpoint manifest containing all selected checkpoint hashes.
2. Only after that manifest is complete and immutable, run the 50-workload test
   evaluation for every job and reveal aggregate test results.

Operators may inspect logs required to diagnose crashes, but may not inspect or
summarize test performance while training or method decisions remain open.

## 6. Endpoint and estimand

The sole primary endpoint is per-workload
`delivery_ratio = delivered_packets / generated_packets`. For each planned
contrast and scenario, let
`D[i,j] = Y_treatment[i,j] - Y_reference[i,j]`, paired on policy seed `i` and
test workload seed `j`. The primary estimand is the equally weighted mean of
the complete `12 x 50` difference matrix. Report raw delivery ratios and
differences; percentage-point differences are `100 x D`.

Secondary endpoints are `drop_rate`, `throughput_packets_per_slot`,
`average_delay_slots`, `p95_delay_slots`, `mean_queue_packets`,
`routing_switches`, and `global_control_overhead_ratio`. They are explicitly
secondary and cannot be promoted after test inspection.

## 7. Frozen statistical analysis

The independent algorithmic replication unit is the trained policy seed, not
one of the 600 crossed episode cells. Test workloads are paired repeated
measurements shared across policy seeds and configurations.

For every scenario-contrast-endpoint matrix:

1. Report treatment mean, reference mean, paired mean difference, 12
   policy-seed means, standard deviation across those means, positive-seed
   fraction, paired Cohen's `dz`, Hedges' `gz`, and matched-pairs rank-biserial
   correlation.
2. Form a 95% percentile crossed (pigeonhole) bootstrap interval using 5,000
   replicates. In each replicate, independently resample 12 whole policy-seed
   rows and 50 whole workload-seed columns with replacement, then average the
   indexed crossed matrix. Bootstrap RNG seeds are deterministic functions of
   scenario, contrast, and endpoint and are recorded in the analysis manifest.
   Specifically, use `18000 + 1000*s + 100*c + m`, where `s`, `c`, and `m`
   are zero-based indices in the scenario order in Section 1, contrast order in
   Section 3, and endpoint order (primary first, then the secondary order in
   Section 6), respectively.
3. Average `D[i,j]` over the 50 workloads to obtain 12 policy-seed differences
   `d[i]`. The confirmatory raw p-value is a two-sided exact sign-flip test of
   `abs(mean(d))`, enumerating all `2^12 = 4096` sign assignments and including
   ties in the tail.
4. Report a two-sided Wilcoxon signed-rank test on the same 12 `d[i]` values as
   a sensitivity analysis. It does not replace, rescue, or override the exact
   sign-flip result. Zeros/ties and the implementation method are recorded.

Missing cells, duplicate cells, non-finite values, or unmatched seed panels are
errors; they are never treated as independent observations or silently dropped.

### Multiplicity families

- **Primary confirmatory family:** the 8 planned contrasts x 5 scenarios = 40
  `delivery_ratio` exact sign-flip p-values. Apply two-sided Holm FWER control at
  `alpha=0.05`. Confirmatory significance requires the Holm-adjusted value to be
  at most 0.05; the confidence interval and effect magnitude must also be shown.
- **Within-endpoint sensitivity families:** for each endpoint separately,
  report Holm and Benjamini-Hochberg (BH) adjusted exact-test p-values across its
  40 tests. Secondary endpoint families are exploratory.
- **Global sensitivity family:** across all 8 endpoints x 40 tests = 320 tests,
  additionally report global Holm-FWER and BH-FDR adjustments. Global BH uses
  `q=0.05` and is exploratory; it cannot create a primary claim.
- Wilcoxon sensitivity p-values are reported alongside the exact-test values
  and are not used to declare the primary confirmatory result.

All raw and adjusted p-values are reported; `p<...` substitutions are forbidden.

## 8. Failures, missingness, reuse, and stopping rules

Historical checkpoint reuse is exactly **zero**. This includes prior 5k
ablations and prior 50k `no_lifetime`/`proposed` runs. Every one of the 540
confirmatory training jobs starts under this frozen protocol in the new matrix
directory. A completed job from this same protocol may be retained after an
orchestration restart only when its full code/config fingerprint and checkpoint
SHA-256 validate; this is resumption, not historical reuse.

Each failed training or evaluation phase receives at most one automatic retry
(two total attempts). A retry uses the identical scenario, configuration,
policy seed, code fingerprint, and budget. Seeds are never replaced. A failure
after the retry is recorded with the exception, timestamps, logs, and partial
artifact hashes. No outcome is imputed.

There is no efficacy, futility, variance, or significance stopping. The run
stops successfully only when the audit verifies all 540 training phases, all
540 evaluation phases, and exactly 27,000 unique finite test rows. An incomplete
grid may be reported as an operational failure, but no complete-case
confirmatory analysis or headline ablation claim is permitted.

A correctness, leakage, or reproducibility defect triggers a global pause. Any
code/config correction creates a new fingerprint and protocol amendment; all
affected cells are rerun symmetrically before results are inspected again.
Selective reruns based on favorable or unfavorable outcomes are forbidden.

## 9. Provenance and artifact audit

Before the first job, write an immutable matrix spec and record SHA-256 hashes
for this protocol and all executable sources: matrix runner, training entry,
environment, wrapper, variant definitions, model/design code, evaluation code,
statistics code, and the external CleanMARL trainer. Record the Git commit and
dirty status, full command line, Python executable/version, package versions,
PyTorch/CUDA versions, device identity, hostname, and relevant environment
variables.

Each job manifest must contain its stable job ID, scenario, configuration,
policy seed, workload partitions, trainer overrides, full config, code/config
fingerprint, UTC start/end times, attempts, exit status, actual environment
steps, selected-validation score and step, checkpoint path/size/SHA-256, and
log/metric artifact hashes. Each evaluation record additionally binds the
checkpoint hash to the exact 50 test seeds, 50-row shard hash, and evaluator
fingerprint.

Status and manifest writes are atomic. On resume, a missing file, hash mismatch,
fingerprint mismatch, malformed shard, wrong seed set, duplicate row, or stale
configuration invalidates that phase. The final audit records every included
job and shard, the merged CSV SHA-256, expected/observed counts, and a single
`matrix_complete` Boolean. Statistical outputs and figure inputs receive their
own hashes and analysis-manifest references.

## 10. Test-result immutability rule

After any test performance is revealed, the method definition, configuration
set, seeds, endpoints, contrasts, multiplicity families, and inferential methods
are immutable for this study. Test results may motivate clearly labelled future
work only. They may not be used to alter this matrix and rerun it as though the
revision had been predeclared.

## Protocol amendment A1: JSON round-trip resume fix

On 2026-08-31, after the four fixed calibration cells completed but before any
confirmatory test evaluation, an `--audit-only` resume invocation exposed an
operational defect: the in-memory matrix specification represented each
`changed_flags` sequence as a tuple, while the persisted JSON represented it as
a list. The semantic content and SHA-256 were identical, but direct Python
equality rejected the resumed invocation as a different matrix.

No validation or test performance value was inspected while identifying this
defect; only phase status, duration, exit state, and GPU telemetry were viewed.
The runner was amended to normalize the complete specification to JSON-native
types before hashing, writing, or comparing it, and a write/read/resume
regression test was added. The runner and protocol-document hashes were also
added to both phase fingerprints. Finally, each training invocation now writes
under a directory named by its complete training fingerprint: retries and
crash recovery for the same fingerprint share that directory, while a changed
fingerprint cannot discover or reuse an older checkpoint.

This amendment changes the executable-code fingerprint. Consequently, all four
calibration cells completed under the pre-amendment fingerprint, plus a
subsequent zero-duration checkpoint-discovery attempt that exposed the missing
path isolation, are invalidated and must be rerun symmetrically under the final
amended fingerprint. No method, configuration, seed, endpoint, contrast,
budget, multiplicity family, checkpoint rule, or inferential procedure changed.

## Appendix A. Non-result runtime estimate

This appendix is for resource scheduling only and must not appear as empirical
evidence for the method. The pre-calibration estimate is approximately **109
GPU-job-hours** for 540 training jobs (about 12.1 minutes/job on average).
With two concurrent jobs, the arithmetic lower bound is 54.5 wall-clock hours;
the operational planning interval is **60-75 hours**, with **84 hours reserved**
for retry, validation, evaluation, merge, and audit overhead.

Before the remaining launch, timing is recalibrated on these four fixed matrix
jobs under the final frozen code and hardware:

1. `low_load/proposed/seed_2009387241`
2. `hotspot_high_load/flat_critic/seed_688652842`
3. `frequent_break/no_ppo_protection/seed_1824192069`
4. `fault_links/with_hard_lifetime_mask/seed_1446495455`

During calibration, only wall-clock duration, GPU/resource telemetry, and
failure status may be viewed; performance metrics are not inspected or
aggregated. Successful calibration artifacts count toward the formal 540-cell
matrix because they use the identical frozen protocol. The updated ETA must
state the four observed durations, aggregate GPU-job-hours, achieved two-job
throughput, and its extrapolation rule. Updating the scheduling estimate does
not permit a design, stopping-rule, or analysis change.
