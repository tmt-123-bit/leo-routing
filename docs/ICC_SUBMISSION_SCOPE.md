# ICC Submission Scope and Claim Contract

## Decision

The ICC paper is scoped as:

> **Distributed Candidate-Set MAPPO for Reducing Cached Next-Hop Churn in Dynamic LEO Networks**

Here, *cached-next-hop churn* is the network-wide count of accepted changes to
the per-satellite cached next hop for the same
`(satellite, destination, traffic class)` tuple per episode. It is an
unnormalized episode total, not a per-satellite average or an end-to-end path
recomputation count. The term does not imply Lyapunov or queue stability,
throughput optimality, control-plane convergence, or a convergence guarantee.

This scope uses the completed corrected evaluation of validation-best
checkpoints from 50k-step training runs. It does not require a new model or new
training to draft the paper. It also does not turn the retrospective evidence
into a preregistered confirmatory study.

## Ready submission assets

- Seed-aware two-panel figure: `figures/icc_delivery_stability.pdf` and
  `figures/icc_delivery_stability.png`.
- Exact-value LaTeX table: `figures/icc_main_results.tex`.
- Frozen-source generator: `src/make_icc_submission_assets.py`.
- Source and rendering contract tests:
  `src/test_make_icc_submission_assets.py`.

The generator rejects changed comparison or statistical-manifest hashes and
requires the source manifest to remain retrospective (`confirmatory=false`).
It also binds the EXP-004 experiment manifest and a canonical provenance hash
over all 40 `validation_best.pt` checkpoints and their run manifests.

## Three contributions

1. A shared, permutation-equivariant candidate-set actor scores fixed candidate
   slots, uses feasible candidates for symmetric aggregation, and masks
   infeasible actions before categorical selection. This supports changing
   local connectivity during decentralized execution.
2. A combined CTDE system design uses a topology-aware graph critic, a team
   objective, and centered local credit during training. Credit-active agents
   are satellites holding a head-of-line packet, including no-route cases; the
   centered deviation sums to zero over that set. The adopted `proposed`
   variant does not use a predictive remaining-link-lifetime signal, lifetime
   reward, or a lifetime hard mask.
3. A seed-level paired evaluation characterizes the delivery-versus-cached-
   next-hop-churn trade-off across five dynamic scenarios. It reports both the
   consistent churn reduction and the hotspot delivery failure.

MAPPO itself is not claimed as novel. The paper contribution is the auditable
candidate-set LEO routing design and its bounded empirical claim.
Because the complete 50k-step component ablation is unavailable, the paper
does not attribute the measured effect to the actor, critic, or credit term in
isolation and does not claim that any one component is necessary.

## Required related-work positioning

The six-page paper must reserve a compact related-work subsection. It should
position four groups with primary citations: classical snapshot/low-churn LEO
routing (including Werner-style virtual topology and SHORT), Q-routing and
learning-based LEO routing, permutation-equivariant set policies plus invalid
action masking, and graph-based CTDE/MAPPO. The five implemented baselines must
either receive their original citation or be labeled as internal heuristics.
Unreproduced published methods may motivate the gap but must not be described
as direct empirical competitors.

## Method boundary

Included in the paper method:

- a 24-satellite synthetic multi-plane LEO slot simulator with
  Walker-Delta-style indexing and connectivity;
- decentralized next-hop decisions for each head-of-line packet;
- a fixed 26-dimensional feature vector for each candidate, with queue, link,
  geometric progress, packet-context, traffic-class, and route-change context; its
  lifetime coordinate is retained for checkpoint compatibility but is
  identically zero in the adopted `proposed/no_lifetime` policy;
- a shared candidate scorer and feasibility action mask;
- a graph critic during centralized training;
- team reward plus centered local credit;
- physical topology changes and link failures executed by the environment.

Excluded from the paper method and contribution:

- neighbor-state Age of Information (AoI);
- a communication-budget constraint, learned messaging, or a Lagrangian budget;
- predictive link-lifetime features, reward, or hard masks;
- the congestion-context experimental candidates;
- all post-hoc hysteresis/shield variants;
- theoretical queue stability or throughput-optimality claims.

The packet waiting-time feature must not be relabeled as AoI. Physical link
breaks in the environment must not be relabeled as a lifetime-aware policy.
The simulator's soft control-overhead cost is an auxiliary reward term, not an
enforced communication budget or a learned state-exchange policy.

At inference, *decentralized local observation* means the satellite's own
head-of-line packet and cache state together with incident-link telemetry and
one-hop-neighbor queue/contention telemetry. The simulator exposes the freshest
slot state synchronously. It does not model how that telemetry is signaled,
its refresh delay, staleness, or communication cost. Decentralized execution
therefore means no training-time graph critic or centralized route computation
is required at inference; it does not mean zero inter-satellite state exchange.

## Frozen headline evidence

Source data:

- `experiments/eval-main/episode_metrics.csv` (`5,250` rows);
- `experiments/eval-main/experiment_manifest.json`;
- `experiments/legacy-reanalysis/eval-main/paired_tests.csv`;
- `experiments/legacy-reanalysis/eval-main/statistical_analysis_manifest.json`.

The design contains five scenarios, eight independently trained MAPPO policy
seeds per scenario, and 50 common held-out workloads per policy seed. This is
40 MAPPO runs and 400 paired policy-seed/workload cells per scenario. Each
MAPPO training run had a 50k-environment-step target; the run manifests record
50,040 steps because updates finish on a rollout boundary. Evaluation used its
validation-best checkpoint, selected on 50 validation workloads disjoint from
the 50 held-out test workloads. The legacy lexicographic validation rule first
maximized delivery ratio, then mean reward, then preferred lower drop rate and
lower average delay. Q-routing used 500 training episodes and is paired by
policy seed and workload.

Among the five implemented baselines (`delay_only`, `full_heuristic`, global
Dijkstra, OSPF-ECMP, and Q-routing), Q-routing has the highest held-out mean
delivery in every scenario in this artifact. It was identified
retrospectively. The table therefore reports the full five-scenario comparison
rather than selecting only favorable scenarios.

Delivery differences are MAPPO minus Q-routing in percentage points. Intervals
are 5,000-resample crossed pigeonhole-bootstrap 95% intervals.

| Scenario | MAPPO | Q-routing | Delivery difference, pp [95% CI] | MAPPO-higher seeds | Raw exact p | Retrospective five-scenario Holm p |
|---|---:|---:|---:|---:|---:|---:|
| `low_load` | 0.912803 | 0.912197 | +0.060606 [-0.143939, +0.280398] | 6/8 | 0.1875 | 0.3750 |
| `medium_load` | 0.787982 | 0.784557 | +0.342448 [+0.042936, +0.666667] | 8/8 | 0.0078125 | 0.0390625 |
| `hotspot_high_load` | 0.289930 | 0.308555 | -1.862500 [-2.104025, -1.641988] | 0/8 | 0.0078125 | 0.0390625 |
| `frequent_break` | 0.758568 | 0.752096 | +0.647135 [+0.283854, +1.040397] | 8/8 | 0.0078125 | 0.0390625 |
| `fault_links` | 0.763359 | 0.762135 | +0.122396 [-0.305990, +0.566439] | 4/8 | 0.4921875 | 0.4921875 |

Next-hop-switch differences are MAPPO minus Q-routing in the network-wide
count of accepted per-satellite cached-next-hop changes per episode. The cache
key is `(satellite, destination, traffic class)`.

| Scenario | MAPPO | Q-routing | Switch difference [95% CI] | Relative reduction | Direction | Raw exact p | Retrospective five-scenario Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `low_load` | 0.9325 | 2.2200 | -1.2875 [-1.7800, -0.8000] | 58.00% | 8/8 fewer | 0.0078125 | 0.0390625 |
| `medium_load` | 31.0325 | 59.0875 | -28.0550 [-33.3728, -21.1774] | 47.48% | 8/8 fewer | 0.0078125 | 0.0390625 |
| `hotspot_high_load` | 138.8925 | 186.3325 | -47.4400 [-66.9434, -28.2348] | 25.46% | 8/8 fewer | 0.0078125 | 0.0390625 |
| `frequent_break` | 54.7875 | 76.9925 | -22.2050 [-27.9927, -16.6049] | 28.84% | 8/8 fewer | 0.0078125 | 0.0390625 |
| `fault_links` | 44.5725 | 65.8950 | -21.3225 [-27.2728, -15.2293] | 32.36% | 8/8 fewer | 0.0078125 | 0.0390625 |

The defensible interpretation is:

- small delivery gains in `medium_load` and `frequent_break`;
- no clear delivery difference in `low_load` or `fault_links`;
- a consistent delivery loss in `hotspot_high_load`;
- fewer route switches in every evaluated scenario.

## Statistical contract

- The independent algorithm replication unit is the trained policy seed
  (`n=8`), not the 50 workloads or the 400 episode rows.
- For each seed, compute the paired MAPPO-minus-Q-routing mean over the 50
  common workloads, then apply a two-sided exact sign-flip test to the eight
  seed means. There are `2^8=256` sign assignments, so the smallest attainable
  two-sided p-value is `0.0078125`.
- Use the 5,000-resample crossed pigeonhole bootstrap, resampling both policy
  seeds and workload seeds, for 95% intervals.
- Q-routing was retrospectively identified as the highest-held-out-mean-
  delivery method among the five implemented baselines in each scenario.
  Holm adjustment over the five scenario comparisons is a retrospective
  sensitivity analysis, not a preregistered confirmatory decision.
- `eval-main` re-evaluates the same 40 `train-main/no_lifetime` checkpoints with
  the corrected wrapper/evaluator. It is not an independent retraining
  replication.
- The headline environment is a synthetic 24-satellite slot simulator with
  simplified orbital-geodetic features and time-varying multi-plane links; it
  is not a full TLE/SGP4 or Hypatia orbital propagation experiment.
- Do not quote the legacy episode-level p-values that treated repeated
  workloads as independent observations.

## Six-page allocation

| Page | Content | Planned artifact |
|---:|---|---|
| 1 | Abstract; motivation; operational definition of cached-next-hop churn; three contributions | No large figure |
| 2 | Compact related work; dynamic graph, queues, candidate action set, objective | Fig. 1: system and CTDE pipeline |
| 3 | Candidate-set actor, action mask, graph critic, centered credit, compact Algorithm 1 | Method equations and half-column algorithm |
| 4 | Five scenarios, baselines, 8-by-50 paired design, training budget, seed-level statistics | Table I: setup/statistical contract |
| 5 | Full delivery and route-switch results, including hotspot failure | Fig. 2: two-panel effects; Table II: exact values |
| 6 | Discussion, limitations, conclusion, references | No new empirical claim |

The final page rule must be checked against the target year's ICC call for
papers. The outline assumes a six-page hard budget including references.

## Figure and table mapping

| Artifact | Required content | Source | Claim supported |
|---|---|---|---|
| Fig. 1 | Dynamic LEO graph to local candidate sets; shared actor; masked next-hop action; training-only graph critic | `src/leo_multiagent_env.py`, `src/mappo_design.py` | Architecture and decentralized execution |
| Fig. 2(a) | Five delivery differences versus Q-routing with crossed 95% intervals; hotspot highlighted as a loss | seed-aware `paired_tests.csv` | Scenario-dependent delivery effect |
| Fig. 2(b) | Five network-wide counts of per-satellite next-hop-switch differences with crossed 95% intervals | seed-aware `paired_tests.csv` | Cross-scenario cached-next-hop-churn effect |
| Table I | Environment, scenarios, policy/workload seeds, training budgets, baselines, inference unit, tests | experiment and statistics manifests | Reproducibility |
| Table II | Exact delivery and switching values, intervals, seed directions, and retrospective Holm values | frozen tables above | Complete headline evidence |

Do not reuse a plot if it was generated from the superseded episode-level
statistics. Fig. 2 and Table II must be regenerated from
`experiments/legacy-reanalysis/eval-main`.

## Hotspot and shield boundary

The hotspot result is a failure case, not an ablation to hide: MAPPO loses
`1.862500` delivery percentage points to Q-routing with `0/8` seeds favoring
MAPPO, even though it reduces switches by `47.44` per episode (`25.46%`). The
paper should describe this as a delivery-churn trade-off under concentrated
high load.

The v7-v11 shield work is not a contribution or a successful extension. V11
ended with `selection_status=no_eligible_candidate`, passed only `5/8` exposed
design cells, and did not open fresh validation or test workloads. At most one
sentence may appear in limitations stating that an exploratory deployment-time
shield failed its frozen all-cell promotion gate. Shield results must not enter
the title, abstract, contribution list, Fig. 2, or the main performance table.

## Claim matrix

Allowed:

- "MAPPO reduced route switches relative to Q-routing in all five evaluated
  scenarios."
- "Delivery improvements were scenario-dependent and were observed in medium
  load and frequent link breaks."
- "The hotspot experiment revealed a consistent delivery-churn trade-off."
- "The results are a retrospective seed-level reanalysis of validation-best
  checkpoints from completed 50k-step training runs."

Not allowed:

- "MAPPO outperformed every baseline in every scenario."
- "The method is robust under hotspot congestion."
- "The method is AoI-aware, communication-budget-aware, or lifetime-aware."
- "The shield solves route instability."
- "The incomplete 50k ablation proves component necessity."
- "These results establish top-journal quality or guarantee ICC acceptance."

## Evidence not promoted into the ICC core

- The 5k legacy ablation is exploratory only.
- The 9-by-5-by-12 50k ablation matrix is incomplete and cannot support a
  component-necessity claim.
- The archived `repro-check` has insufficient final-variant provenance and an
  older evaluator; it is only a version-sensitivity check.
- Scale, TLE, sweep, and ns-3 artifacts require their own provenance and
  seed-level audit before being added. They are not needed for the bounded ICC
  claim above.
- The historical ns-3 closed-loop summaries predate the current bridge and
  episode-isolation corrections. They must not be copied into the ICC paper;
  only a fresh, separately labeled rerun may be considered after audit.
