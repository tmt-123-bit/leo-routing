# LEO MAPPO Routing - Completed Legacy Results

> Default evidence package: the completed 8-seed, 50k-step `eval-main` evaluation. No new training is required to reproduce this summary.
> Evidence status: retrospective reanalysis, not preregistered confirmatory evidence and not by itself a guarantee of IEEE top-journal acceptance.
> Inference unit: independently trained policy seed. The 50 common workload seeds are paired repeated measurements, not 50 independent policy runs.
> Historical labels are aliases only: `no_lifetime` means `proposed` (L0), while `full` means `with_hard_lifetime_mask` (L3).

## 1. Corrected completed 50k evaluation

Verified source: `experiments/eval-main/episode_metrics.csv` (5250 episode rows). The table below selects the highest-delivery baseline in each scenario; in this artifact it is Q-routing in all five scenarios.

| scenario | MAPPO | strongest baseline | gap, pp [crossed 95% CI] | MAPPO-higher seeds | exact p | retrospective Holm p |
|---|---:|---:|---:|---:|---:|---:|
| low_load | 0.9128 | q_routing 0.9122 | +0.061 [-0.144, +0.280] | 6/8 | 0.1875 | 0.375 |
| medium_load | 0.7880 | q_routing 0.7846 | +0.342 [+0.043, +0.667] | 8/8 | 0.0078125 | 0.0390625 |
| hotspot_high_load | 0.2899 | q_routing 0.3086 | -1.863 [-2.104, -1.642] | 0/8 | 0.0078125 | 0.0390625 |
| frequent_break | 0.7586 | q_routing 0.7521 | +0.647 [+0.284, +1.040] | 8/8 | 0.0078125 | 0.0390625 |
| fault_links | 0.7634 | q_routing 0.7621 | +0.122 [-0.306, +0.566] | 4/8 | 0.4921875 | 0.4921875 |

Interpretation: MAPPO has small delivery gains in medium_load, frequent_break, no clear delivery difference in low_load, fault_links, and a consistent across-seed loss in hotspot_high_load. These Holm values adjust the five retrospectively selected strongest-baseline comparisons; they are sensitivity values, not confirmatory decisions.

Against Q-routing, MAPPO reduces routing switches in all five scenarios by 1.29 to 47.44 switches per episode (25.5% to 58.0%). Every scenario is 8/8 seeds in the same direction; each exact raw p is 0.0078125 and each retrospective five-scenario Holm p is 0.0390625.

## 2. Provenance and archived reproduction check

The three experiment manifests verify that `eval-main` re-evaluates the same 40 completed `train-main/no_lifetime` checkpoints with the corrected wrapper/evaluator. It is a corrected evaluation, not an independent retraining replication.

The separately trained archived directory named `repro-check` is weaker evidence than its name suggests: its manifest does not record the method variant, its checkpoint paths do not identify `no_lifetime`, and it uses the legacy wrapper/evaluator. It is therefore shown only as a version-sensitivity check and cannot validate the final method configuration.

| scenario | corrected eval-main gap vs Dijkstra, pp | archived run gap vs Dijkstra, pp | archived exact p |
|---|---:|---:|---:|
| low_load | +0.22 | -0.45 | 0.0078125 |
| medium_load | +5.31 | +1.00 | 0.015625 |
| hotspot_high_load | -0.90 | -2.01 | 0.0078125 |
| frequent_break | +5.15 | +0.34 | 0.15625 |
| fault_links | +4.89 | +0.85 | 0.015625 |

The archived run is materially weaker in several scenarios and even changes direction in low load. This discrepancy must remain visible in any paper or response to reviewers.

## 3. Legacy 5k ablation (exploratory only)

The older `experiments/ablation/` matrix used only 5,000 training steps (10% of the headline budget). It is a legacy pilot and cannot support a formal component-necessity claim.

The most directly relevant old branch comparison is shown only to document why L0 (`no_lifetime`/`proposed`) was retained. Effects are L0 minus legacy L3 (`full`).

| scenario | exploratory delivery effect, pp [crossed 95% CI] | exact raw p |
|---|---:|---:|
| low_load | +0.78 [+0.50, +1.09] | 0.0078125 |
| medium_load | +8.86 [+6.21, +11.71] | 0.0078125 |
| hotspot_high_load | +2.69 [+1.86, +3.69] | 0.0078125 |
| frequent_break | +41.42 [+38.20, +44.48] | 0.0078125 |
| fault_links | +8.07 [+5.45, +11.58] | 0.0078125 |

## 4. Optional paused 50k ablation

The 9 configurations x 5 scenarios x 12 policy seeds study is not part of the default workflow. Its partial files are retained only so the study can be resumed later if explicitly requested.

**Status: incomplete; no confirmatory ablation estimate is reportable.**

| audited item | observed | required |
|---|---:|---:|
| completed training jobs | 4 | 540 |
| frozen checkpoint records | missing | 540 |
| completed evaluations | 0 | 540 |
| finite, unique test rows | 0 | 27000 |

Publication gates not yet satisfied:

- training_freeze_manifest.json is missing or invalid
- matrix audit does not set matrix_complete=true
- validated training count is 4, not 540
- validated evaluation count is 0, not 540
- validated test row count is 0, not 27000
- episode_metrics_manifest.json is missing or invalid
- statistical_analysis_manifest.json is missing or invalid
- paired_ablation_effects.csv is missing

## 5. Reporting boundary

The previous extremely small headline and fairness p-value statements came from treating repeated workload episodes as independent observations. They are pseudoreplicated, are not valid policy-seed-level evidence, and are not reproduced here. The corrected analysis first averages each paired contrast within policy seed, then applies a two-sided exact sign-flip test across the eight independently trained seeds. With eight seeds the smallest possible two-sided exact p-value is 0.0078125.

The completed data are usable for a manuscript if the claims match the table: small, scenario-dependent delivery effects; a consistent routing-stability advantage; an explicit hotspot failure case; no formal 50k component proof; and no claim that this package alone establishes top-journal-level evidence.

Reanalysis manifests and hashed outputs: `experiments/legacy-reanalysis/`.
