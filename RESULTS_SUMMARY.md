# LEO MAPPO Routing — Consolidated Results Summary (all no_lifetime)

> Auto-generated from experiment aggregates. MAPPO = no_lifetime variant.
> Paired Wilcoxon + Benjamini-Hochberg; throughput-ratio bootstrap CI (B=5000).

## 1. Headline (5 scenarios, 8 policy seeds x 20 workload seeds)

| scenario | MAPPO del | Dijkstra del | gap | MAPPO P95 | Dijkstra P95 |
|---|---|---|---|---|---|
| low_load | 0.904 | 0.907 | -0.3pp | 6.6 | 6.4 |
| medium_load | 0.774 | 0.726 | +4.8pp | 10.6 | 12.6 |
| hotspot_high_load | 0.284 | 0.295 | -1.1pp | 20.7 | 21.0 |
| frequent_break | 0.439 | 0.414 | +2.4pp | 15.9 | 16.9 |
| fault_links | 0.747 | 0.704 | +4.3pp | 11.1 | 13.1 |

MAPPO beats Dijkstra on delivery in medium/frequent_break/fault; lower P95 in 4/5.

## 2. Ablation (variants, full-variant ref; mean delivery over 5 scenarios)

| variant | delivery | vs full |
|---|---|---|
| full (ref) | 0.563 |  — |
| no_credit | 0.475 | -8.8pp |
| no_lifetime | 0.687 | +12.4pp |
| no_queue | 0.554 | -1.0pp |
| flat_critic | 0.574 | +1.1pp |
| no_packet_context | 0.568 | +0.5pp |
| no_ppo_protection | 0.541 | -2.2pp |

no_lifetime large positive (adopted as default); no_credit strongly negative (credit assignment essential to CTDE).

## 3. Scale transfer (zero-shot, n24-trained -> eval)

| scale | thru ratio MAPPO/Dij [95% CI] |
|---|---|
| n110 | 1.055 [1.043, 1.067] |
| n132 | 0.945 [0.934, 0.954] |
| n24 | 1.074 [1.058, 1.090] |
| n66 | 1.192 [1.177, 1.208] |
Beats oracle (ratio>1) at n24/n66/n110; graceful to n132 (0.945).

## 4. Real-topology transfer (zero-shot synthetic -> real TLE)

| topology | thru ratio [95% CI] | MAPPO P95 | Dij P95 |
|---|---|---|---|
| oneweb_24 | 0.935 [0.910, 0.960] | 17.6 | 21.8 |
| starlink_24 | 0.928 [0.909, 0.947] | 18.7 | 20.4 |
~93-94% oracle throughput zero-shot; lower P95 on both real topologies.

## 5. C3 real-topology TRAINING (train Starlink-24; in-dist + cross-constellation)

| eval topology | thru ratio [95% CI] | MAPPO P95 | Dij P95 |
|---|---|---|---|
| tle_oneweb_24 | 0.960 [0.945, 0.976] | 18.3 | 21.3 |
| tle_starlink_24 | 0.935 [0.920, 0.952] | 18.7 | 20.0 |
train-on-real (0.935) = zero-shot transfer (0.928) -> transfer suffices; cross-constellation OneWeb 0.960 with -3.0 slots P95.

## 6. Load sweep (the congestion-awareness win)

| load | MAPPO del | Dij del | gap | thru ratio [CI] | MAPPO P95 | Dij P95 |
|---|---|---|---|---|---|---|
| exo2 | 0.904 | 0.908 | -0.4pp | 0.995 [0.989,1.002] | 6.1 | 6.1 |
| exo4 | 0.876 | 0.836 | +3.9pp | 1.047 [1.027,1.066] | 7.4 | 9.3 |
| exo6 | 0.783 | 0.733 | +5.1pp | 1.069 [1.057,1.082] | 10.2 | 12.7 |
| exo8 | 0.655 | 0.595 | +6.0pp | 1.101 [1.086,1.116] | 13.1 | 15.9 |
| exo10 | 0.546 | 0.511 | +3.5pp | 1.068 [1.051,1.086] | 15.8 | 18.6 |
| exo14 | 0.397 | 0.379 | +1.8pp | 1.046 [1.031,1.064] | 18.8 | 19.8 |
| exo20 | 0.266 | 0.258 | +0.7pp | 1.028 [1.012,1.044] | 21.3 | 21.5 |
| exo28 | 0.221 | 0.215 | +0.6pp | 1.029 [1.016,1.042] | 22.2 | 22.4 |
MAPPO beats oracle on delivery exo4-exo14; thru ratio up to 1.10 at exo8.

## 7. Fault sweep (robustness to link failures)

| fault rate | MAPPO del | Dij del | gap | thru ratio [CI] |
|---|---|---|---|---|
| 0.00 | 0.790 | 0.728 | +6.2pp | 1.084 [1.072,1.099] |
| 0.04 | 0.784 | 0.726 | +5.8pp | 1.080 [1.066,1.095] |
| 0.08 | 0.762 | 0.711 | +5.1pp | 1.072 [1.058,1.086] |
| 0.12 | 0.681 | 0.616 | +6.5pp | 1.105 [1.085,1.127] |
| 0.16 | 0.655 | 0.590 | +6.5pp | 1.111 [1.085,1.137] |
| 0.20 | 0.598 | 0.532 | +6.5pp | 1.123 [1.085,1.162] |
MAPPO beats oracle +6pp at every fault rate; thru ratio 1.08->1.12 as failures intensify.

## 8. Fairness (mechanism)

MAPPO load imbalance < Dijkstra all 5 scenarios (paired Wilcoxon p=1.8e-15). Shortest-path hot-spot-prone; MAPPO trades a sliver of path-optimality for load spread.

## 9. Convergence / reproducibility

8-seed training CV 1-5% (easy/moderate), ~9% (hardest hotspot). See fig_convergence.

## 10. Reward-weight sensitivity (robustness)

| config | delivery | vs baseline | P95 | imbalance |
|---|---|---|---|---|
| baseline | 0.789 | +0.0pp | 10.33 | 0.707 |
| deliver_lo | 0.790 | +0.1pp | 10.18 | 0.700 |
| deliver_hi | 0.790 | +0.1pp | 10.36 | 0.697 |
| load_lo | 0.784 | -0.5pp | 10.60 | 0.709 |
| load_hi | 0.795 | +0.6pp | 10.01 | 0.691 |
| switch_lo | 0.796 | +0.7pp | 9.95 | 0.697 |
| switch_hi | 0.774 | -1.5pp | 10.73 | 0.711 |

All perturbations within +/-1.5pp of baseline; every config beats Dijkstra (+5-7pp). w_deliver inert (±0.1pp at 0.5x/2x); w_load mildly helpful (w_load=2.0 -> +0.6pp, lower P95 + imbalance); w_switch is the only weight that moves delivery >1pp (w_switch=1.0 (5x) -> -1.5pp: over-penalizing switching makes routing too sticky, costs delivery). The congestion-awareness win is NOT an artifact of a specific reward-weight setting.