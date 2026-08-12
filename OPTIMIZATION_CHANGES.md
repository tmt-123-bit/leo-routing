# Optimization Changelog — toward an IEEE-quality LEO MAPPO routing paper

This log records every code change made in the optimization pass (2026-08-09),
each grounded in the multi-agent audit (`wf_ea50fb1e`) and its adversarial
verification stage. Every change was validated against the 28-test suite
(`test_mappo_design.py`, green throughout) and a training smoke test.

All changes are reversible via git (baseline commit `639e88a` in this repo;
`/f/cleanmarl` is its own repo with the trainer change as a working-tree diff).

> **Batching rule (from the audit):** every environment/reward change here
> INVALIDATES prior experiment tables. They must be re-run as ONE batch before
> any number is reported. See `run_ieee_reproduction.sh`.

---

## Phase 1 — Trainer (root-cause Critic fix)
File: `/f/cleanmarl/cleanmarl/mappo.py` (the trainer actually executed by
`run_exp004_mappo.py`; `cleanmarl_mappo_leo.py` is a byte-identical dead copy).

The Critic was doing sign-SGD: 99.98 % of updates were clipped at 0.5 because
value targets drifted to −10…−100 (raw returns) while normalization was off.

| Change | Was | Now | Why (verified) |
|---|---|---|---|
| `normalize_return` default | `False` | `True` | Root cause of 99.98 % critic clipping. Targets standardized by per-round (ret_mu, ret_std). |
| `clip_gradients` (actor) | `0.5` | `1.0` | Actor median pre-clip was ~0.54 → 0.5 clipped 42–69 %, distorting the PPO trust region. |
| `critic_clip_gradients` (new) | shared 0.5 | `10.0` | Separate critic threshold. Post-normalization critic grad ~1–10; 10.0 bounds spikes, rarely clips healthy updates. |
| Critic loss | `F.mse_loss` | `F.smooth_l1_loss` (Huber) | Caps per-sample gradient from outlier targets (slots with many drops). |

**Validation (600-step smoke, medium_load):** explained_variance 0.026 → 0.58
over 5 updates (previously ~10 K steps to reach 0.7–0.8 → **~10× faster Critic
convergence**). Critic pre-clip grad median dropped from ~36–42 to ~3–10. Actor
no longer over-clipped. KL stays ~1e-4 (target 0.02) — stable.

**Verification correction applied:** the audit synthesis claimed normalization
"nearly eliminates" clipping. The adversarial verifier refuted this with
existing `CURRENT-20260717` data: clipping stays 60–95 % at threshold 0.5
because median grad ~1.0–1.4 still exceeds 0.5. Hence the **separate, higher
`critic_clip_gradients`** — following the verifier, not the raw synthesis.

**`normalize_reward` left False** (verifier: untested anywhere; it modifies
rewards *before* GAE and could interact with terminated/truncated bootstrap).
Test as a separate experiment before adopting.

---

## Phase 2 — Environment fidelity (batched; invalidates prior tables)
Files: `leo_marl_env.py`, `leo_multiagent_env.py`.

### 2.1 Real link breaks (the headline fidelity fix) — `leo_marl_env.py`
`available` was hardcoded `True` for every link, so `frequent_break` had **zero
physical effect** (proven: 300/300 ablation runs byte-identical to `medium_load`).
- Added `_break_uniform(t,u,v)` — a deterministic hash → U[0,1), stable across
  the multiple `_build_topology` calls within one slot.
- short-T_rem links now physically fail with `P(break) = 1 − t_rem/t_safe` per
  slot (≈16.7 % for t_rem=2.5, t_safe=3.0), setting `available=False`.
- **Effect:** the link-lifetime mask becomes *proactive* (avoid soon-to-fail
  links) vs *reactive* in `no_lifetime` — the ablation is now meaningful.
- **Validated:** 16.2 % break rate measured (theory 16.7 %); `medium_load`
  unaffected (0 short-T_rem links).

### 2.2 Bandwidth / congestion (was completely inert) — `leo_marl_env.py`, `leo_multiagent_env.py`
With `capacity=100`, `demand=1`, `link_capacity_packets=1`, the bandwidth mask
**never triggered** and load/bandwidth features carried zero information
(proven: delivery identical across all cap/demand settings). To make
congestion structurally possible:
- `capacity_mbps` 100 → **30**, `packet_demand_mbps` 1 → **10**
- `link_capacity_packets` 1 → **3** (statistical multiplexing — the verifier's
  physically-correct option C)

**Validated:** env stable across all 5 scenarios (no collapse); rho range
~0–0.34 (28× richer signal than baseline 0–0.012). NB: the hard mask triggers
only under traffic that *converges* on a link (a destination-routing policy);
a naive policy spreads load and won't show it. **Final cap/demand/lcp values
must be confirmed with a trained-policy load sweep before publication.**

### 2.3 C_imbalance downweight — `leo_multiagent_env.py`
`global_imbalance_weight` 0.5 → **0.1**. The Jain index is computed over all
edges including zero-utilization ones; with the old inert bandwidth it was an
**anti-signal** (better policies penalized *more*: dijkstra 0.743 vs random
0.587) contributing −0.354/slot — the largest negative reward term. The
verifier proved the proposed `>1e-6` filter is ineffective (load_decay keeps
residuals >1e-6 for ~23 slots); downweighting is the correct fix. The metric
becomes meaningful once congestion binds (2.2).

### 2.4 fault_links reliability — `leo_marl_env.py`
`reliability_penalty` 0.25 → **0.10**. At 0.25, fault-link reliability 0.745 is
below every class floor (0.86/0.88/0.94) → all fault links masked for all
classes (binary, no QoS differentiation). At 0.10, reliability ~0.895 lets
class-0 use fault links while class-2 cannot — activating the class-awareness.

### 2.5 Contention feature (#8, was dead) — `leo_multiagent_env.py`
Candidate feature index 7 was hardcoded `0.0`. Now
`previous_contention[v] / max_degree` — 1-hop local contention at the neighbor
(already tracked for the global state). Gives the actor direct load-balancing
signal.

### 2.6 Delay idle penalty — `leo_multiagent_env.py`
`delay_cost` for a no-traffic slot was `1.0`; now `0.0`. The old value
spuriously penalized idle slots and created a reward discontinuity vs the
delivered/accepted branches.

### 2.7 Credit-assignment bugs — `leo_multiagent_env.py`
- **no_route vanish:** `policy_active` excluded no_route agents (HOL packet,
  no feasible candidate) via `any(action_mask[1:])`, so their penalty was
  multiplied by 0 and they faced no individual consequence. Removed the guard.
- **deadline-drop misattribution:** a packet forwarded to node X at slot t could
  be expired at slot t, penalizing X (zero chance to act). Now attributes to
  `previous_node` (the forwarder that chose the path); falls back to `owner`
  for never-forwarded packets.
- **Deferred (design choice):** delivery bonus (+w_deliver) currently goes only
  to the last hop; intermediates go negative after centering. Splitting it
  across `visited` is a reward-design change left for a dedicated ablation.

Zero-mean credit is preserved by construction (centering is over the active
set). All 28 tests green.

---

## Phase 3 — Training configuration
| Change | File | Was | Now | Why |
|---|---|---|---|---|
| `--train-seed-count` | `run_exp004_mappo.py` + `mappo.py` | 20 | **200** | 20 seeds were each cycled ~83× (not ~21× — the audit's math was 4× off, corrected by the verifier); 13/15 runs peaked then degraded 1–9 %. 200 → ~8 reps each. Same step count, no wall-time cost. |
| LR schedule | `mappo.py` | dead code | **wired** (gated by `lr_decay=True`) | `linear_schedule` existed but was never called. Now holds base LR for the first half, decays to 12.5 % over the second half — stabilizes late training. |

---

## What was NOT changed (and why)
- **Graph-vs-flat Critic decision** — confounded until the Phase-1 fix is in;
  re-run the ablation at 50 K steps with normalization before deciding.
- **Bandwidth exact values** — need a trained-policy load sweep.
- **Reward-weight sensitivity** — prior QUICK sweep ran on a 12 % delivery
  baseline (vs 69 % at FULL); re-run at FULL budget before trusting any weight.
- **ns-3 fixture scaling / Starlink-24 fixture** — requires the ns-3 build run.

## Reproducibility hazard flagged (not a code change — a process fix)
`run_exp004_mappo.py` defaults to `--mode quick` (300 steps); the README never
documents `--mode full`. A reviewer running the default gets 300-step MAPPO
that loses to Q-routing. **Before submission:** re-run at FULL and document the
exact command (see `run_ieee_reproduction.sh`). Note code-fingerprint hashes
differ between the FULL and V2 runs — treat code drift as a co-suspect with
training budget.

---

## Phase 4 — Baselines, ablation coverage, budget presets (anti-rejection pass)
Added 2026-08-10 to close the three code-side items on the audit's
anti-rejection checklist that do not require a full re-run to land.

### 4a. OSPF/ECMP baseline (the #1 missing baseline)
File: `mappo_evaluation.py` (new `OspfEcmpPolicy`) + `run_exp004_mappo.py`
(import, evaluation loop, `paired_tests` baseline list).

The prior baseline set was {delay_only, full_heuristic, random, global_dijkstra,
q_routing}. A LEO routing paper without **OSPF/ECMP** — the canonical distributed
incumbent — is desk-reject bait: reviewers cannot tell how much of MAPPO's gain
is "learning" versus "having a sensible shortest-path policy at all".

`OspfEcmpPolicy` forwards along the delay-weighted shortest path (one reverse
Dijkstra from the destination, cached per `(slot, dst)` since the topology is
frozen within a slot) and splits load across the **equal-cost multi-path
(ECMP)** next-hop set by seeded random choice. It uses the same information set
as `GlobalDijkstraPolicy` (full current link state), so it sits in the
centralized-oracle column; the only difference from plain Dijkstra is ECMP load
splitting, which is exactly where it should help under multipath/hotspot traffic.

**Validation (3-seed smoke, medium/frequent/fault):** no crash, **zero mask
violations**, delivery within ±0.2 pp of Dijkstra on light loads — correct, since
ECMP only diverges from shortest-path once traffic converges on shared links.
The interesting MAPPO-vs-ECMP numbers will appear in hotspot/fault after the
FULL re-run.

### 4b. Ablation across all 5 scenarios
File: `run_ablation_experiments.py` — `SCENARIOS` expanded from
`["medium_load", "frequent_break"]` to all five (`ALL_SCENARIOS`).

The original ablation ran 2 scenarios; two of four headline ablation conclusions
flipped or vanished in the 2nd scenario, so generalizing from 2 is unsafe. The
`--scenarios` choices now accept all five, so `run_ieee_reproduction.sh` STEP 3
runs without an argparse error. With the Phase-2 real-break fix, the lifetime
ablation (`no_lifetime`) is now a meaningful test (was vacuous when
`available=True` was hardcoded).

### 4c. Budget-sensitivity presets
File: `run_exp004_mappo.py` — `--mode` now accepts `quick | x2k | x10k | full`;
`mode_config()` adds `x2k` (2 000 steps) and `x10k` (10 000 steps) presets over
all five scenarios. `run_ieee_reproduction.sh` STEP 4 (`budget` stage) loops
these to produce the delivery-vs-steps crossover figure — turning the old
reproducibility bug (quick loses, full wins) into a characterized
sample-efficiency result.

**Validation:** 28/28 tests green; `--help` shows `{quick,x2k,x10k,full}`;
ablation accepts the 5-scenario invocation; OSPF/ECMP end-to-end smoke passes.

### 4d. Minimum-detectable-effect (MDE) report
File: `compute_mde.py` (new) + `run_ieee_reproduction.sh` STEP 5 (`mde` stage).

Every "no significant difference" claim needs an MDE or a reviewer rejects it as
possibly underpowered. The script reads `episode_metrics.csv`, reconstructs
per-workload paired differences (averaging over policy seeds, mirroring
`paired_tests`), and reports for every (scenario, metric, policy-pair):

    MDE = (z_{1-alpha/2} + z_{1-power}) * sigma_d / sqrt(n)

plus the observed difference, Wilcoxon p, and a flag for nulls whose observed
effect is below the MDE. Parametric (paired-t) bound labelled as approximate for
the Wilcoxon test — standard practice.

**Validation (old EXP-004-FULL data):** frequent_break MAPPO vs global_dijkstra
— observed +0.08 pp, Wilcoxon p = 1.0, **MDE = 1.02 pp (2.5 % of baseline)**.
So the claim "MAPPO ≈ Dijkstra on frequent_break" is honest up to ±1 pp — any
difference ≥ ~1 pp would have been detected. Re-run on the new data after the
FULL re-run.

### 4e. Reproducibility manifest
File: `write_repro_manifest.py` (new).

Emits one top-level `repro_manifest.json` with: pip freeze, git HEAD + dirty
status of both repos, SHA256 of every project + cleanmarl source file, and
optional fixture hashes. Closes the last anti-rejection item. Run with the
experiment python before submission (commit both repos first for a clean
manifest). **Validation:** ran cleanly — 71 source files + 201 pip pkgs hashed;
correctly flagged both repos as dirty (uncommitted optimization changes).

---

## Phase 5 — GPU enablement (infrastructure, not algorithm)
The project's compute bottleneck was not the algorithm — it was that PyTorch was
the **CPU-only** build while the machine has an idle **RTX 3070 Laptop GPU
(8 GB)**. Set up to run on GPU:

- `F:\leo-venv` — project venv on F: (honors "heavy stuff on F:"; C: was 89 % full)
  with CUDA torch; installer = `setup_venv_f.sh` (pip cache on F:, CUDA-index
  fallback cu128→cu126→cu124→cu118).
- `run_ieee_reproduction.sh` auto-detects the venv (`PY=/f/leo-venv/Scripts/python.exe`)
  when present; `DEVICE=cuda` threads `--device cuda` through to the trainer
  subprocess (verified: `train_one` uses `sys.executable` + `--device args.device`).
- `GPU_SETUP.md` — driver-update + reboot + verify guide (driver 512.36/CUDA 11.6
  is too old for modern CUDA torch; latest 610.88 supports CUDA 12.x).
- Expected: 10–50× speedup; the 50 K×8-seed×5-scenario suite goes from
  infeasible (days on CPU) to ~tens of minutes–hours on the RTX 3070.

**Algorithm-correctness audit (done before the GPU run, no changes needed):**
GAE terminated/truncated bootstrap ✓, advantage normalization over valid+active
✓, return normalization symmetric on prediction+target ✓, actor loss masked ✓,
action masking in both sampling and loss paths ✓, KL early-stop wired ✓,
separate gradient clipping intact ✓. The learning implementation is sound; the
only real fix was the Phase-1 `normalize_return`.

