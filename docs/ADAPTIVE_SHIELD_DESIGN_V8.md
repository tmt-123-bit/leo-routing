# Adaptive proposed-checkpoint shield design (v8)

## Scope

This is an exposed-panel engineering design stage. It is exploratory, is not a
validation or test, and cannot support a paper claim or model promotion. It
does not train or alter any actor weight. It selects one global post-hoc shield
for the already frozen `proposed` checkpoints.

The design stage may use only the already exposed v7 validation workloads
`60001..60010`. It must not read or run the reserved v7 test panel
`70001..70025` or either v8 holdout panel. Every artifact records
`paper_claim_allowed=false` and `promotion_decision_allowed=false`.

## Frozen candidate grid

All candidates use the multiplicative state-aware rule

```text
effective_bonus = stay_bonus
                * (1 - urgency_relief * clip(urgency, 0, 1))
                * (1 - class_2_relief * class_2)
```

with the Cartesian grid:

- `stay_bonus`: `0.60, 0.80, 1.00`;
- `urgency_relief`: `0.50, 0.75, 1.00`;
- `class_2_relief`: `0.75, 1.00`.

This gives exactly 18 candidates. Each candidate is evaluated for both
scenarios, all four frozen policy seeds, and all ten exposed workloads. One
global parameter triple is selected across all eight scenario by policy-seed
cells. Per-scenario and per-seed tuning are prohibited.

The proposed and raw-context reference rows are reused from the immutable v7
validation artifact. Reuse is allowed only after replaying the v7 spec and
decision hashes, the exact reference grid, checkpoint-source freeze, episode
CSV hash, and the current hashes of every environment/evaluation component
that generated those rows.

## Frozen gates and design buffers

Every one of the selected candidate's eight cells must pass the unchanged v7
gates:

- delivery difference versus proposed at least `-0.003`;
- class-2 delivery difference versus proposed at least `-0.010`;
- avoidable-switch micro-rate strictly below proposed;
- mean routing switches no greater than proposed;
- mean routing switches no greater than raw context.

The avoidable-switch rate is

```text
sum(avoidable_routing_switches) / sum(switch_opportunities)
```

within one scenario by policy-seed by arm cell. It is never the mean of
episode rates.

Because this panel is used for design, eligibility also requires two
predeclared buffers in every cell:

- avoidable-rate improvement at least `0.005`;
- routing-switch headroom versus raw context at least `2.0`
  switches/episode.

No extra delivery or class-2 buffer is added, and the switch-versus-proposed
gate remains unchanged without an extra buffer.

These buffers do not change the later v8 validation/test gates. They only
prevent choosing a development candidate that sits exactly on a boundary.

## Frozen selection rule

Among eligible candidates, choose the unique lexicographic maximum of:

1. worst delivery-gate slack across all eight cells;
2. worst class-2-gate slack across all eight cells;
3. worst routing-switch headroom versus raw context;
4. worst avoidable-rate improvement;
5. negative `stay_bonus`;
6. `urgency_relief`;
7. `class_2_relief`;
8. negative frozen candidate index.

The final index term is a deterministic last-resort tie breaker. If no
candidate passes all gates and buffers, no selection is frozen and v8 cannot
open a holdout panel.

## Frozen artifacts

`design_selection.json` binds the exact 18-candidate results, all 144 cell
rows, all slacks and buffer decisions, the eligible set, the selection key,
the unique argmax, the design spec, source freeze, v7 freeze, runtime code
fingerprint, and every input/output artifact hash. `load_design_selection`
must independently reload the episode CSV, recompute all cells and selection
keys, and reproduce the selected triple before returning it.

## Commands

Read-only audit and plan:

```powershell
F:\leo-venv\Scripts\python.exe src\run_adaptive_shield_design.py --dry-run --device cuda
```

Run the exposed-panel design:

```powershell
F:\leo-venv\Scripts\python.exe src\run_adaptive_shield_design.py --device cuda
```

This command never trains a policy and never opens a fresh or reserved panel.
