# Age-band shield design result (v11)

## Decision

V11 completed on 2026-09-02 (Asia/Shanghai). The immutable global decision is:

```text
selection_status = no_eligible_candidate
eligible_candidate_indices = []
selected_candidate_index = null
all_cells_pass = false
```

This is an exposed-panel engineering result. It is not validation or test
evidence, does not authorize model promotion, and cannot support a paper
claim. No fresh validation or test workload was accessed.

## Replayed evidence

The completed selection was accepted by the full fail-closed
`load_design_selection` replay. The evidence identifiers are:

```text
v11 design-spec content SHA-256:
14cede5336b7b2ed3b63275fa85a95e6448bb979d251cee7486d016471e4cb22

v11 design-selection internal SHA-256:
133f2ab88b5fc5291b29b92f1cb61ebf9512e71450a9b82a431a916c1d6814c5

v11 design_selection.json file SHA-256:
ac338b5967b985c2efafa4d728c1350d54e45393308213267e3af663323ed19c

bound v10 design-selection internal SHA-256:
1f51e4a0bdfab0381e224403193a38783b1e54b6b550a3f6450d8754a05765f2
```

Execution performed no training. It reused the frozen control's 8 logical
jobs and 80 episode rows, and evaluated exactly 8 new jobs and 80 new episodes
for `age_band_mix=1.0`, using only workloads `60001..60010`.

The repository test suite passed `344/344`. A completed-state runner reentry
also replayed the selection without modifying any of the 22 v11 artifact
files: hashes, lengths, and modification times were identical before and
after reentry.

## Candidate 1 cell results

Buffer slack is reported after subtracting the frozen minimum buffer. A
negative value is a strict failure and is not rounded to zero.

| Scenario | Policy seed | Delivery diff. | Delivery buffer slack | Raw-switch buffer slack | Avoidable buffer slack | Result |
|---|---:|---:|---:|---:|---:|---|
| `medium_load` | 1710210210 | 0.003125 | 0.005604167 | 33.0 | 0.052737576 | pass |
| `medium_load` | 2078783072 | 0.000000 | 0.002479167 | 11.2 | 0.064230184 | pass |
| `medium_load` | 1047581915 | -0.002604167 | -0.000125000 | 13.8 | 0.043305255 | fail: delivery buffer |
| `medium_load` | 1245825580 | -0.003125000 | -0.000645833 | 11.3 | 0.049741598 | fail: delivery gate and buffer |
| `hotspot_high_load` | 1710210210 | 0.001800 | 0.004279167 | 24.9 | 0.047113333 | pass |
| `hotspot_high_load` | 2078783072 | 0.001800 | 0.004279167 | 55.4 | 0.033995107 | pass |
| `hotspot_high_load` | 1047581915 | 0.001400 | 0.003879167 | 87.3 | 0.038136292 | pass |
| `hotspot_high_load` | 1245825580 | 0.003000 | 0.005479167 | -0.2 | 0.096098426 | fail: raw-switch buffer |

Candidate 1 passed every class-2 and avoidable-switch requirement. Its global
worst slacks were:

```text
delivery-buffer slack:   -0.0006458333333332667
class-2 gate slack:       0.002531167967360073
raw-switch buffer slack: -0.20000000000000284
avoidable buffer slack:   0.03399510749952826
```

Moving the calm penalty into the middle age band substantially improved the
control candidate's worst delivery-buffer slack from `-0.0037708333` to
`-0.0006458333`. It did not satisfy the frozen global rule: two medium-load
cells retained delivery failures, and one hotspot cell lost 0.2 switches of
the required raw-context headroom.

## Termination

The preregistered v11 rule declares this the final exposed-panel design
attempt. Because the result is `no_eligible_candidate`, the shield design
search on this panel stops. The observed near misses cannot be used to add a
v12 parameter, change a band edge, relax a buffer, or open the reserved fresh
validation/test panels. Doing so would turn the remaining evidence into
post-selection tuning and would invalidate the intended claim boundary.
