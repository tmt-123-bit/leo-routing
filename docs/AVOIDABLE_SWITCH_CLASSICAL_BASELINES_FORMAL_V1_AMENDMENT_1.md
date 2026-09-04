# Avoidable-Switch Classical Baselines Formal v1 - Amendment 1

Authorized on 2026-09-04 before any independent-gate or sealed-test workload
was instantiated. The first execution completed all 16 frozen Q-routing
training jobs, then failed while validating `training_freeze.json`.

The writer emits canonical JSON with lexicographically sorted object keys, but
the validator compared the deserialized `jobs` mapping's iteration order with
the registry's generation order. JSON object order has no scientific meaning.
All job identifiers and every job entry were otherwise validated separately.

The persisted training and gate validators are corrected to require exact
equality of the sorted job-ID sets, while retaining the existing per-job
manifest, model, status, attempt, and hash checks. The first clean rerun
completed all 16 training jobs and all 34 independent-gate shards before the
same order-only defect was exposed in the gate shard inventory validator. It
is retained at `experiments/avoidable-switch-classical-baselines-formal-v1-r2`.
No environment, method, hyperparameter, seed, workload, metric, retry, or
statistical rule changes.

The failed directory
`experiments/avoidable-switch-classical-baselines-formal-v1` is retained as
incident evidence. A clean symmetric rerun is authorized at
`experiments/avoidable-switch-classical-baselines-formal-v1-r3`; no artifact
from either failed directory may enter the r3 evidence chain. Sealed-test
access remains forbidden.
