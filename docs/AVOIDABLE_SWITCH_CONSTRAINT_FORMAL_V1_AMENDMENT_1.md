# AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1 — Amendment 1

Operator-authorized on 2026-09-03 (local UTC+8). This amendment is written
before the symmetric rerun it authorizes, as required by the protocol's
documented-amendment clause. It changes no frozen parameter, no seed identity,
no code, and no statistical contract.

## 1. Incident record

Two runner interruptions occurred on 2026-09-03 during the first execution of
`experiments/avoidable-switch-constraint-formal-v1`
(invocations `93cc772e…` and `e3b0c390…`):

### 1.1 First interruption (~09:48 UTC, orphaned process tree)

The entire runner process tree (runner PID 1368 and trainer children) died
without a traceback; stdout ended mid-heartbeat and stderr remained empty.
Classification and recovery were fully inside the frozen contract: both
in-flight jobs were recovered on re-entry as
`orphaned_process_interruption` with `retry_authorized=true` and resumed
exactly from same-run `latest.pt` copies (attempt 2, resume sources hashed).
This interruption required no amendment and is recorded here only for
completeness.

### 1.2 Second interruption (12:18:12–13 UTC, EACCES on atomic status writes)

Immediately after scheduling `hotspot_high_load/qos_only_constrained/
seed_2136406109`, the runner's atomic status write
(`os.replace` of a `.tmp` file onto the job-status JSON) failed with
`PermissionError [WinError 5]` (errno 13, `EACCES`). Within the same second
two further atomic writes failed the same way: the write recording that
job's failure, and the write recording the invocation failure. The runner
then died on the unhandled `PermissionError` in its exception path.

Evidence (all preserved unmodified):

- stderr traceback: `experiments/avoidable-switch-constraint-formal-v1.stderr.r2.log`
- stdout event log: `experiments/avoidable-switch-constraint-formal-v1.stdout.r2.log`
  (final lines: completion of `reward_shaped_control/seed_310818925`,
  failure of `qos_only_constrained/seed_2136406109`)
- job-status records:
  `job_status/training/hotspot_high_load__qos_only_baseline__seed_2136406109.json`
  (attempt 1 `interrupted`, category `execution_interruption`,
  `retry_authorized=false`) and
  `job_status/training/hotspot_high_load__qos_only_constrained__seed_2136406109.json`
  (stale `running` attempt; the failure-status write itself failed).

Consequences under the frozen retry contract:

- `EACCES` is not in the frozen retryable errno allowlist
  (`EAGAIN EBUSY ECONNABORTED ECONNRESET EHOSTUNREACH EINTR EIO ENETDOWN
  ENETRESET ENETUNREACH EPIPE ESTALE ETIMEDOUT`), so the failure was
  fail-closed classified as non-infrastructure.
- The in-flight parallel worker `qos_only_baseline/seed_2136406109`
  (then at the ~45,000-step boundary, `step_45000.pt` written) was cancelled
  by fail-fast shutdown. Because its cancellation cause was a non-infrastructure
  failure, the contract records it as `execution_interruption` with
  `retry_authorized=false`, and re-entry refuses the job
  (`_require_retry_authorization` raises before any submission).
- The training matrix therefore cannot complete inside the original output
  directory: 39/48 jobs completed, 1 further job
  (`qos_only_constrained/seed_2136406109`) had its orphaned child run to
  completion (`final.pt` and `run_manifest.json` written 12:33 UTC) and would
  be recoverable, 1 job is retry-ineligible, and 7 jobs never started.
- Per the protocol: no seed replacement, no imputation, no complete-case
  headline analysis.

## 2. Root-cause assessment

Three independent `os.replace` destination-lock failures within one second,
in two different subdirectories (`job_status/training/` and `invocations/`),
on a machine where the same write pattern had succeeded hundreds of times
earlier the same day. This is the well-known Windows pattern in which an
on-access scanner or indexer briefly holds a freshly written destination file
without delete sharing, making the atomic rename fail transiently. It is an
environmental fault of the execution host, not a code, schema, or scientific
computation defect; no source file changed before or after the incident
(fingerprint re-verified on re-entry at 11:42 UTC).

The frozen errno allowlist does not include `EACCES`. Extending the allowlist
would require a code change (breaking the frozen code fingerprint), so no
in-place retry is contractually possible regardless of the classification
dispute.

## 3. Disposition (operator decision, 2026-09-03)

1. The original directory
   `experiments/avoidable-switch-constraint-formal-v1` is preserved
   unmodified as failed-attempt evidence. Its 39 completed, self-hashed job
   artifacts remain individually valid but the directory can never produce a
   training or validation freeze.
2. A symmetric rerun is authorized in the fresh output directory
   `experiments/avoidable-switch-constraint-formal-v1-r2`, executed by the
   unchanged runner (`src/run_avoidable_switch_constraint_formal.py`,
   identical code fingerprint) with identical frozen parameters. The spec
   does not bind the output path, so the r2 preregistration reproduces the
   original `spec_sha256`
   (`58fd1e77faafefcb67631c3a568116ad60ce2ba3c90c12de1262d6a3709caaf6`)
   byte-for-byte. Seed identities are unchanged; no seed is replaced.
3. No artifact, row, checkpoint, or status record from the failed directory
   enters the r2 evidence chain. The r2 chain stands entirely on its own.
4. A recurrence of the EACCES fault during r2 is possible; if it occurs, the
   same fail-closed rules apply and the matrix again remains incomplete
   pending a further operator decision.
5. The host-level mitigation (antivirus real-time-scan exclusion for the
   experiments tree) was attempted before the rerun; if it could not be
   applied without elevated privileges, that fact is recorded here and the
   rerun proceeds at the noted recurrence risk.

## 4. What this amendment does not change

- The frozen retry contract, seed registry, workload panels, training budget,
  checkpoint-selection rule, statistical test, and validation gate.
- The sealed-test boundary: workload seeds `78001..78050` remain untouched;
  r2 can still only point to the classical-baseline freeze and joint
  authorization as the next required stage.
- The claim contract in `docs/ICC_SUBMISSION_SCOPE.md`, which does not depend
  on this study.
