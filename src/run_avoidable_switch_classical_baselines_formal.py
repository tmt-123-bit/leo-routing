"""Run the preregistered classical-baseline training and gate audit.

This runner intentionally has no sealed-panel execution path.  It retrains the
frozen Q-routing grid, freezes all model arrays, and then evaluates Q-routing,
OSPF-ECMP, and the deterministic Global Dijkstra reference on the independent
gate panel.  The gate rows are descriptive and cannot tune or rescue MAPPO.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import errno
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import numpy as np
import torch

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from ablation_matrix_runner import atomic_write_csv, atomic_write_json, sha256_file
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_evaluation import (
    ConstraintEpisodeMetrics,
    GlobalDijkstraPolicy,
    OspfEcmpPolicy,
    QRoutingPolicy,
    evaluate_policy_with_constraint_metrics,
    metrics_as_dicts,
)


SCHEMA_VERSION = 1
STUDY_NAME = "ICC-AVOIDABLE-SWITCH-CLASSICAL-BASELINES-FORMAL-v1"
PROTOCOL_FILENAME = "AVOIDABLE_SWITCH_CLASSICAL_BASELINES_FORMAL_V1.md"
METHOD_PROTOCOL_FILENAME = "AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1.md"
DEFAULT_OUTPUT_DIRECTORY_NAME = "avoidable-switch-classical-baselines-formal-v1"

SCENARIOS = ("medium_load", "hotspot_high_load")
Q_ROUTING = "q_routing"
OSPF_ECMP = "ospf_ecmp"
GLOBAL_DIJKSTRA = "global_dijkstra"
METHODS = (Q_ROUTING, OSPF_ECMP, GLOBAL_DIJKSTRA)
ENVIRONMENT_VARIANT = "qos_only"

POLICY_SEEDS = (
    179055553,
    626669596,
    746965870,
    183895110,
    310818925,
    2136406109,
    985998595,
    515636025,
)
TRAIN_WORKLOAD_SEEDS = tuple(range(76001, 76201))
SELECTION_WORKLOAD_SEEDS_METADATA_ONLY = tuple(range(77001, 77011))
GATE_WORKLOAD_SEEDS = tuple(range(77011, 77021))

# The sealed range is metadata only.  No iterable of its seeds and no function
# accepting a panel/range are exposed by this runner.
SEALED_PANEL_METADATA_ONLY = {
    "start": 78001,
    "end": 78050,
    "count": 50,
    "access_authorized": False,
}

Q_TRAIN_EPISODES = 500
Q_N_NODES = 24
Q_ALPHA = 0.3
Q_TRAIN_EPSILON = 0.1
Q_FROZEN_EPSILON = 0.0
Q_INITIAL_VALUE = 10.0
Q_DTYPE = "float32"
Q_SHAPE = (Q_N_NODES + 1, Q_N_NODES + 1, Q_N_NODES + 1)
GLOBAL_DIJKSTRA_SENTINEL_SEED = -1
MAX_ATTEMPTS = 2
MAX_PARALLEL = 2

RETRYABLE_INFRASTRUCTURE_EXCEPTION_NAMES = ("TimeoutError", "ConnectionError")
RETRYABLE_INFRASTRUCTURE_ERRNO_NAMES = (
    "EAGAIN",
    "EBUSY",
    "ECONNABORTED",
    "ECONNRESET",
    "EHOSTUNREACH",
    "EINTR",
    "EIO",
    "ENETDOWN",
    "ENETRESET",
    "ENETUNREACH",
    "EPIPE",
    "ESTALE",
    "ETIMEDOUT",
)
RETRYABLE_INFRASTRUCTURE_ERRNOS = frozenset(
    getattr(errno, name)
    for name in RETRYABLE_INFRASTRUCTURE_ERRNO_NAMES
    if hasattr(errno, name)
)
RETRYABLE_FAILURE_CATEGORIES = {
    "recognized_infrastructure_failure",
    "recognized_infrastructure_cancellation",
    "orphaned_process_interruption",
}

EXPECTED_TRAINING_JOBS = len(SCENARIOS) * len(POLICY_SEEDS)
EXPECTED_Q_TRAIN_EPISODES = EXPECTED_TRAINING_JOBS * Q_TRAIN_EPISODES
EXPECTED_Q_GATE_ROWS = len(SCENARIOS) * len(POLICY_SEEDS) * len(GATE_WORKLOAD_SEEDS)
EXPECTED_OSPF_GATE_ROWS = EXPECTED_Q_GATE_ROWS
EXPECTED_DIJKSTRA_GATE_ROWS = len(SCENARIOS) * len(GATE_WORKLOAD_SEEDS)
EXPECTED_GATE_ROWS = (
    EXPECTED_Q_GATE_ROWS + EXPECTED_OSPF_GATE_ROWS + EXPECTED_DIJKSTRA_GATE_ROWS
)
EXPECTED_EVALUATION_JOBS = (
    len(SCENARIOS) * (len(POLICY_SEEDS) + len(POLICY_SEEDS) + 1)
)

Q_ACTION_SCORE_FORMULA = (
    "Q[satellite,destination,neighbor] + candidate_feature[2] "
    "+ candidate_feature[1] + candidate_feature[4]"
)
Q_ACTION_SCORE_COMPONENTS = (
    "tabular_q",
    "normalized_edge_delay",
    "normalized_neighbor_queue_occupancy",
    "link_load_rho",
)
Q_TRANSITION_TARGET_FORMULA = (
    "normalized_edge_delay + normalized_neighbor_queue_occupancy "
    "+ min_next_neighbor_Q"
)
Q_ROUTING_CLASS_SOURCE_SHA256 = (
    "d09d7cf6d16be0d2d8d939f4fbec01151867ff7ccf2fdb3c9814e170281d9fe2"
)
OSPF_ECMP_CLASS_SOURCE_SHA256 = (
    "9752b3b5a093412e44bc85d3dd3440c357949948140dc4d0a208f4ad8fce327b"
)
GLOBAL_DIJKSTRA_CLASS_SOURCE_SHA256 = (
    "4045cf6504e80ef0eef232fe3738be9c97663095449f0e0982ff7011db2a44f3"
)
STRUCTURED_EVALUATOR_SOURCE_SHA256 = (
    "3ef7f984f240529ed34743fb76e5c846660838e6a952c4951fb3984810956ec6"
)
METHOD_SOURCE_SHA256 = {
    Q_ROUTING: Q_ROUTING_CLASS_SOURCE_SHA256,
    OSPF_ECMP: OSPF_ECMP_CLASS_SOURCE_SHA256,
    GLOBAL_DIJKSTRA: GLOBAL_DIJKSTRA_CLASS_SOURCE_SHA256,
}
ACTUAL_PROJECT_SOURCE = Path(__file__).resolve().parent


class ConfigurationError(ValueError):
    """Raised when runtime state differs from the frozen protocol."""


class JobCancelled(RuntimeError):
    """Raised at a job boundary after another parallel job fails."""

    def __init__(self, message: str, *, infrastructure_cause: bool = False):
        super().__init__(message)
        self.infrastructure_cause = infrastructure_cause


@dataclass(frozen=True)
class QTrainingJob:
    index: int
    scenario: str
    policy_seed: int

    @property
    def job_id(self) -> str:
        return f"{self.scenario}/{Q_ROUTING}/seed_{self.policy_seed}"

    @property
    def slug(self) -> str:
        return f"{self.scenario}__{Q_ROUTING}__seed_{self.policy_seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "scenario": self.scenario,
            "method": Q_ROUTING,
            "policy_seed": self.policy_seed,
            "replicate_kind": "independently_trained_model",
        }


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    scenario: str
    method: str
    policy_seed: int
    replicate_kind: str

    @property
    def job_id(self) -> str:
        return f"{self.scenario}/{self.method}/seed_{self.policy_seed}"

    @property
    def slug(self) -> str:
        return f"{self.scenario}__{self.method}__seed_{self.policy_seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "scenario": self.scenario,
            "method": self.method,
            "policy_seed": self.policy_seed,
            "replicate_kind": self.replicate_kind,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    if field in result:
        raise ConfigurationError(f"self-hash field already exists: {field}")
    result[field] = sha256_json(result)
    return result


def validate_self_hash(record: Mapping[str, Any], field: str) -> None:
    observed = record.get(field)
    body = {key: value for key, value in record.items() if key != field}
    if not isinstance(observed, str) or observed != sha256_json(body):
        raise ConfigurationError(f"self-hash mismatch: {field}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON artifact is not an object: {path}")
    return value


def ensure_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != dict(value):
            raise ConfigurationError(f"immutable artifact mismatch: {path}")
        return
    atomic_write_json(path, dict(value))


def _source_hash(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def source_contract() -> dict[str, Any]:
    observed = {
        "QRoutingPolicy": _source_hash(QRoutingPolicy),
        "OspfEcmpPolicy": _source_hash(OspfEcmpPolicy),
        "GlobalDijkstraPolicy": _source_hash(GlobalDijkstraPolicy),
        "evaluate_policy_with_constraint_metrics": _source_hash(
            evaluate_policy_with_constraint_metrics
        ),
    }
    expected = {
        "QRoutingPolicy": Q_ROUTING_CLASS_SOURCE_SHA256,
        "OspfEcmpPolicy": OSPF_ECMP_CLASS_SOURCE_SHA256,
        "GlobalDijkstraPolicy": GLOBAL_DIJKSTRA_CLASS_SOURCE_SHA256,
        "evaluate_policy_with_constraint_metrics": STRUCTURED_EVALUATOR_SOURCE_SHA256,
    }
    if observed != expected:
        drift = {
            name: {"expected": expected[name], "observed": observed[name]}
            for name in expected
            if observed[name] != expected[name]
        }
        raise ConfigurationError(f"classical source contract drifted: {drift}")
    signature = inspect.signature(QRoutingPolicy)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
    }
    if defaults != {
        "n_nodes": Q_N_NODES,
        "alpha": Q_ALPHA,
        "epsilon": Q_TRAIN_EPSILON,
        "seed": 0,
    }:
        raise ConfigurationError("QRoutingPolicy constructor defaults drifted")
    return {
        "object_source_sha256": observed,
        "q_routing": {
            "action_score_formula": Q_ACTION_SCORE_FORMULA,
            "action_score_components": list(Q_ACTION_SCORE_COMPONENTS),
            "transition_target_formula": Q_TRANSITION_TARGET_FORMULA,
            "transition_update_scope": (
                "every_pre_contention_pending_proposal_without_acceptance_filter"
            ),
            "bootstrap_neighbor_scope": (
                "all_base_graph_neighbors_without_current_feasibility_filter"
            ),
            "epsilon_sampling": (
                "independent_per_decision_bernoulli_then_uniform_candidate"
            ),
            "exact_ties": "first_in_deterministic_candidate_order",
        },
        "ospf_ecmp_tolerance": 1e-6,
        "global_dijkstra_replication": "once_per_scenario_and_workload",
        "structured_evaluator": (
            "mappo_evaluation.evaluate_policy_with_constraint_metrics"
        ),
    }


def build_training_jobs() -> list[QTrainingJob]:
    return [
        QTrainingJob(index, scenario, seed)
        for index, (scenario, seed) in enumerate(
            (scenario, seed)
            for scenario in SCENARIOS
            for seed in POLICY_SEEDS
        )
    ]


def build_evaluation_jobs() -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    for scenario in SCENARIOS:
        for seed in POLICY_SEEDS:
            jobs.append(
                EvaluationJob(
                    len(jobs),
                    scenario,
                    Q_ROUTING,
                    seed,
                    "independently_trained_model",
                )
            )
        for seed in POLICY_SEEDS:
            jobs.append(
                EvaluationJob(
                    len(jobs),
                    scenario,
                    OSPF_ECMP,
                    seed,
                    "stochastic_routing_replicate",
                )
            )
        jobs.append(
            EvaluationJob(
                len(jobs),
                scenario,
                GLOBAL_DIJKSTRA,
                GLOBAL_DIJKSTRA_SENTINEL_SEED,
                "deterministic_workload_reference",
            )
        )
    return jobs


def q_training_workload_order() -> tuple[int, ...]:
    return tuple(
        TRAIN_WORKLOAD_SEEDS[episode % len(TRAIN_WORKLOAD_SEEDS)]
        for episode in range(Q_TRAIN_EPISODES)
    )


def validate_seed_registry() -> None:
    if len(set(POLICY_SEEDS)) != 8 or any(seed < 0 for seed in POLICY_SEEDS):
        raise ConfigurationError("policy-seed registry drifted")
    panels = {
        "train": set(TRAIN_WORKLOAD_SEEDS),
        "selection_metadata_only": set(SELECTION_WORKLOAD_SEEDS_METADATA_ONLY),
        "gate": set(GATE_WORKLOAD_SEEDS),
        "sealed_metadata_only": set(
            range(
                SEALED_PANEL_METADATA_ONLY["start"],
                SEALED_PANEL_METADATA_ONLY["end"] + 1,
            )
        ),
    }
    names = tuple(panels)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if panels[left] & panels[right]:
                raise ConfigurationError(f"workload panels overlap: {left}/{right}")
    if q_training_workload_order() != (
        TRAIN_WORKLOAD_SEEDS * 2 + TRAIN_WORKLOAD_SEEDS[:100]
    ):
        raise ConfigurationError("Q-routing workload traversal drifted")
    if len(build_training_jobs()) != EXPECTED_TRAINING_JOBS:
        raise ConfigurationError("Q-routing training grid drifted")
    if len(build_evaluation_jobs()) != EXPECTED_EVALUATION_JOBS:
        raise ConfigurationError("classical evaluation grid drifted")


def validate_gate_workloads(workload_seeds: Iterable[int]) -> tuple[int, ...]:
    observed = tuple(int(seed) for seed in workload_seeds)
    if observed != GATE_WORKLOAD_SEEDS:
        raise ConfigurationError("classical evaluation workload panel is not the gate")
    return observed


def _git_output(repository_root: Path, arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigurationError("git fingerprint command failed") from error


def validate_project_source(project: Path) -> Path:
    observed = project.resolve()
    if observed != ACTUAL_PROJECT_SOURCE:
        raise ConfigurationError(
            "project source must equal the directory containing the imported runner"
        )
    return ACTUAL_PROJECT_SOURCE


def runtime_source_paths(project_source: Path) -> dict[str, Path]:
    project_source = validate_project_source(project_source)
    repository_root = project_source.parent.resolve()
    return {
        "runner": Path(__file__).resolve(),
        "classical_protocol": repository_root / "docs" / PROTOCOL_FILENAME,
        "method_protocol": repository_root / "docs" / METHOD_PROTOCOL_FILENAME,
        "orchestration_dependency": project_source / "ablation_matrix_runner.py",
        "evaluation": project_source / "mappo_evaluation.py",
        "design": project_source / "mappo_design.py",
        "environment": project_source / "leo_multiagent_env.py",
        "base_environment": project_source / "leo_marl_env.py",
        "wrapper": project_source / "cleanmarl_leo_multiagent_wrapper.py",
        "variant_definitions": project_source / "variant_definitions.py",
    }


def code_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    project_source = validate_project_source(args.project)
    repository_root = project_source.parent.resolve()
    paths = runtime_source_paths(project_source)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ConfigurationError(f"classical runtime sources are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    relevant = [
        str(path.relative_to(repository_root)).replace("\\", "/")
        for path in paths.values()
    ]
    diff = _git_output(repository_root, ["diff", "--binary", "HEAD", "--", *relevant])
    try:
        dependency_process = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise ConfigurationError("cannot capture Python dependency inventory") from error
    dependency_inventory = sorted(
        {
            line.strip()
            for line in dependency_process.stdout.splitlines()
            if line.strip()
        },
        key=lambda line: (line.casefold(), line),
    )
    dependency_inventory_bytes = (
        ("\n".join(dependency_inventory) + "\n").encode("utf-8")
        if dependency_inventory
        else b""
    )
    return {
        "files": hashes,
        "source_contract": source_contract(),
        "git_head": _git_output(repository_root, ["rev-parse", "HEAD"]).strip(),
        "relevant_dirty_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "dependencies": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
            "python_dependency_inventory_command": [
                str(Path(sys.executable).resolve()),
                "-m",
                "pip",
                "freeze",
                "--all",
            ],
            "python_dependency_inventory": dependency_inventory,
            "python_dependency_inventory_normalization": (
                "trim_nonempty_exact_deduplicate_casefold_then_codepoint_sort_lf"
            ),
            "python_dependency_inventory_sha256": hashlib.sha256(
                dependency_inventory_bytes
            ).hexdigest(),
        },
    }


def assert_runtime_fingerprint(
    args: argparse.Namespace, expected: Mapping[str, Any]
) -> None:
    if code_fingerprint(args) != dict(expected):
        raise ConfigurationError("classical runtime fingerprint changed")


def retry_contract() -> dict[str, Any]:
    return {
        "maximum_attempts_per_job": MAX_ATTEMPTS,
        "automatic_and_reentry_retry_rule": (
            "recognized_infrastructure_failures_only"
        ),
        "retryable_exception_types": list(
            RETRYABLE_INFRASTRUCTURE_EXCEPTION_NAMES
        ),
        "retryable_oserror_errno_names": list(
            RETRYABLE_INFRASTRUCTURE_ERRNO_NAMES
        ),
        "orphaned_running_attempt_retryable": True,
        "fail_fast_cancellation_retry_rule": (
            "retryable_only_when_bound_to_recognized_infrastructure_cause"
        ),
        "configuration_audit_scientific_fail_fast": True,
        "reentry_requires_recorded_authorization": True,
    }


def q_contract() -> dict[str, Any]:
    order = q_training_workload_order()
    return {
        "implementation": "mappo_evaluation.QRoutingPolicy",
        "n_nodes": Q_N_NODES,
        "alpha": Q_ALPHA,
        "training_epsilon": Q_TRAIN_EPSILON,
        "frozen_epsilon": Q_FROZEN_EPSILON,
        "q_initialization": Q_INITIAL_VALUE,
        "q_dtype": Q_DTYPE,
        "q_shape": list(Q_SHAPE),
        "training_episodes": Q_TRAIN_EPISODES,
        "performance_early_stopping": False,
        "workload_order": list(order),
        "workload_order_sha256": sha256_json(list(order)),
        "action_score_formula": Q_ACTION_SCORE_FORMULA,
        "action_score_components": list(Q_ACTION_SCORE_COMPONENTS),
        "transition_target_formula": Q_TRANSITION_TARGET_FORMULA,
        "transition_update_scope": (
            "every_pre_contention_pending_proposal_without_acceptance_filter"
        ),
        "bootstrap_neighbor_scope": (
            "all_base_graph_neighbors_without_current_feasibility_filter"
        ),
        "epsilon_sampling": {
            "trial_scope": "independent_per_decision",
            "probability": Q_TRAIN_EPSILON,
            "conditional_choice": "uniform_over_ordered_feasible_candidates",
        },
        "discount_factor": None,
        "replay_buffer": False,
        "target_network": False,
    }


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    project_source = validate_project_source(args.project)
    validate_seed_registry()
    fingerprint = code_fingerprint(args)
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "stage": "classical_training_and_independent_gate_only",
        "paper_claim_allowed": False,
        "performance_gate_applied": False,
        "mappo_tuning_allowed": False,
        "promotion_decision_allowed": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "sealed_test_access_authorized": False,
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "environment_variant": ENVIRONMENT_VARIANT,
        "policy_seeds": list(POLICY_SEEDS),
        "workload_registry": {
            "training": list(TRAIN_WORKLOAD_SEEDS),
            "checkpoint_selection_metadata_only": list(
                SELECTION_WORKLOAD_SEEDS_METADATA_ONLY
            ),
            "independent_gate": list(GATE_WORKLOAD_SEEDS),
            "sealed_test_metadata_only": dict(SEALED_PANEL_METADATA_ONLY),
        },
        "q_routing": q_contract(),
        "ospf_ecmp": {
            "implementation": "mappo_evaluation.OspfEcmpPolicy",
            "tie_tolerance": 1e-6,
            "rng_initialization": "once_before_each_complete_policy_replicate",
            "rng_reset_between_workloads": False,
            "replicate_kind": "stochastic_routing_replicate",
        },
        "global_dijkstra": {
            "implementation": "mappo_evaluation.GlobalDijkstraPolicy",
            "policy_seed": GLOBAL_DIJKSTRA_SENTINEL_SEED,
            "replicate_kind": "deterministic_workload_reference",
            "evaluations_per_scenario_workload": 1,
        },
        "execution": {
            "maximum_attempts_per_job": MAX_ATTEMPTS,
            "retry_contract": retry_contract(),
            "maximum_parallel_jobs": MAX_PARALLEL,
            "q_training_jobs": EXPECTED_TRAINING_JOBS,
            "q_training_episodes": EXPECTED_Q_TRAIN_EPISODES,
            "checkpoint_selection_evaluations": 0,
            "gate_evaluation_jobs": EXPECTED_EVALUATION_JOBS,
            "gate_evaluations": EXPECTED_GATE_ROWS,
            "gate_evaluations_by_method": {
                Q_ROUTING: EXPECTED_Q_GATE_ROWS,
                OSPF_ECMP: EXPECTED_OSPF_GATE_ROWS,
                GLOBAL_DIJKSTRA: EXPECTED_DIJKSTRA_GATE_ROWS,
            },
            "sealed_test_evaluations": 0,
        },
        "descriptive_reporting": {
            "pooled_count_audit": "integrity_only_not_an_algorithmic_estimand",
            "policy_identity_equal_delivery": (
                "mean_workload_delivery_ratio_within_identity_then_equal_identity_mean"
            ),
            "policy_identity_equal_decision_rate": (
                "ratio_of_sums_within_identity_then_equal_identity_mean_when_all_defined"
            ),
            "zero_decision_opportunity_rule": (
                "retain_identity_rate_as_null_report_identity_and_set_equal_mean_null_"
                "unless_all_identity_rates_are_defined"
            ),
            "global_dijkstra_identity_count": 1,
        },
        "jobs": {
            "q_training": [job.as_dict() for job in build_training_jobs()],
            "independent_gate": [job.as_dict() for job in build_evaluation_jobs()],
        },
        "paths": {
            "project": str(project_source),
            "classical_protocol": str(
                (project_source.parent / "docs" / PROTOCOL_FILENAME).resolve()
            ),
            "method_protocol": str(
                (project_source.parent / "docs" / METHOD_PROTOCOL_FILENAME).resolve()
            ),
        },
        "code_fingerprint": fingerprint,
    }
    return self_hashed(body, "spec_sha256")


def validate_runtime_environment(args: argparse.Namespace, *, execute: bool) -> None:
    validate_project_source(args.project)
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise ConfigurationError("LEO_REWARD_OVERRIDES must be unset")
    if args.max_parallel not in {1, MAX_PARALLEL}:
        raise ConfigurationError("max_parallel must be one or two")
    if execute:
        expected_python = Path("F:/leo-venv/Scripts/python.exe").resolve()
        if Path(sys.executable).resolve() != expected_python:
            raise ConfigurationError(
                f"classical runner requires {expected_python}, observed {sys.executable}"
            )


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_within(left, right) or _path_within(right, left)


def validate_output_isolation(args: argparse.Namespace) -> None:
    project_source = validate_project_source(args.project)
    output = args.output.resolve()
    repository_root = project_source.parent.resolve()
    experiments_root = (repository_root / "experiments").resolve()
    if output.parent != experiments_root or output.name in {"", ".", ".."}:
        raise ConfigurationError(
            "classical output must be one dedicated child of experiments"
        )
    protected = (
        project_source,
        repository_root / ".git",
        repository_root / "docs",
        experiments_root / "avoidable-switch-constraint-smoke-v1",
        experiments_root / "avoidable-switch-constraint-smoke-v1-r1",
        experiments_root / "avoidable-switch-constraint-formal-v1",
    )
    if any(_paths_overlap(output, path) for path in protected):
        raise ConfigurationError("classical output overlaps protected evidence/runtime")


def validate_new_output_state(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise ConfigurationError("classical output exists and is not a directory")
    if any(output.iterdir()) and not (output / "preregistration.json").is_file():
        raise ConfigurationError(
            "non-empty classical output has no immutable preregistration"
        )


def training_job_root(args: argparse.Namespace, job: QTrainingJob) -> Path:
    return (
        args.output
        / "q_models"
        / job.scenario
        / f"seed_{job.policy_seed}"
    )


def training_status_path(args: argparse.Namespace, job: QTrainingJob) -> Path:
    return args.output / "job_status" / "training" / f"{job.slug}.json"


def evaluation_status_path(args: argparse.Namespace, job: EvaluationJob) -> Path:
    return args.output / "job_status" / "gate" / f"{job.slug}.json"


def evaluation_shard_paths(
    args: argparse.Namespace, job: EvaluationJob
) -> tuple[Path, Path, Path]:
    root = args.output / "gate_shards"
    return (
        root / f"{job.slug}.jsonl",
        root / f"{job.slug}.csv",
        root / f"{job.slug}.manifest.json",
    )


def _tree_as_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tree_as_lists(item) for item in value]
    if isinstance(value, list):
        return [_tree_as_lists(item) for item in value]
    return value


def _tree_as_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tree_as_tuples(item) for item in value)
    return value


def q_array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _model_metadata(
    job: QTrainingJob,
    spec: Mapping[str, Any],
    policy: QRoutingPolicy,
    completed_episodes: int,
    *,
    frozen: bool,
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "source_fingerprint_sha256": sha256_json(spec["code_fingerprint"]),
        "job": job.as_dict(),
        "completed_episodes": completed_episodes,
        "next_episode": (
            completed_episodes if completed_episodes < Q_TRAIN_EPISODES else None
        ),
        "next_workload_seed": (
            q_training_workload_order()[completed_episodes]
            if completed_episodes < Q_TRAIN_EPISODES
            else None
        ),
        "frozen": frozen,
        "training": bool(policy.training),
        "epsilon": float(policy.epsilon),
        "alpha": float(policy.alpha),
        "n_nodes": int(policy.n_nodes),
        "q_dtype": str(policy.q.dtype),
        "q_shape": list(policy.q.shape),
        "q_array_sha256": q_array_sha256(policy.q),
        "policy_rng_state": _tree_as_lists(policy.rng.getstate()),
        "q_contract": q_contract(),
    }
    return self_hashed(body, "model_state_sha256")


def _atomic_write_model(
    path: Path,
    job: QTrainingJob,
    spec: Mapping[str, Any],
    policy: QRoutingPolicy,
    completed_episodes: int,
    *,
    frozen: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _model_metadata(job, spec, policy, completed_episodes, frozen=frozen)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(
                handle,
                q=np.asarray(policy.q, dtype=np.float32),
                metadata=np.asarray(
                    canonical_json_bytes(metadata).decode("utf-8"), dtype=np.str_
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_q_model(
    path: Path,
    job: QTrainingJob,
    spec: Mapping[str, Any],
    *,
    require_frozen: bool | None,
) -> tuple[QRoutingPolicy, dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"q", "metadata"}:
                raise ConfigurationError("Q model archive members drifted")
            q = np.array(archive["q"], copy=True)
            raw_metadata = archive["metadata"]
            if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
                raise ConfigurationError("Q model metadata encoding drifted")
            metadata = json.loads(str(raw_metadata.item()))
    except ConfigurationError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot load Q model: {path}") from error
    if not isinstance(metadata, dict):
        raise ConfigurationError("Q model metadata is not an object")
    validate_self_hash(metadata, "model_state_sha256")
    expected_metadata_fields = {
        "schema_version",
        "study_name",
        "spec_sha256",
        "source_fingerprint_sha256",
        "job",
        "completed_episodes",
        "next_episode",
        "next_workload_seed",
        "frozen",
        "training",
        "epsilon",
        "alpha",
        "n_nodes",
        "q_dtype",
        "q_shape",
        "q_array_sha256",
        "policy_rng_state",
        "q_contract",
        "model_state_sha256",
    }
    if set(metadata) != expected_metadata_fields:
        raise ConfigurationError("Q model metadata schema drifted")
    expected_fixed = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "source_fingerprint_sha256": sha256_json(spec["code_fingerprint"]),
        "job": job.as_dict(),
        "alpha": Q_ALPHA,
        "n_nodes": Q_N_NODES,
        "q_dtype": Q_DTYPE,
        "q_shape": list(Q_SHAPE),
        "q_contract": q_contract(),
    }
    if any(metadata.get(field) != value for field, value in expected_fixed.items()):
        raise ConfigurationError("Q model metadata contract drifted")
    completed = metadata.get("completed_episodes")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ConfigurationError("Q model completed episode count is invalid")
    if not 0 <= completed <= Q_TRAIN_EPISODES:
        raise ConfigurationError("Q model completed episode count is out of range")
    frozen = metadata.get("frozen")
    if not isinstance(frozen, bool) or (
        require_frozen is not None and frozen is not require_frozen
    ):
        raise ConfigurationError("Q model frozen state drifted")
    expected_next = completed if completed < Q_TRAIN_EPISODES else None
    expected_seed = (
        q_training_workload_order()[completed]
        if completed < Q_TRAIN_EPISODES
        else None
    )
    if (
        metadata.get("next_episode") != expected_next
        or metadata.get("next_workload_seed") != expected_seed
    ):
        raise ConfigurationError("Q model resume cursor drifted")
    if q.dtype != np.float32 or q.shape != Q_SHAPE or not np.isfinite(q).all():
        raise ConfigurationError("Q model array schema drifted")
    if metadata.get("q_array_sha256") != q_array_sha256(q):
        raise ConfigurationError("Q model array hash drifted")
    expected_training = not frozen
    expected_epsilon = Q_FROZEN_EPSILON if frozen else Q_TRAIN_EPSILON
    if (
        metadata.get("training") is not expected_training
        or metadata.get("epsilon") != expected_epsilon
        or (frozen and completed != Q_TRAIN_EPISODES)
    ):
        raise ConfigurationError("Q model policy state drifted")
    rng_state = metadata.get("policy_rng_state")
    if not isinstance(rng_state, list):
        raise ConfigurationError("Q model RNG state is missing")
    policy = QRoutingPolicy(
        n_nodes=Q_N_NODES,
        alpha=Q_ALPHA,
        epsilon=Q_TRAIN_EPSILON,
        seed=job.policy_seed,
    )
    policy.q = q
    try:
        policy.rng.setstate(_tree_as_tuples(rng_state))
    except (TypeError, ValueError) as error:
        raise ConfigurationError("Q model RNG state is invalid") from error
    if frozen:
        policy.freeze()
    return policy, metadata


def _atomic_copy(source: Path, destination: Path) -> str:
    source_hash = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise ConfigurationError("immutable resume source differs")
        return source_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != source_hash:
            raise ConfigurationError("resume source copy hash mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def _fresh_q_policy(job: QTrainingJob) -> QRoutingPolicy:
    policy = QRoutingPolicy(
        n_nodes=Q_N_NODES,
        alpha=Q_ALPHA,
        epsilon=Q_TRAIN_EPSILON,
        seed=job.policy_seed,
    )
    if (
        policy.q.dtype != np.float32
        or policy.q.shape != Q_SHAPE
        or not np.all(policy.q == np.float32(Q_INITIAL_VALUE))
    ):
        raise ConfigurationError("Q-routing initialization drifted")
    return policy


def _train_one_episode(
    policy: QRoutingPolicy,
    job: QTrainingJob,
    workload_seed: int,
    wrapper_factory: Callable[..., Any],
) -> None:
    wrapper = wrapper_factory(
        scenario=job.scenario,
        seed=workload_seed,
        variant=ENVIRONMENT_VARIANT,
    )
    try:
        observation, _ = wrapper.reset(seed=workload_seed)
        policy.bind(wrapper)
        terminated = truncated = False
        while not terminated and not truncated:
            actions = policy(observation, wrapper.get_avail_actions())
            observation, _, terminated, truncated, info = wrapper.step(actions)
            policy.observe_transition(info)
    finally:
        wrapper.close()
    policy.pending = []


def _new_status(job: QTrainingJob | EvaluationJob, spec_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec_sha256,
        "job": job.as_dict(),
        "status": "pending",
        "attempts": [],
    }


def _load_status(
    path: Path,
    job: QTrainingJob | EvaluationJob,
    spec_sha256: str,
) -> dict[str, Any]:
    status = read_json(path) if path.is_file() else _new_status(job, spec_sha256)
    if (
        status.get("schema_version") != SCHEMA_VERSION
        or status.get("study_name") != STUDY_NAME
        or status.get("spec_sha256") != spec_sha256
        or status.get("job") != job.as_dict()
        or not isinstance(status.get("attempts"), list)
        or len(status["attempts"]) > MAX_ATTEMPTS
    ):
        raise ConfigurationError(f"job status contract drifted: {job.job_id}")
    allowed_attempt_states = {"running", "failed", "interrupted", "completed"}
    if status.get("status") not in {"pending", *allowed_attempt_states}:
        raise ConfigurationError(f"job status value drifted: {job.job_id}")
    for index, attempt in enumerate(status["attempts"], start=1):
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt") != index
            or attempt.get("status") not in allowed_attempt_states
            or not isinstance(attempt.get("started_at_utc"), str)
        ):
            raise ConfigurationError(f"job attempt history drifted: {job.job_id}")
        if attempt.get("status") != "running" and not isinstance(
            attempt.get("finished_at_utc"), str
        ):
            raise ConfigurationError(f"finished job attempt omits timestamp: {job.job_id}")
        if attempt.get("status") == "running" and index != len(status["attempts"]):
            raise ConfigurationError(f"non-final job attempt is still running: {job.job_id}")
        if attempt.get("status") == "completed" and index != len(status["attempts"]):
            raise ConfigurationError(f"completed job has later attempts: {job.job_id}")
        if attempt.get("status") in {"failed", "interrupted"}:
            failure_category = attempt.get("failure_category")
            retry_authorized = attempt.get("retry_authorized")
            if (
                not isinstance(failure_category, str)
                or not isinstance(retry_authorized, bool)
                or retry_authorized
                is not (failure_category in RETRYABLE_FAILURE_CATEGORIES)
                or not isinstance(attempt.get("error"), str)
            ):
                raise ConfigurationError(
                    f"job failure/retry audit drifted: {job.job_id}"
                )
    if not status["attempts"] and status.get("status") != "pending":
        raise ConfigurationError(f"empty job attempt history is not pending: {job.job_id}")
    if status["attempts"] and status.get("status") != status["attempts"][-1].get(
        "status"
    ):
        raise ConfigurationError(f"job status/last-attempt state disagree: {job.job_id}")
    return status


def _begin_attempt(status: dict[str, Any], path: Path) -> dict[str, Any]:
    if len(status["attempts"]) >= MAX_ATTEMPTS:
        raise ConfigurationError("maximum two attempts already consumed")
    attempt = {
        "attempt": len(status["attempts"]) + 1,
        "status": "running",
        "started_at_utc": utc_now(),
    }
    status["attempts"].append(attempt)
    status["status"] = "running"
    atomic_write_json(path, status)
    return attempt


def _finish_attempt(
    status: dict[str, Any],
    path: Path,
    state: str,
    **updates: Any,
) -> None:
    attempt = status["attempts"][-1]
    if attempt.get("status") != "running":
        raise ConfigurationError("attempt transition is not from running")
    if state not in {"failed", "interrupted", "completed"}:
        raise ConfigurationError("attempt terminal state is invalid")
    if state in {"failed", "interrupted"}:
        failure_category = updates.get("failure_category")
        retry_authorized = updates.get("retry_authorized")
        if (
            not isinstance(failure_category, str)
            or not isinstance(retry_authorized, bool)
            or retry_authorized
            is not (failure_category in RETRYABLE_FAILURE_CATEGORIES)
            or not isinstance(updates.get("error"), str)
        ):
            raise ConfigurationError("failed attempt omits retry classification")
    attempt.update(status=state, finished_at_utc=utc_now(), **updates)
    status["status"] = "completed" if state == "completed" else state
    atomic_write_json(path, status)


def _is_retryable_infrastructure_failure(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return isinstance(error, OSError) and error.errno in RETRYABLE_INFRASTRUCTURE_ERRNOS


def _failure_category(error: BaseException) -> str:
    if isinstance(error, JobCancelled) and error.infrastructure_cause:
        return "recognized_infrastructure_cancellation"
    if _is_retryable_infrastructure_failure(error):
        return "recognized_infrastructure_failure"
    if isinstance(error, ConfigurationError):
        return "configuration_or_audit_failure"
    if isinstance(error, (KeyboardInterrupt, JobCancelled, SystemExit)):
        return "execution_interruption"
    return "scientific_or_runtime_failure"


def _job_cancelled(message: str, cancel_event: threading.Event) -> JobCancelled:
    cause = getattr(cancel_event, "failure_cause", None)
    return JobCancelled(
        message,
        infrastructure_cause=(
            isinstance(cause, BaseException)
            and _is_retryable_infrastructure_failure(cause)
        ),
    )


def _finish_attempt_from_error(
    status: dict[str, Any],
    path: Path,
    error: BaseException,
) -> bool:
    category = _failure_category(error)
    retry_authorized = category in RETRYABLE_FAILURE_CATEGORIES
    _finish_attempt(
        status,
        path,
        (
            "interrupted"
            if isinstance(error, (KeyboardInterrupt, JobCancelled))
            else "failed"
        ),
        error=repr(error),
        failure_category=category,
        retry_authorized=retry_authorized,
    )
    return retry_authorized


def _require_retry_authorization(
    status: Mapping[str, Any], job: QTrainingJob | EvaluationJob
) -> None:
    attempts = status.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return
    previous = attempts[-1]
    if previous.get("status") not in {"failed", "interrupted"}:
        raise ConfigurationError(f"job is not at a retry boundary: {job.job_id}")
    if (
        previous.get("retry_authorized") is not True
        or previous.get("failure_category") not in RETRYABLE_FAILURE_CATEGORIES
    ):
        raise ConfigurationError(
            f"prior job failure is not infrastructure-retry eligible: {job.job_id}"
        )


def _prepare_latest_checkpoint(
    status: dict[str, Any],
    status_path: Path,
    latest: Path,
    job: QTrainingJob,
    spec: Mapping[str, Any],
    policy: QRoutingPolicy,
    completed_episodes: int,
    *,
    frozen: bool,
) -> dict[str, Any]:
    attempt = status["attempts"][-1]
    if attempt.get("status") != "running":
        raise ConfigurationError("checkpoint prepare requires a running attempt")
    target_metadata = _model_metadata(
        job, spec, policy, completed_episodes, frozen=frozen
    )
    previous: dict[str, Any] = {"exists": latest.is_file()}
    if latest.is_file():
        _, previous_metadata = load_q_model(
            latest, job, spec, require_frozen=None
        )
        previous = {
            "exists": True,
            "sha256": sha256_file(latest),
            "completed_episodes": previous_metadata["completed_episodes"],
            "frozen": previous_metadata["frozen"],
            "model_state_sha256": previous_metadata["model_state_sha256"],
        }
        if attempt.get("latest_checkpoint_sha256") not in {
            None,
            previous["sha256"],
        }:
            raise ConfigurationError("attempt latest pointer changed before prepare")
    transaction = {
        "schema_version": 1,
        "transaction_id": uuid.uuid4().hex,
        "state": "prepared",
        "target_path": str(latest.resolve()),
        "target_completed_episodes": completed_episodes,
        "target_frozen": frozen,
        "target_q_array_sha256": target_metadata["q_array_sha256"],
        "target_model_state_sha256": target_metadata["model_state_sha256"],
        "previous": previous,
        "prepared_at_utc": utc_now(),
    }
    attempt["checkpoint_transaction"] = transaction
    atomic_write_json(status_path, status)
    return transaction


def _checkpoint_matches_transaction(
    metadata: Mapping[str, Any], transaction: Mapping[str, Any]
) -> bool:
    return (
        metadata.get("completed_episodes")
        == transaction.get("target_completed_episodes")
        and metadata.get("frozen") is transaction.get("target_frozen")
        and metadata.get("q_array_sha256")
        == transaction.get("target_q_array_sha256")
        and metadata.get("model_state_sha256")
        == transaction.get("target_model_state_sha256")
    )


def _commit_latest_checkpoint(
    status: dict[str, Any],
    status_path: Path,
    latest: Path,
    job: QTrainingJob,
    spec: Mapping[str, Any],
    transaction: dict[str, Any],
    *,
    recovered: bool,
) -> None:
    attempt = status["attempts"][-1]
    if attempt.get("checkpoint_transaction") is not transaction:
        raise ConfigurationError("checkpoint transaction identity drifted")
    _, metadata = load_q_model(latest, job, spec, require_frozen=None)
    if not _checkpoint_matches_transaction(metadata, transaction):
        raise ConfigurationError("written checkpoint differs from prepared state")
    checkpoint_hash = sha256_file(latest)
    transaction.update(
        state=("recovered_committed" if recovered else "committed"),
        checkpoint_sha256=checkpoint_hash,
        committed_at_utc=utc_now(),
    )
    attempt.update(
        latest_checkpoint=str(latest.resolve()),
        latest_checkpoint_sha256=checkpoint_hash,
        latest_checkpoint_model_state_sha256=metadata["model_state_sha256"],
        completed_episodes=int(metadata["completed_episodes"]),
        latest_checkpoint_frozen=bool(metadata["frozen"]),
    )
    atomic_write_json(status_path, status)


def _write_latest_checkpoint(
    status: dict[str, Any],
    status_path: Path,
    latest: Path,
    job: QTrainingJob,
    spec: Mapping[str, Any],
    policy: QRoutingPolicy,
    completed_episodes: int,
    *,
    frozen: bool,
) -> None:
    transaction = _prepare_latest_checkpoint(
        status,
        status_path,
        latest,
        job,
        spec,
        policy,
        completed_episodes,
        frozen=frozen,
    )
    _atomic_write_model(
        latest,
        job,
        spec,
        policy,
        completed_episodes,
        frozen=frozen,
    )
    _commit_latest_checkpoint(
        status,
        status_path,
        latest,
        job,
        spec,
        transaction,
        recovered=False,
    )


def _reconcile_latest_checkpoint(
    status: dict[str, Any],
    status_path: Path,
    latest: Path,
    job: QTrainingJob,
    spec: Mapping[str, Any],
) -> None:
    if not status["attempts"]:
        if latest.exists():
            raise ConfigurationError("orphan Q checkpoint has no attempt transaction")
        return
    attempt = status["attempts"][-1]
    transaction = attempt.get("checkpoint_transaction")
    if transaction is None:
        if latest.exists():
            expected_hash = attempt.get("latest_checkpoint_sha256")
            if expected_hash != sha256_file(latest):
                raise ConfigurationError("Q checkpoint is not bound by job status")
            load_q_model(latest, job, spec, require_frozen=None)
        return
    if not isinstance(transaction, dict):
        raise ConfigurationError("checkpoint transaction is malformed")
    if (
        transaction.get("schema_version") != 1
        or transaction.get("target_path") != str(latest.resolve())
        or transaction.get("state")
        not in {"prepared", "committed", "recovered_committed", "not_applied"}
    ):
        raise ConfigurationError("checkpoint transaction contract drifted")
    state = transaction["state"]
    if state == "prepared":
        previous = transaction.get("previous")
        if not isinstance(previous, Mapping) or not isinstance(
            previous.get("exists"), bool
        ):
            raise ConfigurationError("checkpoint transaction previous state drifted")
        if latest.is_file():
            _, metadata = load_q_model(latest, job, spec, require_frozen=None)
            if _checkpoint_matches_transaction(metadata, transaction):
                _commit_latest_checkpoint(
                    status,
                    status_path,
                    latest,
                    job,
                    spec,
                    transaction,
                    recovered=True,
                )
                return
            if (
                previous["exists"]
                and previous.get("sha256") == sha256_file(latest)
                and previous.get("model_state_sha256")
                == metadata.get("model_state_sha256")
            ):
                transaction.update(state="not_applied", reconciled_at_utc=utc_now())
                attempt.update(
                    latest_checkpoint=str(latest.resolve()),
                    latest_checkpoint_sha256=previous["sha256"],
                    latest_checkpoint_model_state_sha256=metadata[
                        "model_state_sha256"
                    ],
                    completed_episodes=int(metadata["completed_episodes"]),
                    latest_checkpoint_frozen=bool(metadata["frozen"]),
                )
                atomic_write_json(status_path, status)
                return
            raise ConfigurationError("prepared checkpoint transaction is ambiguous")
        if previous["exists"]:
            raise ConfigurationError("prepared transaction lost its prior checkpoint")
        transaction.update(state="not_applied", reconciled_at_utc=utc_now())
        atomic_write_json(status_path, status)
        return
    if state in {"committed", "recovered_committed"}:
        if not latest.is_file() or transaction.get("checkpoint_sha256") != sha256_file(
            latest
        ):
            raise ConfigurationError("committed checkpoint transaction changed")
        _, metadata = load_q_model(latest, job, spec, require_frozen=None)
        if not _checkpoint_matches_transaction(metadata, transaction):
            raise ConfigurationError("committed checkpoint transaction state drifted")
        if (
            attempt.get("latest_checkpoint_sha256") != sha256_file(latest)
            or attempt.get("completed_episodes")
            != metadata["completed_episodes"]
            or attempt.get("latest_checkpoint_frozen") is not metadata["frozen"]
        ):
            raise ConfigurationError("committed checkpoint/status pointer drifted")
        return
    previous = transaction.get("previous")
    if not isinstance(previous, Mapping):
        raise ConfigurationError("not-applied checkpoint transaction lacks prior state")
    if previous.get("exists"):
        if not latest.is_file() or previous.get("sha256") != sha256_file(latest):
            raise ConfigurationError("not-applied transaction prior checkpoint changed")
        load_q_model(latest, job, spec, require_frozen=None)
    elif latest.exists():
        raise ConfigurationError("not-applied transaction unexpectedly has checkpoint")


def _recover_training_status(
    status: dict[str, Any],
    status_path: Path,
    latest: Path,
    job: QTrainingJob,
    spec: Mapping[str, Any],
) -> None:
    _reconcile_latest_checkpoint(status, status_path, latest, job, spec)
    if status["attempts"] and status["attempts"][-1].get("status") == "running":
        _finish_attempt(
            status,
            status_path,
            "interrupted",
            error="orphaned running attempt recovered at a checkpoint transaction boundary",
            failure_category="orphaned_process_interruption",
            retry_authorized=True,
        )


def _training_manifest_path(args: argparse.Namespace, job: QTrainingJob) -> Path:
    return training_job_root(args, job) / "model_manifest.json"


def _completed_status(
    path: Path,
    job: QTrainingJob | EvaluationJob,
    spec_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"completed job status is missing: {job.job_id}")
    status = _load_status(path, job, spec_sha256)
    if (
        status.get("status") != "completed"
        or not status["attempts"]
        or status["attempts"][-1].get("status") != "completed"
        or len(status["attempts"]) > MAX_ATTEMPTS
    ):
        raise ConfigurationError(f"completed job status is inconsistent: {job.job_id}")
    return status


def validate_training_manifest(
    args: argparse.Namespace,
    job: QTrainingJob,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    path = _training_manifest_path(args, job)
    if not path.is_file():
        raise ConfigurationError(f"Q model manifest is missing: {job.job_id}")
    manifest = read_json(path)
    validate_self_hash(manifest, "manifest_sha256")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "job": job.as_dict(),
        "training_episodes": Q_TRAIN_EPISODES,
        "training_workload_order": list(q_training_workload_order()),
        "training_workload_order_sha256": q_contract()["workload_order_sha256"],
        "q_contract": q_contract(),
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ConfigurationError(f"Q model manifest drifted: {job.job_id}")
    artifact = manifest.get("model")
    if not isinstance(artifact, Mapping):
        raise ConfigurationError("Q model manifest omits model artifact")
    model_path = Path(str(artifact.get("path", ""))).resolve()
    root = training_job_root(args, job).resolve()
    if (
        model_path != root / "model.npz"
        or not _path_within(model_path, root)
        or not model_path.is_file()
        or artifact.get("sha256") != sha256_file(model_path)
    ):
        raise ConfigurationError("Q model artifact path/hash drifted")
    _, metadata = load_q_model(
        model_path, job, spec, require_frozen=True
    )
    if (
        artifact.get("q_array_sha256") != metadata["q_array_sha256"]
        or manifest.get("policy_rng_seed") != job.policy_seed
        or manifest.get("model_dtype") != Q_DTYPE
        or manifest.get("model_shape") != list(Q_SHAPE)
    ):
        raise ConfigurationError("Q model manifest array/RNG schema drifted")
    status_path = training_status_path(args, job).resolve()
    status_artifact = manifest.get("status")
    if (
        not isinstance(status_artifact, Mapping)
        or Path(str(status_artifact.get("path", ""))).resolve() != status_path
        or not status_path.is_file()
        or status_artifact.get("sha256") != sha256_file(status_path)
    ):
        raise ConfigurationError("Q model status artifact drifted")
    status = _completed_status(status_path, job, spec["spec_sha256"])
    final_attempt = status["attempts"][-1]
    latest_path = root / "latest.npz"
    if (
        final_attempt.get("latest_checkpoint") != str(latest_path)
        or not latest_path.is_file()
        or final_attempt.get("latest_checkpoint_sha256") != sha256_file(latest_path)
        or final_attempt.get("latest_checkpoint_frozen") is not True
    ):
        raise ConfigurationError("Q final status latest-checkpoint pointer drifted")
    _, latest_metadata = load_q_model(
        latest_path, job, spec, require_frozen=True
    )
    if latest_metadata["model_state_sha256"] != metadata["model_state_sha256"]:
        raise ConfigurationError("Q final/latest model states disagree")
    if (
        manifest.get("implementation") != "mappo_evaluation.QRoutingPolicy"
        or manifest.get("implementation_source_sha256")
        != Q_ROUTING_CLASS_SOURCE_SHA256
        or isinstance(manifest.get("attempt_count"), bool)
        or not isinstance(manifest.get("attempt_count"), int)
        or not 1 <= manifest["attempt_count"] <= MAX_ATTEMPTS
        or manifest["attempt_count"] != len(status["attempts"])
        or final_attempt.get("completed_episodes") != Q_TRAIN_EPISODES
        or final_attempt.get("model_sha256") != artifact.get("sha256")
    ):
        raise ConfigurationError("Q model implementation/attempt audit drifted")
    resume_sources = manifest.get("resume_sources")
    if not isinstance(resume_sources, list):
        raise ConfigurationError("Q model manifest omits resume-source inventory")
    seen_attempts: set[int] = set()
    for entry in resume_sources:
        if not isinstance(entry, Mapping):
            raise ConfigurationError("Q model resume-source entry is malformed")
        attempt_number = entry.get("attempt")
        source = Path(str(entry.get("path", ""))).resolve()
        if (
            isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or not 2 <= attempt_number <= manifest["attempt_count"]
            or attempt_number in seen_attempts
            or source != root / f"resume_source_attempt_{attempt_number}.npz"
            or not _path_within(source, root)
            or not source.is_file()
            or entry.get("sha256") != sha256_file(source)
        ):
            raise ConfigurationError("Q model resume-source audit drifted")
        load_q_model(source, job, spec, require_frozen=None)
        seen_attempts.add(attempt_number)
    expected_resume_sources = [
        {
            "attempt": attempt["attempt"],
            "path": attempt["resume_source"],
            "sha256": attempt["resume_source_sha256"],
        }
        for attempt in status["attempts"]
        if "resume_source" in attempt
    ]
    if resume_sources != expected_resume_sources:
        raise ConfigurationError("Q model resume inventory/status disagree")
    for attempt in status["attempts"]:
        transaction = attempt.get("checkpoint_transaction")
        if transaction is not None and (
            not isinstance(transaction, Mapping)
            or transaction.get("state")
            not in {"committed", "recovered_committed", "not_applied"}
        ):
            raise ConfigurationError("Q completed status has unresolved transaction")
    final_transaction = final_attempt.get("checkpoint_transaction")
    if (
        not isinstance(final_transaction, Mapping)
        or final_transaction.get("state")
        not in {"committed", "recovered_committed"}
        or final_transaction.get("target_completed_episodes") != Q_TRAIN_EPISODES
        or final_transaction.get("target_frozen") is not True
        or final_transaction.get("checkpoint_sha256") != sha256_file(latest_path)
    ):
        raise ConfigurationError("Q final checkpoint transaction drifted")
    return manifest


def _write_training_manifest(
    args: argparse.Namespace,
    job: QTrainingJob,
    spec: Mapping[str, Any],
    model_path: Path,
    metadata: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    status_path = training_status_path(args, job).resolve()
    completed_status = _completed_status(status_path, job, spec["spec_sha256"])
    if dict(status) != completed_status:
        raise ConfigurationError("in-memory and persisted Q status disagree")
    resume_sources = []
    for attempt in completed_status["attempts"]:
        if "resume_source" in attempt:
            resume_sources.append(
                {
                    "attempt": attempt["attempt"],
                    "path": attempt["resume_source"],
                    "sha256": attempt["resume_source_sha256"],
                }
            )
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "job": job.as_dict(),
        "implementation": "mappo_evaluation.QRoutingPolicy",
        "implementation_source_sha256": Q_ROUTING_CLASS_SOURCE_SHA256,
        "training_episodes": Q_TRAIN_EPISODES,
        "training_workload_order": list(q_training_workload_order()),
        "training_workload_order_sha256": q_contract()["workload_order_sha256"],
        "q_contract": q_contract(),
        "policy_rng_seed": job.policy_seed,
        "model_dtype": Q_DTYPE,
        "model_shape": list(Q_SHAPE),
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "q_array_sha256": metadata["q_array_sha256"],
        },
        "status": {
            "path": str(status_path),
            "sha256": sha256_file(status_path),
        },
        "attempt_count": len(completed_status["attempts"]),
        "resume_sources": resume_sources,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "completed_at_utc": utc_now(),
    }
    manifest = self_hashed(body, "manifest_sha256")
    ensure_immutable_json(_training_manifest_path(args, job), manifest)
    return validate_training_manifest(args, job, spec)


def train_q_job(
    args: argparse.Namespace,
    job: QTrainingJob,
    spec: Mapping[str, Any],
    cancel_event: threading.Event,
    *,
    wrapper_factory: Callable[..., Any] = CleanMARLLeoMultiAgentWrapper,
) -> dict[str, Any]:
    manifest_path = _training_manifest_path(args, job)
    if manifest_path.is_file():
        return validate_training_manifest(args, job, spec)
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    root = training_job_root(args, job)
    root.mkdir(parents=True, exist_ok=True)
    status_path = training_status_path(args, job)
    status = _load_status(status_path, job, spec["spec_sha256"])
    latest = root / "latest.npz"
    final_model = root / "model.npz"
    _recover_training_status(status, status_path, latest, job, spec)
    if status.get("status") == "completed":
        if not final_model.is_file():
            raise ConfigurationError("completed Q status has no final model artifact")
        _, final_metadata = load_q_model(
            final_model, job, spec, require_frozen=True
        )
        if status["attempts"][-1].get("model_sha256") != sha256_file(final_model):
            raise ConfigurationError("completed Q status final-model hash drifted")
        return _write_training_manifest(
            args, job, spec, final_model, final_metadata, status
        )
    while len(status["attempts"]) < MAX_ATTEMPTS:
        _reconcile_latest_checkpoint(status, status_path, latest, job, spec)
        if cancel_event.is_set():
            raise _job_cancelled(
                f"training cancelled before attempt: {job.job_id}", cancel_event
            )
        _require_retry_authorization(status, job)
        attempt = _begin_attempt(status, status_path)
        try:
            if latest.is_file():
                if attempt["attempt"] == 1:
                    raise ConfigurationError(
                        "orphan Q checkpoint exists without a prior attempt"
                    )
                prior_attempt = status["attempts"][-2]
                if (
                    prior_attempt.get("latest_checkpoint")
                    != str(latest.resolve())
                    or prior_attempt.get("latest_checkpoint_sha256")
                    != sha256_file(latest)
                ):
                    raise ConfigurationError(
                        "latest Q checkpoint is not bound by the prior attempt"
                    )
                resume_source = root / f"resume_source_attempt_{attempt['attempt']}.npz"
                resume_hash = _atomic_copy(latest, resume_source)
                policy, metadata = load_q_model(
                    resume_source, job, spec, require_frozen=None
                )
                completed = int(metadata["completed_episodes"])
                attempt.update(
                    resume_source=str(resume_source.resolve()),
                    resume_source_sha256=resume_hash,
                    resumed_from_episode=completed,
                )
                atomic_write_json(status_path, status)
            else:
                if final_model.exists():
                    raise ConfigurationError("orphan final Q model has no latest checkpoint")
                policy = _fresh_q_policy(job)
                completed = 0
                attempt["resumed_from_episode"] = 0
                atomic_write_json(status_path, status)
            if completed == Q_TRAIN_EPISODES:
                if not policy.training:
                    pass
                else:
                    policy.freeze()
            else:
                if not policy.training or policy.epsilon != Q_TRAIN_EPSILON:
                    raise ConfigurationError("resumed Q policy is not in training state")
                order = q_training_workload_order()
                for episode in range(completed, Q_TRAIN_EPISODES):
                    if cancel_event.is_set():
                        raise _job_cancelled(
                            f"training cancelled: {job.job_id}", cancel_event
                        )
                    _train_one_episode(policy, job, order[episode], wrapper_factory)
                    completed = episode + 1
                    _write_latest_checkpoint(
                        status,
                        status_path,
                        latest,
                        job,
                        spec,
                        policy,
                        completed,
                        frozen=False,
                    )
                policy.freeze()
            _write_latest_checkpoint(
                status,
                status_path,
                latest,
                job,
                spec,
                policy,
                Q_TRAIN_EPISODES,
                frozen=True,
            )
            _atomic_write_model(
                final_model,
                job,
                spec,
                policy,
                Q_TRAIN_EPISODES,
                frozen=True,
            )
            _, final_metadata = load_q_model(
                final_model, job, spec, require_frozen=True
            )
            _finish_attempt(
                status,
                status_path,
                "completed",
                completed_episodes=Q_TRAIN_EPISODES,
                model_sha256=sha256_file(final_model),
            )
            return _write_training_manifest(
                args, job, spec, final_model, final_metadata, status
            )
        except BaseException as error:
            if status["attempts"][-1].get("status") != "running":
                raise
            retry_authorized = _finish_attempt_from_error(
                status, status_path, error
            )
            if isinstance(error, (KeyboardInterrupt, JobCancelled)):
                raise
            if not retry_authorized:
                raise
            if len(status["attempts"]) >= MAX_ATTEMPTS:
                raise
            _reconcile_latest_checkpoint(status, status_path, latest, job, spec)
    raise ConfigurationError(f"maximum attempts consumed: {job.job_id}")


def _run_parallel_fail_fast(
    jobs: Sequence[Any],
    max_parallel: int,
    worker: Callable[[Any, threading.Event], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    cancel_event = threading.Event()
    completed: dict[str, dict[str, Any]] = {}
    iterator = iter(jobs)
    futures: dict[Any, Any] = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        for _ in range(min(max_parallel, len(jobs))):
            job = next(iterator, None)
            if job is not None:
                futures[executor.submit(worker, job, cancel_event)] = job
        failure: BaseException | None = None
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    completed[job.job_id] = future.result()
                except BaseException as error:
                    if failure is None:
                        failure = error
                        setattr(cancel_event, "failure_cause", error)
                    cancel_event.set()
            if failure is None:
                for _ in range(len(done)):
                    job = next(iterator, None)
                    if job is not None:
                        futures[executor.submit(worker, job, cancel_event)] = job
            else:
                for future in futures:
                    future.cancel()
        if failure is not None:
            raise failure
    return {job.job_id: completed[job.job_id] for job in jobs}


def run_all_training_jobs(
    args: argparse.Namespace, spec: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    jobs = build_training_jobs()
    return _run_parallel_fail_fast(
        jobs,
        args.max_parallel,
        lambda job, cancel: train_q_job(args, job, spec, cancel),
    )


def validate_training_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = args.output / "training_freeze.json"
    observed = dict(freeze) if freeze is not None else read_json(path)
    validate_self_hash(observed, "freeze_sha256")
    if (
        observed.get("schema_version") != SCHEMA_VERSION
        or observed.get("study_name") != STUDY_NAME
        or observed.get("spec_sha256") != spec["spec_sha256"]
        or observed.get("training_complete") is not True
        or observed.get("training_job_count") != EXPECTED_TRAINING_JOBS
        or observed.get("total_training_episodes") != EXPECTED_Q_TRAIN_EPISODES
        or observed.get("test_panel_consulted") is not False
        or observed.get("test_access_count") != 0
        or observed.get("sealed_test_instantiated") is not False
    ):
        raise ConfigurationError("classical training freeze contract drifted")
    jobs = observed.get("jobs")
    expected_job_ids = [job.job_id for job in build_training_jobs()]
    if not isinstance(jobs, Mapping) or sorted(jobs) != sorted(expected_job_ids):
        raise ConfigurationError("classical training freeze grid drifted")
    for job in build_training_jobs():
        manifest = validate_training_manifest(args, job, spec)
        entry = jobs[job.job_id]
        manifest_path = _training_manifest_path(args, job)
        if entry != {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "model": manifest["model"],
            "status": manifest["status"],
            "attempt_count": manifest["attempt_count"],
        }:
            raise ConfigurationError(f"training freeze entry drifted: {job.job_id}")
    return observed


def write_training_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    path = args.output / "training_freeze.json"
    if path.is_file():
        return validate_training_freeze(args, spec)
    expected_ids = [job.job_id for job in build_training_jobs()]
    if list(manifests) != expected_ids:
        raise ConfigurationError("cannot freeze partial/reordered Q model grid")
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_complete": True,
        "training_job_count": EXPECTED_TRAINING_JOBS,
        "total_training_episodes": EXPECTED_Q_TRAIN_EPISODES,
        "jobs": {
            job.job_id: {
                "manifest_path": str(_training_manifest_path(args, job).resolve()),
                "manifest_file_sha256": sha256_file(_training_manifest_path(args, job)),
                "manifest_sha256": manifests[job.job_id]["manifest_sha256"],
                "model": manifests[job.job_id]["model"],
                "status": manifests[job.job_id]["status"],
                "attempt_count": manifests[job.job_id]["attempt_count"],
            }
            for job in build_training_jobs()
        },
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "completed_at_utc": utc_now(),
    }
    freeze = self_hashed(body, "freeze_sha256")
    ensure_immutable_json(path, freeze)
    return validate_training_freeze(args, spec)


BASE_METRIC_FIELDS = tuple(field.name for field in fields(ConstraintEpisodeMetrics))
GATE_ROW_FIELDS = (
    "schema_version",
    "study_name",
    "evaluation_role",
    "spec_sha256",
    "training_freeze_sha256",
    "method",
    "replicate_kind",
    "environment_variant",
    "source_model_sha256",
    "decision_forced_switch_cost",
    *BASE_METRIC_FIELDS,
)
INTEGER_METRIC_FIELDS = {
    "schema_version",
    "workload_seed",
    "generated",
    "delivered",
    "dropped",
    "backlog",
    "max_queue_packets",
    "routing_switches",
    "avoidable_routing_switches",
    "forced_routing_switches",
    "switch_opportunities",
    "decision_avoidable_switches",
    "decision_switch_opportunities",
    "decision_forced_switches",
    "decision_forced_switch_cost",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ConfigurationError(f"gate {field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"gate {field} is not finite")
    return result


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConfigurationError(f"gate {field} is not an integer")
    result = int(value)
    if result < 0:
        raise ConfigurationError(f"gate {field} is negative")
    return result


def _close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _q_training_job_for(job: EvaluationJob) -> QTrainingJob:
    matches = [
        candidate
        for candidate in build_training_jobs()
        if candidate.scenario == job.scenario
        and candidate.policy_seed == job.policy_seed
    ]
    if job.method != Q_ROUTING or len(matches) != 1:
        raise ConfigurationError("Q evaluation has no unique frozen model")
    return matches[0]


def _source_model_hash(
    job: EvaluationJob, training_freeze: Mapping[str, Any]
) -> str | None:
    if job.method != Q_ROUTING:
        return None
    training_job = _q_training_job_for(job)
    return str(training_freeze["jobs"][training_job.job_id]["model"]["sha256"])


def validate_gate_rows(
    job: EvaluationJob,
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != len(GATE_WORKLOAD_SEEDS):
        raise ConfigurationError(f"gate shard row count drifted: {job.job_id}")
    expected_model_hash = _source_model_hash(job, training_freeze)
    decision_cost = 0
    decision_opportunity = 0
    generated_total = 0
    delivered_total = 0
    for index, row in enumerate(rows):
        if len(row) != len(GATE_ROW_FIELDS) or set(row) != set(GATE_ROW_FIELDS):
            raise ConfigurationError("gate row schema drifted")
        expected_metadata = {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "evaluation_role": "classical_independent_gate_descriptive",
            "spec_sha256": spec["spec_sha256"],
            "training_freeze_sha256": training_freeze["freeze_sha256"],
            "method": job.method,
            "replicate_kind": job.replicate_kind,
            "environment_variant": ENVIRONMENT_VARIANT,
            "source_model_sha256": expected_model_hash,
            "decision_forced_switch_cost": 0,
            "scenario": job.scenario,
            "policy": job.method,
            "policy_seed": job.policy_seed,
            "workload_seed": GATE_WORKLOAD_SEEDS[index],
        }
        if any(row.get(field) != value for field, value in expected_metadata.items()):
            raise ConfigurationError(f"gate row identity drifted: {job.job_id}")
        if isinstance(row["policy_seed"], bool) or not isinstance(
            row["policy_seed"], (int, np.integer)
        ):
            raise ConfigurationError("gate policy seed is not an integer")
        for field in INTEGER_METRIC_FIELDS:
            _nonnegative_integer(row[field], field)
        for field in GATE_ROW_FIELDS:
            if field in INTEGER_METRIC_FIELDS or field in {
                "study_name",
                "evaluation_role",
                "spec_sha256",
                "training_freeze_sha256",
                "method",
                "replicate_kind",
                "environment_variant",
                "source_model_sha256",
                "scenario",
                "policy",
            }:
                continue
            _finite_number(row[field], field)
        generated = int(row["generated"])
        delivered = int(row["delivered"])
        dropped = int(row["dropped"])
        backlog = int(row["backlog"])
        if generated != delivered + dropped + backlog:
            raise ConfigurationError("gate packet accounting drifted")
        if not _close(row["delivery_ratio"], delivered / max(1, generated)):
            raise ConfigurationError("gate delivery ratio drifted")
        if not _close(row["drop_rate"], dropped / max(1, generated)):
            raise ConfigurationError("gate drop rate drifted")
        accepted_avoidable = int(row["avoidable_routing_switches"])
        accepted_forced = int(row["forced_routing_switches"])
        accepted_opportunity = int(row["switch_opportunities"])
        if (
            int(row["routing_switches"]) != accepted_avoidable + accepted_forced
            or accepted_avoidable > accepted_opportunity
            or not _close(
                row["avoidable_switch_rate"],
                accepted_avoidable / max(1, accepted_opportunity),
            )
        ):
            raise ConfigurationError("gate accepted-switch ledger drifted")
        cost = int(row["decision_avoidable_switches"])
        opportunity = int(row["decision_switch_opportunities"])
        forced = int(row["decision_forced_switches"])
        if (
            cost > opportunity
            or accepted_avoidable > cost
            or accepted_opportunity > opportunity
            or accepted_forced > forced
            or not _close(
                row["decision_avoidable_switch_rate"],
                cost / max(1, opportunity),
            )
        ):
            raise ConfigurationError("gate decision-switch ledger drifted")
        if not _close(row["global_switch_cost"], 0.0):
            raise ConfigurationError("qos_only gate row contains switch reward cost")
        for field in (
            "throughput_packets_per_slot",
            "average_delay_slots",
            "p95_delay_slots",
            "mean_queue_packets",
            "global_delay_cost",
            "global_queue_cost",
            "global_load_imbalance",
            "global_drop_cost",
        ):
            if float(row[field]) < 0.0:
                raise ConfigurationError(f"gate metric is negative: {field}")
        if float(row["mean_queue_packets"]) > int(row["max_queue_packets"]):
            raise ConfigurationError("gate mean queue exceeds maximum queue")
        if not 0.0 <= float(row["global_control_overhead_ratio"]) <= 1.0:
            raise ConfigurationError("gate control-overhead ratio is outside [0,1]")
        for field in (
            "delivery_ratio",
            "drop_rate",
            "class_0_delivery_ratio",
            "class_1_delivery_ratio",
            "class_2_delivery_ratio",
            "avoidable_switch_rate",
            "decision_avoidable_switch_rate",
        ):
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ConfigurationError(f"gate ratio is outside [0,1]: {field}")
        decision_cost += cost
        decision_opportunity += opportunity
        generated_total += generated
        delivered_total += delivered
    return {
        "row_count": len(rows),
        "generated": generated_total,
        "delivered": delivered_total,
        "delivery_ratio": delivered_total / max(1, generated_total),
        "decision_avoidable_switches": decision_cost,
        "decision_switch_opportunities": decision_opportunity,
        "decision_avoidable_switch_rate": (
            decision_cost / decision_opportunity
            if decision_opportunity > 0
            else None
        ),
    }


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)).decode("utf-8"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ConfigurationError("gate JSONL row is not an object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read gate JSONL: {path}") from error
    return rows


def _read_gate_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(GATE_ROW_FIELDS):
                raise ConfigurationError("gate CSV header drifted")
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ConfigurationError(f"cannot read gate CSV: {path}") from error
    string_fields = {
        "study_name",
        "evaluation_role",
        "spec_sha256",
        "training_freeze_sha256",
        "method",
        "replicate_kind",
        "environment_variant",
        "scenario",
        "policy",
    }
    rows: list[dict[str, Any]] = []
    try:
        for raw in raw_rows:
            row: dict[str, Any] = {}
            for field in GATE_ROW_FIELDS:
                value = raw[field]
                if field in string_fields:
                    row[field] = value
                elif field == "source_model_sha256":
                    row[field] = value or None
                elif field in INTEGER_METRIC_FIELDS or field == "policy_seed":
                    row[field] = int(value)
                else:
                    row[field] = float(value)
            rows.append(row)
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("gate CSV value schema drifted") from error
    return rows


def _make_evaluation_policy(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
) -> Any:
    if job.method == Q_ROUTING:
        training_job = _q_training_job_for(job)
        entry = training_freeze["jobs"][training_job.job_id]["model"]
        path = Path(entry["path"]).resolve()
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ConfigurationError("frozen Q model changed before gate evaluation")
        policy, _ = load_q_model(path, training_job, spec, require_frozen=True)
        return policy
    if job.method == OSPF_ECMP:
        return OspfEcmpPolicy(seed=job.policy_seed)
    if job.method == GLOBAL_DIJKSTRA:
        if job.policy_seed != GLOBAL_DIJKSTRA_SENTINEL_SEED:
            raise ConfigurationError("Global Dijkstra sentinel seed drifted")
        return GlobalDijkstraPolicy()
    raise ConfigurationError(f"unknown classical method: {job.method}")


def validate_evaluation_shard(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jsonl_path, csv_path, manifest_path = evaluation_shard_paths(args, job)
    if not manifest_path.is_file():
        raise ConfigurationError(f"gate shard manifest is missing: {job.job_id}")
    manifest = read_json(manifest_path)
    validate_self_hash(manifest, "manifest_sha256")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "job": job.as_dict(),
        "evaluation_role": "classical_independent_gate_descriptive",
        "environment_variant": ENVIRONMENT_VARIANT,
        "gate_workload_seeds": list(GATE_WORKLOAD_SEEDS),
        "checkpoint_selection_evaluations": 0,
        "implementation_source_sha256": METHOD_SOURCE_SHA256[job.method],
        "source_model_sha256": _source_model_hash(job, training_freeze),
        "row_fields": list(GATE_ROW_FIELDS),
        "row_count": len(GATE_WORKLOAD_SEEDS),
        "performance_gate_applied": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ConfigurationError(f"gate shard manifest drifted: {job.job_id}")
    if (
        manifest.get("jsonl_path") != str(jsonl_path.resolve())
        or manifest.get("csv_path") != str(csv_path.resolve())
    ):
        raise ConfigurationError(f"gate shard artifact path drifted: {job.job_id}")
    for artifact_path, field in (
        (jsonl_path, "jsonl_sha256"),
        (csv_path, "csv_sha256"),
    ):
        if not artifact_path.is_file() or manifest.get(field) != sha256_file(artifact_path):
            raise ConfigurationError(f"gate shard artifact drifted: {job.job_id}")
    rows = _read_jsonl(jsonl_path)
    csv_rows = _read_gate_csv(csv_path)
    if csv_rows != rows:
        raise ConfigurationError(f"gate JSONL/CSV disagree: {job.job_id}")
    summary = validate_gate_rows(job, rows, spec, training_freeze)
    if manifest.get("summary") != summary:
        raise ConfigurationError(f"gate shard summary drifted: {job.job_id}")
    status_path = evaluation_status_path(args, job).resolve()
    status_artifact = manifest.get("status")
    if (
        not isinstance(status_artifact, Mapping)
        or Path(str(status_artifact.get("path", ""))).resolve() != status_path
        or not status_path.is_file()
        or status_artifact.get("sha256") != sha256_file(status_path)
    ):
        raise ConfigurationError(f"gate status artifact drifted: {job.job_id}")
    status = _completed_status(status_path, job, spec["spec_sha256"])
    final_attempt = status["attempts"][-1]
    if (
        manifest.get("attempt_count") != len(status["attempts"])
        or final_attempt.get("row_count") != len(rows)
        or final_attempt.get("jsonl_sha256") != manifest.get("jsonl_sha256")
        or final_attempt.get("csv_sha256") != manifest.get("csv_sha256")
    ):
        raise ConfigurationError(f"gate completion status drifted: {job.job_id}")
    return manifest, rows


def _finalize_evaluation_manifest(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    status: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    jsonl_path, csv_path, manifest_path = evaluation_shard_paths(args, job)
    status_path = evaluation_status_path(args, job).resolve()
    completed_status = _completed_status(status_path, job, spec["spec_sha256"])
    if dict(status) != completed_status:
        raise ConfigurationError("in-memory and persisted gate status disagree")
    if not jsonl_path.is_file() or not csv_path.is_file():
        raise ConfigurationError("completed gate status has missing row artifacts")
    materialized = [dict(row) for row in rows]
    if _read_jsonl(jsonl_path) != materialized or _read_gate_csv(csv_path) != materialized:
        raise ConfigurationError("completed gate row artifacts disagree")
    summary = validate_gate_rows(job, materialized, spec, training_freeze)
    jsonl_hash = sha256_file(jsonl_path)
    csv_hash = sha256_file(csv_path)
    final_attempt = completed_status["attempts"][-1]
    if (
        final_attempt.get("row_count") != len(materialized)
        or final_attempt.get("jsonl_sha256") != jsonl_hash
        or final_attempt.get("csv_sha256") != csv_hash
    ):
        raise ConfigurationError("completed gate status does not bind row artifacts")
    source_model_hash = _source_model_hash(job, training_freeze)
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "job": job.as_dict(),
        "evaluation_role": "classical_independent_gate_descriptive",
        "environment_variant": ENVIRONMENT_VARIANT,
        "gate_workload_seeds": list(GATE_WORKLOAD_SEEDS),
        "checkpoint_selection_evaluations": 0,
        "implementation_source_sha256": METHOD_SOURCE_SHA256[job.method],
        "source_model_sha256": source_model_hash,
        "row_fields": list(GATE_ROW_FIELDS),
        "row_count": len(materialized),
        "summary": summary,
        "jsonl_path": str(jsonl_path.resolve()),
        "jsonl_sha256": jsonl_hash,
        "csv_path": str(csv_path.resolve()),
        "csv_sha256": csv_hash,
        "status": {
            "path": str(status_path),
            "sha256": sha256_file(status_path),
        },
        "attempt_count": len(completed_status["attempts"]),
        "performance_gate_applied": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "completed_at_utc": final_attempt["finished_at_utc"],
    }
    manifest = self_hashed(body, "manifest_sha256")
    ensure_immutable_json(manifest_path, manifest)
    return validate_evaluation_shard(args, job, spec, training_freeze)[0]


def evaluate_gate_job(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    cancel_event: threading.Event,
) -> dict[str, Any]:
    jsonl_path, csv_path, manifest_path = evaluation_shard_paths(args, job)
    if manifest_path.is_file():
        return validate_evaluation_shard(args, job, spec, training_freeze)[0]
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    status_path = evaluation_status_path(args, job)
    status = _load_status(status_path, job, spec["spec_sha256"])
    if status.get("status") == "completed":
        jsonl_path, csv_path, _ = evaluation_shard_paths(args, job)
        if not jsonl_path.is_file() or not csv_path.is_file():
            raise ConfigurationError("completed gate status has no complete row artifacts")
        return _finalize_evaluation_manifest(
            args,
            job,
            spec,
            training_freeze,
            status,
            _read_jsonl(jsonl_path),
        )
    if status["attempts"] and status["attempts"][-1].get("status") == "running":
        _finish_attempt(
            status,
            status_path,
            "interrupted",
            error="orphaned gate attempt recovered before a complete shard existed",
            failure_category="orphaned_process_interruption",
            retry_authorized=True,
        )
    while len(status["attempts"]) < MAX_ATTEMPTS:
        if cancel_event.is_set():
            raise _job_cancelled(
                f"gate job cancelled before attempt: {job.job_id}", cancel_event
            )
        _require_retry_authorization(status, job)
        _begin_attempt(status, status_path)
        try:
            workloads = validate_gate_workloads(GATE_WORKLOAD_SEEDS)
            policy = _make_evaluation_policy(args, job, spec, training_freeze)
            evaluated = []
            for workload_seed in workloads:
                if cancel_event.is_set():
                    raise _job_cancelled(
                        f"gate job cancelled before workload {workload_seed}: {job.job_id}",
                        cancel_event,
                    )
                episode_rows = evaluate_policy_with_constraint_metrics(
                    scenario=job.scenario,
                    policy_name=job.method,
                    policy=policy,
                    policy_seed=job.policy_seed,
                    workload_seeds=(workload_seed,),
                    variant=ENVIRONMENT_VARIANT,
                )
                if len(episode_rows) != 1:
                    raise ConfigurationError(
                        "structured evaluator did not return one row per workload"
                    )
                evaluated.extend(episode_rows)
            if not all(type(row) is ConstraintEpisodeMetrics for row in evaluated):
                raise ConfigurationError("structured evaluator returned wrong row type")
            source_model_hash = _source_model_hash(job, training_freeze)
            rows = [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_name": STUDY_NAME,
                    "evaluation_role": "classical_independent_gate_descriptive",
                    "spec_sha256": spec["spec_sha256"],
                    "training_freeze_sha256": training_freeze["freeze_sha256"],
                    "method": job.method,
                    "replicate_kind": job.replicate_kind,
                    "environment_variant": ENVIRONMENT_VARIANT,
                    "source_model_sha256": source_model_hash,
                    "decision_forced_switch_cost": 0,
                    **metric,
                }
                for metric in metrics_as_dicts(evaluated)
            ]
            summary = validate_gate_rows(job, rows, spec, training_freeze)
            _atomic_write_jsonl(jsonl_path, rows)
            atomic_write_csv(csv_path, rows, fieldnames=GATE_ROW_FIELDS)
            jsonl_hash = sha256_file(jsonl_path)
            csv_hash = sha256_file(csv_path)
            _finish_attempt(
                status,
                status_path,
                "completed",
                row_count=len(rows),
                jsonl_sha256=jsonl_hash,
                csv_sha256=csv_hash,
            )
            return _finalize_evaluation_manifest(
                args, job, spec, training_freeze, status, rows
            )
        except BaseException as error:
            if status["attempts"][-1].get("status") != "running":
                raise
            retry_authorized = _finish_attempt_from_error(
                status, status_path, error
            )
            if isinstance(error, (KeyboardInterrupt, JobCancelled)):
                raise
            if not retry_authorized:
                raise
            if len(status["attempts"]) >= MAX_ATTEMPTS:
                raise
    raise ConfigurationError(f"maximum attempts consumed: {job.job_id}")


def run_all_gate_jobs(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    validate_training_freeze(args, spec, training_freeze)
    jobs = build_evaluation_jobs()
    return _run_parallel_fail_fast(
        jobs,
        args.max_parallel,
        lambda job, cancel: evaluate_gate_job(
            args, job, spec, training_freeze, cancel
        ),
    )


def _write_or_verify_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    if path.is_file():
        if _read_jsonl(path) != [dict(row) for row in rows]:
            raise ConfigurationError(f"immutable merged JSONL mismatch: {path}")
        return
    _atomic_write_jsonl(path, rows)


def _descriptive_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for scenario in SCENARIOS:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario and row["method"] == method
            ]
            cost = sum(int(row["decision_avoidable_switches"]) for row in selected)
            opportunity = sum(
                int(row["decision_switch_opportunities"]) for row in selected
            )
            generated = sum(int(row["generated"]) for row in selected)
            delivered = sum(int(row["delivered"]) for row in selected)
            identity_rows: dict[int, list[Mapping[str, Any]]] = {}
            for row in selected:
                identity_rows.setdefault(int(row["policy_seed"]), []).append(row)
            identity_summaries = []
            for policy_seed in sorted(identity_rows):
                replicate = identity_rows[policy_seed]
                replicate_cost = sum(
                    int(row["decision_avoidable_switches"]) for row in replicate
                )
                replicate_opportunity = sum(
                    int(row["decision_switch_opportunities"]) for row in replicate
                )
                identity_summaries.append(
                    {
                        "policy_seed": policy_seed,
                        "delivery_ratio": (
                            float(
                                np.mean(
                                    [float(row["delivery_ratio"]) for row in replicate]
                                )
                            )
                            if replicate
                            else None
                        ),
                        "decision_avoidable_switches": replicate_cost,
                        "decision_switch_opportunities": replicate_opportunity,
                        "decision_avoidable_switch_rate": (
                            replicate_cost / replicate_opportunity
                            if replicate_opportunity > 0
                            else None
                        ),
                    }
                )
            identity_rates = [
                item["decision_avoidable_switch_rate"] for item in identity_summaries
            ]
            summaries[f"{scenario}/{method}"] = {
                "row_count": len(selected),
                "policy_identity_count": len(identity_summaries),
                "pooled_count_audit": {
                    "generated": generated,
                    "delivered": delivered,
                    "delivery_ratio": delivered / max(1, generated),
                    "decision_avoidable_switches": cost,
                    "decision_switch_opportunities": opportunity,
                    "decision_avoidable_switch_rate": (
                        cost / opportunity if opportunity > 0 else None
                    ),
                    "inferential_role": "pooled_integrity_audit_only",
                },
                "policy_identity_equal_descriptive": {
                    "delivery_ratio": (
                        float(
                            np.mean(
                                [item["delivery_ratio"] for item in identity_summaries]
                            )
                        )
                        if identity_summaries
                        else None
                    ),
                    "decision_avoidable_switch_rate": (
                        float(np.mean(identity_rates))
                        if identity_rates and all(rate is not None for rate in identity_rates)
                        else None
                    ),
                    "undefined_rate_identity_count": sum(
                        rate is None for rate in identity_rates
                    ),
                    "undefined_rate_policy_seeds": [
                        item["policy_seed"]
                        for item in identity_summaries
                        if item["decision_avoidable_switch_rate"] is None
                    ],
                    "identities": identity_summaries,
                },
            }
    return summaries


def validate_gate_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    path = args.output / "classical_gate_freeze.json"
    freeze = read_json(path)
    validate_self_hash(freeze, "gate_freeze_sha256")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "gate_complete": True,
        "gate_row_count": EXPECTED_GATE_ROWS,
        "gate_rows_by_method": {
            Q_ROUTING: EXPECTED_Q_GATE_ROWS,
            OSPF_ECMP: EXPECTED_OSPF_GATE_ROWS,
            GLOBAL_DIJKSTRA: EXPECTED_DIJKSTRA_GATE_ROWS,
        },
        "checkpoint_selection_evaluations": 0,
        "performance_gate_applied": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "sealed_test_access_authorized": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
    }
    if any(freeze.get(field) != value for field, value in expected.items()):
        raise ConfigurationError("classical gate freeze contract drifted")
    expected_merged_paths = {
        "merged_jsonl": (args.output / "classical_gate_rows.jsonl").resolve(),
        "merged_csv": (args.output / "classical_gate_rows.csv").resolve(),
    }
    for field, expected_path in expected_merged_paths.items():
        artifact = freeze.get(field)
        if not isinstance(artifact, Mapping):
            raise ConfigurationError(f"classical gate freeze omits {field}")
        artifact_path = Path(str(artifact.get("path", ""))).resolve()
        if (
            artifact_path != expected_path
            or not _path_within(artifact_path, args.output.resolve())
            or not artifact_path.is_file()
            or artifact.get("sha256") != sha256_file(artifact_path)
        ):
            raise ConfigurationError(f"classical gate freeze artifact drifted: {field}")
    rows = _read_jsonl(expected_merged_paths["merged_jsonl"])
    if _read_gate_csv(expected_merged_paths["merged_csv"]) != rows:
        raise ConfigurationError("merged classical gate JSONL/CSV disagree")
    if len(rows) != EXPECTED_GATE_ROWS:
        raise ConfigurationError("merged classical gate row count drifted")
    shards = freeze.get("shards")
    expected_shard_ids = [job.job_id for job in build_evaluation_jobs()]
    if not isinstance(shards, Mapping) or sorted(shards) != sorted(expected_shard_ids):
        raise ConfigurationError("classical gate freeze shard inventory is malformed")
    cursor = 0
    for job in build_evaluation_jobs():
        shard_manifest, shard_rows = validate_evaluation_shard(
            args, job, spec, training_freeze
        )
        if rows[cursor : cursor + len(shard_rows)] != shard_rows:
            raise ConfigurationError("merged classical gate row order drifted")
        cursor += len(shard_rows)
        entry = shards.get(job.job_id)
        manifest_path = evaluation_shard_paths(args, job)[2]
        if entry != {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_sha256": shard_manifest["manifest_sha256"],
            "status": shard_manifest["status"],
            "attempt_count": shard_manifest["attempt_count"],
        }:
            raise ConfigurationError(f"gate freeze shard drifted: {job.job_id}")
    if freeze.get("descriptive_summaries") != _descriptive_summaries(rows):
        raise ConfigurationError("classical descriptive summaries drifted")
    return freeze


def write_gate_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    freeze_path = args.output / "classical_gate_freeze.json"
    if freeze_path.is_file():
        return validate_gate_freeze(args, spec, training_freeze)
    expected_ids = [job.job_id for job in build_evaluation_jobs()]
    if list(manifests) != expected_ids:
        raise ConfigurationError("cannot freeze partial/reordered classical gate grid")
    rows: list[dict[str, Any]] = []
    for job in build_evaluation_jobs():
        observed, shard_rows = validate_evaluation_shard(
            args, job, spec, training_freeze
        )
        if observed != manifests[job.job_id]:
            raise ConfigurationError(f"gate shard changed before freeze: {job.job_id}")
        rows.extend(shard_rows)
    if len(rows) != EXPECTED_GATE_ROWS:
        raise ConfigurationError("classical gate grid is incomplete")
    expected_keys = {
        (job.scenario, job.method, job.policy_seed, workload_seed)
        for job in build_evaluation_jobs()
        for workload_seed in GATE_WORKLOAD_SEEDS
    }
    observed_keys = {
        (
            row["scenario"],
            row["method"],
            row["policy_seed"],
            row["workload_seed"],
        )
        for row in rows
    }
    if len(observed_keys) != len(rows) or observed_keys != expected_keys:
        raise ConfigurationError("classical gate has missing/duplicate cells")
    merged_jsonl = args.output / "classical_gate_rows.jsonl"
    merged_csv = args.output / "classical_gate_rows.csv"
    _write_or_verify_jsonl(merged_jsonl, rows)
    atomic_write_csv(merged_csv, rows, fieldnames=GATE_ROW_FIELDS)
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "gate_complete": True,
        "gate_row_count": len(rows),
        "gate_rows_by_method": {
            Q_ROUTING: EXPECTED_Q_GATE_ROWS,
            OSPF_ECMP: EXPECTED_OSPF_GATE_ROWS,
            GLOBAL_DIJKSTRA: EXPECTED_DIJKSTRA_GATE_ROWS,
        },
        "checkpoint_selection_evaluations": 0,
        "performance_gate_applied": False,
        "descriptive_summaries": _descriptive_summaries(rows),
        "merged_jsonl": {
            "path": str(merged_jsonl.resolve()),
            "sha256": sha256_file(merged_jsonl),
        },
        "merged_csv": {
            "path": str(merged_csv.resolve()),
            "sha256": sha256_file(merged_csv),
        },
        "shards": {
            job.job_id: {
                "manifest_path": str(evaluation_shard_paths(args, job)[2].resolve()),
                "manifest_file_sha256": sha256_file(
                    evaluation_shard_paths(args, job)[2]
                ),
                "manifest_sha256": manifests[job.job_id]["manifest_sha256"],
                "status": manifests[job.job_id]["status"],
                "attempt_count": manifests[job.job_id]["attempt_count"],
            }
            for job in build_evaluation_jobs()
        },
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "sealed_test_access_authorized": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "completed_at_utc": utc_now(),
    }
    freeze = self_hashed(body, "gate_freeze_sha256")
    ensure_immutable_json(freeze_path, freeze)
    return validate_gate_freeze(args, spec, training_freeze)


def _pid_is_active(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def _invocation_lock_guard(output: Path):
    """Serialize main-lock inspection so stale recovery cannot race acquisition."""
    guard_path = output / ".classical_runner.lock.guard"
    descriptor = os.open(guard_path, os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ConfigurationError(
                "another classical runner is acquiring or releasing the output lock"
            ) from error
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def invocation_lock(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".classical_runner.lock"
    hostname = socket.gethostname()
    token = uuid.uuid4().hex
    payload = {
        "schema_version": 1,
        "study_name": STUDY_NAME,
        "pid": os.getpid(),
        "host": hostname,
        "output": str(output),
        "token": token,
        "created_at_utc": utc_now(),
    }
    with _invocation_lock_guard(output):
        if path.exists():
            try:
                existing = read_json(path)
            except ConfigurationError as error:
                raise ConfigurationError("existing classical lock is malformed") from error
            existing_pid = existing.get("pid")
            if (
                existing.get("schema_version") != 1
                or existing.get("study_name") != STUDY_NAME
                or isinstance(existing_pid, bool)
                or not isinstance(existing_pid, int)
                or existing_pid <= 0
                or existing.get("output") != str(output)
                or existing.get("host") != hostname
                or not isinstance(existing.get("token"), str)
                or not isinstance(existing.get("created_at_utc"), str)
            ):
                raise ConfigurationError(
                    "existing classical lock is not a recoverable same-output local lock"
                )
            if _pid_is_active(existing_pid):
                raise ConfigurationError(
                    f"another classical runner is active with pid {existing_pid}"
                )
            path.unlink()
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ConfigurationError(
                "another classical runner invocation owns the output"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    try:
        yield
    finally:
        with _invocation_lock_guard(output):
            try:
                observed = read_json(path)
            except ConfigurationError:
                observed = None
            if observed == payload:
                path.unlink(missing_ok=True)


def dry_run_summary(spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "q_training_jobs": EXPECTED_TRAINING_JOBS,
        "q_training_episodes": EXPECTED_Q_TRAIN_EPISODES,
        "checkpoint_selection_evaluations": 0,
        "gate_evaluation_jobs": EXPECTED_EVALUATION_JOBS,
        "gate_evaluations": EXPECTED_GATE_ROWS,
        "gate_evaluations_by_method": {
            Q_ROUTING: EXPECTED_Q_GATE_ROWS,
            OSPF_ECMP: EXPECTED_OSPF_GATE_ROWS,
            GLOBAL_DIJKSTRA: EXPECTED_DIJKSTRA_GATE_ROWS,
        },
        "sealed_test_evaluations": 0,
        "test_panel_consulted": False,
        "output": str(output.resolve()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run frozen classical baselines through independent gate only."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "experiments" / DEFAULT_OUTPUT_DIRECTORY_NAME,
    )
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    args.project = ACTUAL_PROJECT_SOURCE
    validate_runtime_environment(args, execute=not args.dry_run)
    validate_output_isolation(args)
    validate_new_output_state(args.output)
    spec = build_spec(args)
    if args.dry_run:
        print(json.dumps(dry_run_summary(spec, args.output), indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_immutable_json(args.output / "preregistration.json", spec)
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    invocation_id = uuid.uuid4().hex
    invocation_path = args.output / "invocations" / f"{invocation_id}.json"
    invocation = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "invocation_id": invocation_id,
        "spec_sha256": spec["spec_sha256"],
        "status": "running",
        "started_at_utc": utc_now(),
        "maximum_attempts_per_job": MAX_ATTEMPTS,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
    }
    atomic_write_json(invocation_path, invocation)
    try:
        with invocation_lock(args.output):
            training_manifests = run_all_training_jobs(args, spec)
            training_freeze = write_training_freeze(args, spec, training_manifests)
            assert_runtime_fingerprint(args, spec["code_fingerprint"])
            gate_manifests = run_all_gate_jobs(args, spec, training_freeze)
            gate_freeze = write_gate_freeze(
                args, spec, training_freeze, gate_manifests
            )
        invocation.update(
            status="completed",
            finished_at_utc=utc_now(),
            training_freeze_sha256=training_freeze["freeze_sha256"],
            gate_freeze_sha256=gate_freeze["gate_freeze_sha256"],
            sealed_test_access_authorized=False,
            paper_claim_allowed=False,
        )
        atomic_write_json(invocation_path, invocation)
        print(
            f"classical gate audit complete: rows={EXPECTED_GATE_ROWS} "
            f"output={args.output}",
            flush=True,
        )
        return 0
    except BaseException as error:
        invocation.update(
            status=("interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
            finished_at_utc=utc_now(),
            error=repr(error),
            sealed_test_access_authorized=False,
            paper_claim_allowed=False,
        )
        atomic_write_json(invocation_path, invocation)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
