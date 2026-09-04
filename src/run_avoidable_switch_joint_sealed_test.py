from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import run_avoidable_switch_classical_baselines_formal as classical
import run_avoidable_switch_constraint_formal as formal


STUDY_NAME = "ICC-AVOIDABLE-SWITCH-JOINT-SEALED-TEST-v1"
TEST_SEEDS = tuple(range(78001, 78051))
BOOTSTRAP_DRAWS = 5000
DELIVERY_MARGIN = -0.02
RATE_BUDGET = 0.12
METHOD_DIR = Path("experiments/avoidable-switch-constraint-formal-v1-r2")
CLASSICAL_DIR = Path("experiments/avoidable-switch-classical-baselines-formal-v1-r3")
PROTOCOL = Path("docs/AVOIDABLE_SWITCH_JOINT_SEALED_TEST_V1.md")
OUTPUT = Path("experiments/avoidable-switch-joint-sealed-test-v1")
EXPECTED_FREEZES = {
    "method_training": "153b5cd85305f3f4c9a03fd549eea3e06785a0d1a447b96ec9005098c7d25742",
    "method_validation": "d3c039657802e9c784c82cec31024e5098b31e5a3e217e69bbc64dad5a8469d6",
    "classical_training": "df7cf104d30360df898ef8239b64789c16e0baa9e87c78c52c08a676ecb62eba",
    "classical_gate": "36bac779a6961ce38eb6872ec0f9cfa3953898ceb00d9b5bd7f10ce80fbc3a1e",
}


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def validate_self_hash(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    body = dict(value)
    body.pop(field, None)
    expected = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if observed != expected:
        raise ContractError(f"self hash mismatch: {field}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def paths() -> dict[str, Path]:
    root = project_root()
    return {
        "method": (root / METHOD_DIR).resolve(),
        "classical": (root / CLASSICAL_DIR).resolve(),
        "protocol": (root / PROTOCOL).resolve(),
        "runner": Path(__file__).resolve(),
    }


def load_prerequisites() -> dict[str, Any]:
    bound = paths()
    method_training = read_json(bound["method"] / "training_freeze.json")
    method_validation = read_json(bound["method"] / "validation_freeze.json")
    classical_training = read_json(bound["classical"] / "training_freeze.json")
    classical_gate = read_json(bound["classical"] / "classical_gate_freeze.json")
    for value, field in (
        (method_training, "freeze_sha256"),
        (method_validation, "validation_freeze_sha256"),
        (classical_training, "freeze_sha256"),
        (classical_gate, "gate_freeze_sha256"),
    ):
        validate_self_hash(value, field)
    observed = {
        "method_training": method_training["freeze_sha256"],
        "method_validation": method_validation["validation_freeze_sha256"],
        "classical_training": classical_training["freeze_sha256"],
        "classical_gate": classical_gate["gate_freeze_sha256"],
    }
    if observed != EXPECTED_FREEZES:
        raise ContractError("prerequisite freeze binding changed")
    if not all(
        value.get("test_access_count") == 0
        and value.get("test_panel_consulted") is False
        for value in (method_training, method_validation, classical_training, classical_gate)
    ):
        raise ContractError("a prerequisite reports prior sealed-test access")
    if (
        method_validation.get("formal_validation_gate_passed") is not True
        or method_validation.get("integrity_gates_passed") is not True
        or classical_gate.get("gate_complete") is not True
    ):
        raise ContractError("prerequisite validation is incomplete")
    return {
        "paths": bound,
        "method_training": method_training,
        "method_validation": method_validation,
        "classical_training": classical_training,
        "classical_gate": classical_gate,
    }


def authorization_body(prereq: Mapping[str, Any]) -> dict[str, Any]:
    bound = prereq["paths"]
    return {
        "schema_version": 1,
        "study_name": STUDY_NAME,
        "authorization": "operator_authorized_one_time_complete_grid",
        "authorized_at_local": "2026-09-04",
        "protocol": {"path": str(bound["protocol"]), "sha256": sha256_file(bound["protocol"])},
        "runner": {"path": str(bound["runner"]), "sha256": sha256_file(bound["runner"])},
        "freezes": dict(EXPECTED_FREEZES),
        "scenarios": list(formal.SCENARIOS),
        "mappo_arms": list(formal.ARMS),
        "classical_methods": [classical.Q_ROUTING, classical.OSPF_ECMP, classical.GLOBAL_DIJKSTRA],
        "policy_seeds": list(formal.POLICY_SEEDS),
        "sealed_workload_seeds": list(TEST_SEEDS),
        "expected_row_count": 4100,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "delivery_noninferiority_threshold": DELIVERY_MARGIN,
        "decision_rate_budget": RATE_BUDGET,
        "test_access_count": 0,
        "test_panel_consulted": False,
        "sealed_test_instantiated": False,
    }


def prepare(output: Path) -> dict[str, Any]:
    prereq = load_prerequisites()
    auth = self_hashed(authorization_body(prereq), "authorization_sha256")
    path = output / "sealed_test_authorization.json"
    if path.exists():
        observed = read_json(path)
        validate_self_hash(observed, "authorization_sha256")
        if observed != auth:
            raise ContractError("authorization artifact changed")
        return observed
    if output.exists() and any(output.iterdir()):
        raise ContractError("new sealed-test output contains unexpected files")
    write_json(path, auth)
    return auth


def validate_authorization(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereq = load_prerequisites()
    path = output / "sealed_test_authorization.json"
    auth = read_json(path)
    validate_self_hash(auth, "authorization_sha256")
    expected = self_hashed(authorization_body(prereq), "authorization_sha256")
    if auth != expected:
        raise ContractError("sealed-test authorization binding changed")
    return auth, prereq


def metric_rows(metrics: Iterable[Any]) -> list[dict[str, Any]]:
    return formal.metrics_as_dicts(list(metrics))


def evaluate_mappo(prereq: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    freeze = prereq["method_training"]
    for job in formal.build_jobs():
        artifact = freeze["jobs"][job.job_id]
        checkpoint = artifact["artifacts"]["selected_checkpoint"]
        checkpoint_path = Path(checkpoint["path"]).resolve()
        if sha256_file(checkpoint_path) != checkpoint["sha256"]:
            raise ContractError(f"MAPPO checkpoint changed: {job.job_id}")
        policy, _ = formal.load_checkpoint_policy(checkpoint_path, device="cuda")
        result = formal.evaluate_policy_with_constraint_metrics(
            scenario=job.scenario,
            policy_name=job.arm,
            policy=policy,
            policy_seed=job.policy_seed,
            workload_seeds=TEST_SEEDS,
            variant=job.environment_variant,
        )
        for metric in metric_rows(result):
            rows.append(
                {
                    "study_name": STUDY_NAME,
                    "evaluation_role": "sealed_test",
                    "method": job.arm,
                    "replicate_kind": "independent_policy_seed",
                    "source_artifact_sha256": checkpoint["sha256"],
                    **metric,
                }
            )
    return rows


def evaluate_classical(prereq: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bound = prereq["paths"]
    spec = read_json(bound["classical"] / "preregistration.json")
    args = SimpleNamespace(output=bound["classical"])
    freeze = prereq["classical_training"]
    for job in classical.build_evaluation_jobs():
        policy = classical._make_evaluation_policy(args, job, spec, freeze)
        result = formal.evaluate_policy_with_constraint_metrics(
            scenario=job.scenario,
            policy_name=job.method,
            policy=policy,
            policy_seed=job.policy_seed,
            workload_seeds=TEST_SEEDS,
            variant=classical.ENVIRONMENT_VARIANT,
        )
        source_hash = classical._source_model_hash(job, freeze)
        for metric in metric_rows(result):
            rows.append(
                {
                    "study_name": STUDY_NAME,
                    "evaluation_role": "sealed_test",
                    "method": job.method,
                    "replicate_kind": job.replicate_kind,
                    "source_artifact_sha256": source_hash,
                    **metric,
                }
            )
    return rows


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    tolerance = 8.0 * np.finfo(float).eps * max(1.0, observed)
    hits = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        if abs(float(np.mean(values * np.asarray(signs)))) >= observed - tolerance:
            hits += 1
    return hits / (2 ** len(values))


def rng_for(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def bootstrap_delivery(diff: np.ndarray, scenario: str) -> np.ndarray:
    rng = rng_for(STUDY_NAME, scenario, "delivery")
    n_seed, n_workload = diff.shape
    draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        si = rng.integers(0, n_seed, n_seed)
        wi = rng.integers(0, n_workload, n_workload)
        draws[index] = diff[np.ix_(si, wi)].mean()
    return draws


def bootstrap_rates(
    treatment_cost: np.ndarray,
    treatment_opp: np.ndarray,
    reference_cost: np.ndarray,
    reference_opp: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng_for(STUDY_NAME, scenario, "rates")
    n_seed, n_workload = treatment_cost.shape
    effect = np.empty(BOOTSTRAP_DRAWS)
    treatment = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        si = rng.integers(0, n_seed, n_seed)
        wi = rng.integers(0, n_workload, n_workload)
        tc = treatment_cost[np.ix_(si, wi)].sum(axis=1)
        to = treatment_opp[np.ix_(si, wi)].sum(axis=1)
        rc = reference_cost[np.ix_(si, wi)].sum(axis=1)
        ro = reference_opp[np.ix_(si, wi)].sum(axis=1)
        if np.any(to == 0) or np.any(ro == 0):
            raise ContractError("undefined rate in bootstrap")
        tr = tc / to
        rr = rc / ro
        treatment[index] = tr.mean()
        effect[index] = (tr - rr).mean()
    return effect, treatment


def holm(raw: Sequence[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda index: raw[index])
    adjusted = [0.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def row_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (row["scenario"], row["method"], row["policy_seed"], row["workload_seed"])
        if key in result:
            raise ContractError(f"duplicate sealed-test row: {key}")
        result[key] = row
    return result


def matrix(
    index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    scenario: str,
    method: str,
    field: str,
) -> np.ndarray:
    return np.asarray(
        [
            [index[(scenario, method, seed, workload)][field] for workload in TEST_SEEDS]
            for seed in formal.POLICY_SEEDS
        ],
        dtype=float,
    )


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 4100:
        raise ContractError(f"sealed-test row count is {len(rows)}, expected 4100")
    index = row_index(rows)
    scenarios: list[dict[str, Any]] = []
    p_values: list[float] = []
    p_targets: list[dict[str, Any]] = []
    for scenario in formal.SCENARIOS:
        delivery_t = matrix(index, scenario, formal.ARM_CONSTRAINED, "delivery_ratio")
        delivery_b = matrix(index, scenario, formal.ARM_BASELINE, "delivery_ratio")
        delivery_diff = delivery_t - delivery_b
        delivery_seed = delivery_diff.mean(axis=1)
        delivery_draws = bootstrap_delivery(delivery_diff, scenario)
        tc = matrix(index, scenario, formal.ARM_CONSTRAINED, "decision_avoidable_switches")
        to = matrix(index, scenario, formal.ARM_CONSTRAINED, "decision_switch_opportunities")
        bc = matrix(index, scenario, formal.ARM_BASELINE, "decision_avoidable_switches")
        bo = matrix(index, scenario, formal.ARM_BASELINE, "decision_switch_opportunities")
        if np.any(to.sum(axis=1) == 0) or np.any(bo.sum(axis=1) == 0):
            raise ContractError("undefined primary policy-seed rate")
        treatment_seed_rate = tc.sum(axis=1) / to.sum(axis=1)
        baseline_seed_rate = bc.sum(axis=1) / bo.sum(axis=1)
        rate_seed_diff = treatment_seed_rate - baseline_seed_rate
        rate_draws, treatment_draws = bootstrap_rates(tc, to, bc, bo, scenario)
        delivery_record = {
            "endpoint": "delivery_ratio_difference",
            "estimate": float(delivery_seed.mean()),
            "ci95": [float(np.quantile(delivery_draws, 0.025)), float(np.quantile(delivery_draws, 0.975))],
            "one_sided_95_lower": float(np.quantile(delivery_draws, 0.05)),
            "raw_exact_p": exact_sign_flip(delivery_seed),
            "policy_seed_effects": delivery_seed.tolist(),
        }
        rate_record = {
            "endpoint": "decision_avoidable_switch_rate_difference",
            "estimate": float(rate_seed_diff.mean()),
            "ci95": [float(np.quantile(rate_draws, 0.025)), float(np.quantile(rate_draws, 0.975))],
            "one_sided_95_upper": float(np.quantile(rate_draws, 0.95)),
            "raw_exact_p": exact_sign_flip(rate_seed_diff),
            "policy_seed_effects": rate_seed_diff.tolist(),
        }
        p_values.extend([delivery_record["raw_exact_p"], rate_record["raw_exact_p"]])
        p_targets.extend([delivery_record, rate_record])
        constrained_rate = {
            "estimate": float(treatment_seed_rate.mean()),
            "one_sided_95_upper": float(np.quantile(treatment_draws, 0.95)),
            "maximum_policy_seed_rate": float(treatment_seed_rate.max()),
            "policy_seed_rates": treatment_seed_rate.tolist(),
        }
        gates = {
            "delivery_noninferiority": delivery_record["one_sided_95_lower"] >= DELIVERY_MARGIN,
            "rate_reduction": rate_record["one_sided_95_upper"] < 0.0,
            "rate_budget_upper": constrained_rate["one_sided_95_upper"] <= RATE_BUDGET,
            "every_policy_seed_within_budget": constrained_rate["maximum_policy_seed_rate"] <= RATE_BUDGET,
        }
        method_summaries = []
        for method in (*formal.ARMS, classical.Q_ROUTING, classical.OSPF_ECMP):
            delivery = matrix(index, scenario, method, "delivery_ratio")
            cost = matrix(index, scenario, method, "decision_avoidable_switches")
            opp = matrix(index, scenario, method, "decision_switch_opportunities")
            rates = np.divide(cost.sum(axis=1), opp.sum(axis=1), where=opp.sum(axis=1) != 0)
            method_summaries.append(
                {"method": method, "delivery_ratio": float(delivery.mean()), "decision_avoidable_switch_rate": float(rates.mean())}
            )
        dijkstra_rows = [
            row for row in rows
            if row["scenario"] == scenario and row["method"] == classical.GLOBAL_DIJKSTRA
        ]
        method_summaries.append(
            {
                "method": classical.GLOBAL_DIJKSTRA,
                "delivery_ratio": float(np.mean([row["delivery_ratio"] for row in dijkstra_rows])),
                "decision_avoidable_switch_rate": float(
                    sum(row["decision_avoidable_switches"] for row in dijkstra_rows)
                    / sum(row["decision_switch_opportunities"] for row in dijkstra_rows)
                ),
            }
        )
        scenarios.append(
            {
                "scenario": scenario,
                "delivery": delivery_record,
                "rate_difference": rate_record,
                "constrained_rate": constrained_rate,
                "gates": gates,
                "all_gates_passed": all(gates.values()),
                "method_summaries": method_summaries,
            }
        )
    for record, adjusted in zip(p_targets, holm(p_values)):
        record["holm_adjusted_p"] = adjusted
    return {
        "schema_version": 1,
        "study_name": STUDY_NAME,
        "analysis_role": "preregistered_sealed_test",
        "row_count": len(rows),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "scenarios": scenarios,
        "all_primary_gates_passed": all(item["all_gates_passed"] for item in scenarios),
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_bytes(row).decode() + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def execute(output: Path) -> dict[str, Any]:
    auth, prereq = validate_authorization(output)
    if any(output.glob("sealed_test_*rows*")) or (output / "sealed_test_freeze.json").exists():
        raise ContractError("sealed-test output already exists; one-time execution refused")
    rows = evaluate_mappo(prereq) + evaluate_classical(prereq)
    rows.sort(key=lambda row: (row["scenario"], row["method"], row["policy_seed"], row["workload_seed"]))
    stats = summarize(rows)
    rows_jsonl = output / "sealed_test_rows.jsonl"
    rows_csv = output / "sealed_test_rows.csv"
    stats_path = output / "sealed_test_statistics.json"
    write_rows(rows_jsonl, rows)
    write_csv(rows_csv, rows)
    stats = self_hashed(stats, "statistics_sha256")
    write_json(stats_path, stats)
    freeze = self_hashed(
        {
            "schema_version": 1,
            "study_name": STUDY_NAME,
            "authorization_sha256": auth["authorization_sha256"],
            "row_count": len(rows),
            "rows_jsonl": {"path": str(rows_jsonl.resolve()), "sha256": sha256_file(rows_jsonl)},
            "rows_csv": {"path": str(rows_csv.resolve()), "sha256": sha256_file(rows_csv)},
            "statistics": {"path": str(stats_path.resolve()), "sha256": sha256_file(stats_path), "statistics_sha256": stats["statistics_sha256"]},
            "freezes": dict(EXPECTED_FREEZES),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "protocol_sha256": sha256_file(paths()["protocol"]),
            "sealed_test_complete": True,
            "test_panel_consulted": True,
            "test_access_count": len(rows),
            "paper_claim_allowed": stats["all_primary_gates_passed"],
            "promotion_decision_allowed": True,
        },
        "sealed_test_freeze_sha256",
    )
    write_json(output / "sealed_test_freeze.json", freeze)
    return freeze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=project_root() / OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.dry_run:
        load_prerequisites()
        print(json.dumps({"study_name": STUDY_NAME, "expected_rows": 4100, "sealed_test_access": 0}, indent=2))
        return 0
    if args.prepare:
        auth = prepare(output)
        print(f"sealed-test authorization prepared: {auth['authorization_sha256']}")
        return 0
    freeze = execute(output)
    print(f"sealed-test complete: rows={freeze['row_count']} claim={freeze['paper_claim_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
