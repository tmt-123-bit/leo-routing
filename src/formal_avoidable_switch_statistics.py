"""Frozen independent-gate statistics for the formal avoidable-switch study.

The independently trained policy seed is the algorithmic replication unit.
Workload seeds are paired repeated measurements shared by every arm.  Rate
estimands are always reconstructed from decision counts; episode rates are
validated for consistency but are never averaged to form an aggregate rate.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
STUDY_NAME = "ICC-AVOIDABLE-SWITCH-CONSTRAINT-FORMAL-v1"

SCENARIOS = ("medium_load", "hotspot_high_load")
ARM_QOS_ONLY_BASELINE = "qos_only_baseline"
ARM_QOS_ONLY_CONSTRAINED = "qos_only_constrained"
ARM_REWARD_SHAPED_CONTROL = "reward_shaped_control"
ARMS = (
    ARM_QOS_ONLY_BASELINE,
    ARM_QOS_ONLY_CONSTRAINED,
    ARM_REWARD_SHAPED_CONTROL,
)

POLICY_SEED_NAMESPACE = (
    "ICC-AVOIDABLE-SWITCH-CONSTRAINT-v1-formal-policy-seed-"
)


def _derived_policy_seed(index: int) -> int:
    digest = hashlib.sha256(
        f"{POLICY_SEED_NAMESPACE}{index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


POLICY_SEEDS = tuple(_derived_policy_seed(index) for index in range(8))
TRAIN_WORKLOAD_SEEDS = tuple(range(76001, 76201))
SELECTION_WORKLOAD_SEEDS = tuple(range(77001, 77011))
GATE_WORKLOAD_SEEDS = tuple(range(77011, 77021))
# Kept as the matrix-axis name for callers; formal statistics use gate rows only.
VALIDATION_WORKLOAD_SEEDS = GATE_WORKLOAD_SEEDS
EXPECTED_GATE_ROW_COUNT = (
    len(SCENARIOS) * len(ARMS) * len(POLICY_SEEDS) * len(GATE_WORKLOAD_SEEDS)
)

SWITCH_BUDGET = 0.12
DELIVERY_NONINFERIORITY_MARGIN = 0.02
BOOTSTRAP_RESAMPLES = 5000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_MINIMUM_MAXIMUM_DRAWS = 10000
BOOTSTRAP_MAXIMUM_DRAW_MULTIPLIER = 1000
PRIMARY_BOOTSTRAP_SEED_BASE = 93100
SECONDARY_BOOTSTRAP_SEED_BASE = 95100

_CORE_FIELDS = (
    "scenario",
    "arm",
    "policy_seed",
    "workload_seed",
    "delivery_ratio",
    "decision_avoidable_switches",
    "decision_switch_opportunities",
    "decision_forced_switches",
    "decision_forced_switch_cost",
    "decision_avoidable_switch_rate",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_report_hash(
    report: Mapping[str, Any], *, rows: Sequence[Any]
) -> None:
    body = dict(report)
    observed = body.pop("report_sha256", None)
    if observed != sha256_json(body):
        raise ValueError("report_sha256 mismatch")
    frozen_fields = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "analysis_scope": "formal_independent_validation_gate_only",
        "inferential_status": (
            "independent_validation_gate_only_not_test_or_paper_evidence"
        ),
        "validation_only": True,
        "independent_validation_gate": True,
        "checkpoint_selection_rows_included": False,
        "final_test_evidence": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "sealed_test_access_count": 0,
        "sealed_test_access": 0,
        "test_access_count": 0,
        "test_panel_consulted": False,
        "sealed_test_workloads_instantiated": False,
    }
    for field, expected in frozen_fields.items():
        if body.get(field) != expected:
            raise ValueError(f"formal report semantic field {field!r} drifted")
    grid = body.get("grid_audit")
    if not isinstance(grid, Mapping) or (
        grid.get("scenarios") != list(SCENARIOS)
        or grid.get("arms") != list(ARMS)
        or grid.get("policy_seeds") != list(POLICY_SEEDS)
        or grid.get("selection_workload_seeds_excluded")
        != list(SELECTION_WORKLOAD_SEEDS)
        or grid.get("gate_workload_seeds") != list(GATE_WORKLOAD_SEEDS)
        or grid.get("expected_rows") != EXPECTED_GATE_ROW_COUNT
        or grid.get("observed_rows") != EXPECTED_GATE_ROW_COUNT
        or grid.get("grid_complete") is not True
    ):
        raise ValueError("formal report grid contract drifted")
    statistical_contract = body.get("statistical_contract")
    if not isinstance(statistical_contract, Mapping):
        raise ValueError("formal report omits its statistical contract")
    bootstrap = statistical_contract.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or (
        bootstrap.get("resamples") != BOOTSTRAP_RESAMPLES
        or bootstrap.get("maximum_total_draws")
        != max(
            BOOTSTRAP_MINIMUM_MAXIMUM_DRAWS,
            BOOTSTRAP_MAXIMUM_DRAW_MULTIPLIER * BOOTSTRAP_RESAMPLES,
        )
        or bootstrap.get("insufficient_defined_draws") != "fail_closed"
    ):
        raise ValueError("formal report bootstrap contract drifted")
    primary = body.get("primary")
    if not isinstance(primary, Mapping):
        raise ValueError("formal report omits primary results")
    differences = primary.get("paired_differences")
    constrained_rates = primary.get("constrained_rates")
    if not isinstance(differences, list) or len(differences) != 4:
        raise ValueError("formal report primary difference family drifted")
    if not isinstance(constrained_rates, list) or len(constrained_rates) != 2:
        raise ValueError("formal report constrained-rate family drifted")
    observed_adjusted = [record.get("holm_adjusted_p") for record in differences]
    expected_adjusted = _holm_adjusted(
        [record.get("raw_p_value") for record in differences]
    )
    if any(
        not isinstance(observed_value, Real)
        or float(observed_value) != float(expected_value)
        for observed_value, expected_value in zip(
            observed_adjusted, expected_adjusted
        )
    ):
        raise ValueError("formal report Holm adjustment is inconsistent")
    gates = body.get("validation_gates")
    if not isinstance(gates, list) or len(gates) != 8:
        raise ValueError("formal report validation gate family drifted")
    expected_gate = all(record.get("passed") is True for record in gates)
    if body.get("validation_gate_passed") is not expected_gate:
        raise ValueError("formal report aggregate validation gate is inconsistent")
    expected_report = analyze_validation_rows(rows)
    if dict(report) != expected_report:
        raise ValueError("formal report differs from fixed-input recomputation")


def _field(row: Any, name: str) -> Any:
    try:
        if isinstance(row, Mapping):
            return row[name]
        return getattr(row, name)
    except (KeyError, AttributeError) as error:
        raise ValueError(f"validation row is missing required field {name!r}") from error


def _json_native(value: Any, field: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value in full input field {field!r}")
        return value
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("full input mappings must use string keys")
            output[key] = _json_native(item, f"{field}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            _json_native(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(
        f"full input field {field!r} is not JSON-native: {type(value).__name__}"
    )


def _full_input_rows(
    source_rows: Sequence[Any], normalized: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    source_by_cell: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in source_rows:
        if isinstance(row, Mapping):
            source = dict(row)
        elif hasattr(row, "__dict__"):
            source = dict(vars(row))
        else:
            raise ValueError(
                "validation rows must be mappings or objects with a field dictionary"
            )
        native = _json_native(source, "row")
        key = (
            str(_field(row, "scenario")),
            str(_field(row, "arm")),
            int(_field(row, "policy_seed")),
            int(_field(row, "workload_seed")),
        )
        source_by_cell[key] = native
    return [
        source_by_cell[
            (
                row["scenario"],
                row["arm"],
                row["policy_seed"],
                row["workload_seed"],
            )
        ]
        for row in normalized
    ]


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _normalize_rows(rows: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    rows = tuple(rows)
    expected_count = (
        len(SCENARIOS)
        * len(ARMS)
        * len(POLICY_SEEDS)
        * len(VALIDATION_WORKLOAD_SEEDS)
    )
    if len(rows) != expected_count:
        raise ValueError(
            f"formal validation grid requires exactly {expected_count} rows; "
            f"observed {len(rows)}"
        )

    expected_scenarios = set(SCENARIOS)
    expected_arms = set(ARMS)
    expected_policy_seeds = set(POLICY_SEEDS)
    expected_workload_seeds = set(VALIDATION_WORKLOAD_SEEDS)
    cells: dict[tuple[str, str, int, int], dict[str, Any]] = {}

    for source in rows:
        scenario = _field(source, "scenario")
        arm = _field(source, "arm")
        if scenario not in expected_scenarios:
            raise ValueError(f"unexpected formal scenario {scenario!r}")
        if arm not in expected_arms:
            raise ValueError(f"unexpected formal arm {arm!r}")

        policy_seed = _exact_int(_field(source, "policy_seed"), "policy_seed")
        workload_seed = _exact_int(
            _field(source, "workload_seed"), "workload_seed"
        )
        if policy_seed not in expected_policy_seeds:
            raise ValueError(f"unexpected formal policy seed {policy_seed}")
        if workload_seed not in expected_workload_seeds:
            raise ValueError(
                f"unexpected formal independent gate workload seed {workload_seed}"
            )

        delivery = _finite_float(
            _field(source, "delivery_ratio"), "delivery_ratio"
        )
        reported_rate = _finite_float(
            _field(source, "decision_avoidable_switch_rate"),
            "decision_avoidable_switch_rate",
        )
        if not 0.0 <= delivery <= 1.0:
            raise ValueError("delivery_ratio must lie in [0, 1]")
        if not 0.0 <= reported_rate <= 1.0:
            raise ValueError("decision_avoidable_switch_rate must lie in [0, 1]")

        cost = _exact_int(
            _field(source, "decision_avoidable_switches"),
            "decision_avoidable_switches",
        )
        opportunity = _exact_int(
            _field(source, "decision_switch_opportunities"),
            "decision_switch_opportunities",
        )
        forced = _exact_int(
            _field(source, "decision_forced_switches"),
            "decision_forced_switches",
        )
        forced_cost = _exact_int(
            _field(source, "decision_forced_switch_cost"),
            "decision_forced_switch_cost",
        )
        for value, field in (
            (cost, "decision_avoidable_switches"),
            (opportunity, "decision_switch_opportunities"),
            (forced, "decision_forced_switches"),
            (forced_cost, "decision_forced_switch_cost"),
        ):
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        if cost > opportunity:
            raise ValueError("decision avoidable cost exceeds opportunity")
        if forced_cost != 0:
            raise ValueError("forced decision constraint cost must be zero")

        reconstructed_rate = cost / opportunity if opportunity else 0.0
        if not math.isclose(
            reported_rate,
            reconstructed_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "decision_avoidable_switch_rate disagrees with its cost/opportunity"
            )

        key = (str(scenario), str(arm), policy_seed, workload_seed)
        if key in cells:
            raise ValueError(f"duplicate formal validation cell {key!r}")
        cells[key] = {
            "scenario": str(scenario),
            "arm": str(arm),
            "policy_seed": policy_seed,
            "workload_seed": workload_seed,
            "delivery_ratio": delivery,
            "decision_avoidable_switches": cost,
            "decision_switch_opportunities": opportunity,
            "decision_forced_switches": forced,
            "decision_forced_switch_cost": forced_cost,
            "decision_avoidable_switch_rate": reported_rate,
        }

    expected_cells = {
        (scenario, arm, policy_seed, workload_seed)
        for scenario in SCENARIOS
        for arm in ARMS
        for policy_seed in POLICY_SEEDS
        for workload_seed in VALIDATION_WORKLOAD_SEEDS
    }
    missing = expected_cells.difference(cells)
    unexpected = set(cells).difference(expected_cells)
    if missing or unexpected:
        raise ValueError(
            "formal validation rows are not an exactly paired complete grid: "
            f"missing examples={sorted(missing)[:3]!r}, "
            f"unexpected examples={sorted(unexpected)[:3]!r}"
        )

    return tuple(
        cells[(scenario, arm, policy_seed, workload_seed)]
        for scenario in SCENARIOS
        for arm in ARMS
        for policy_seed in POLICY_SEEDS
        for workload_seed in VALIDATION_WORKLOAD_SEEDS
    )


def validate_complete_grid(rows: Sequence[Any]) -> dict[str, Any]:
    source_rows = tuple(rows)
    normalized = _normalize_rows(source_rows)
    _full_input_rows(source_rows, normalized)
    return {
        "grid_complete": True,
        "expected_rows": len(normalized),
        "observed_rows": len(normalized),
        "scenario_count": len(SCENARIOS),
        "arm_count": len(ARMS),
        "policy_seed_count": len(POLICY_SEEDS),
        "workload_seed_count": len(VALIDATION_WORKLOAD_SEEDS),
        "all_values_finite": True,
        "all_costs_not_greater_than_opportunities": True,
        "all_forced_decision_constraint_costs_zero": True,
        "episode_rates_recomputed_and_matched": True,
        "pairing": "same_policy_seed_and_gate_workload_seed_in_every_arm",
    }


def _arm_matrices(
    rows: Sequence[Mapping[str, Any]], scenario: str, arm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {
        (row["scenario"], row["arm"], row["policy_seed"], row["workload_seed"]): row
        for row in rows
    }
    delivery = np.empty((len(POLICY_SEEDS), len(VALIDATION_WORKLOAD_SEEDS)))
    cost = np.empty_like(delivery)
    opportunity = np.empty_like(delivery)
    for policy_index, policy_seed in enumerate(POLICY_SEEDS):
        for workload_index, workload_seed in enumerate(VALIDATION_WORKLOAD_SEEDS):
            row = lookup[(scenario, arm, policy_seed, workload_seed)]
            delivery[policy_index, workload_index] = row["delivery_ratio"]
            cost[policy_index, workload_index] = row[
                "decision_avoidable_switches"
            ]
            opportunity[policy_index, workload_index] = row[
                "decision_switch_opportunities"
            ]
    return delivery, cost, opportunity


def _bootstrap_weights(
    rng: np.random.Generator, count: int, size: int
) -> np.ndarray:
    return rng.multinomial(
        count,
        np.full(count, 1.0 / count),
        size=size,
    ).astype(float)


def _crossed_weighted_totals(
    values: np.ndarray, policy_weights: np.ndarray, workload_weights: np.ndarray
) -> np.ndarray:
    return np.einsum(
        "bi,ij,bj->b", policy_weights, values, workload_weights, optimize=True
    )


def _delivery_difference_bootstrap(
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    rng_seed: int,
    resamples: int,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    policy_weights = _bootstrap_weights(rng, treatment.shape[0], resamples)
    workload_weights = _bootstrap_weights(rng, treatment.shape[1], resamples)
    total = _crossed_weighted_totals(
        treatment - reference, policy_weights, workload_weights
    )
    return total / float(treatment.shape[0] * treatment.shape[1])


def _rate_bootstrap(
    cost: np.ndarray,
    opportunity: np.ndarray,
    *,
    rng_seed: int,
    resamples: int,
) -> tuple[np.ndarray, int, int]:
    return _conditional_rate_bootstrap(
        cost,
        opportunity,
        rng_seed=rng_seed,
        resamples=resamples,
    )


def _rate_difference_bootstrap(
    treatment_cost: np.ndarray,
    treatment_opportunity: np.ndarray,
    reference_cost: np.ndarray,
    reference_opportunity: np.ndarray,
    *,
    rng_seed: int,
    resamples: int,
) -> tuple[np.ndarray, int, int]:
    return _conditional_rate_bootstrap(
        treatment_cost,
        treatment_opportunity,
        reference_cost=reference_cost,
        reference_opportunity=reference_opportunity,
        rng_seed=rng_seed,
        resamples=resamples,
    )


def _conditional_rate_bootstrap(
    treatment_cost: np.ndarray,
    treatment_opportunity: np.ndarray,
    *,
    rng_seed: int,
    resamples: int,
    reference_cost: np.ndarray | None = None,
    reference_opportunity: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Draw defined rate replicates, rejecting zero-denominator resamples."""

    paired = reference_cost is not None or reference_opportunity is not None
    if paired and (reference_cost is None or reference_opportunity is None):
        raise ValueError("paired rate bootstrap requires both reference matrices")
    rng = np.random.default_rng(rng_seed)
    output = np.empty(resamples, dtype=float)
    accepted = 0
    rejected = 0
    draws = 0
    maximum_draws = max(
        BOOTSTRAP_MINIMUM_MAXIMUM_DRAWS,
        BOOTSTRAP_MAXIMUM_DRAW_MULTIPLIER * resamples,
    )

    while accepted < resamples:
        remaining = resamples - accepted
        draw_count = min(max(256, 2 * remaining), 8192)
        draw_count = min(draw_count, maximum_draws - draws)
        if draw_count <= 0:
            raise ValueError(
                "could not obtain enough positive-opportunity bootstrap replicates"
            )
        policy_weights = _bootstrap_weights(
            rng, treatment_cost.shape[0], draw_count
        )
        workload_weights = _bootstrap_weights(
            rng, treatment_cost.shape[1], draw_count
        )

        treatment_numerator = workload_weights @ treatment_cost.T
        treatment_denominator = workload_weights @ treatment_opportunity.T
        treatment_valid = ~np.any(
            (treatment_denominator <= 0.0) & (policy_weights > 0.0), axis=1
        )
        treatment_seed_rates = np.divide(
            treatment_numerator,
            treatment_denominator,
            out=np.zeros_like(treatment_numerator),
            where=treatment_denominator > 0.0,
        )
        values = np.sum(
            policy_weights * treatment_seed_rates, axis=1
        ) / treatment_cost.shape[0]
        valid = treatment_valid

        if paired:
            assert reference_cost is not None
            assert reference_opportunity is not None
            reference_numerator = workload_weights @ reference_cost.T
            reference_denominator = workload_weights @ reference_opportunity.T
            reference_valid = ~np.any(
                (reference_denominator <= 0.0) & (policy_weights > 0.0), axis=1
            )
            reference_seed_rates = np.divide(
                reference_numerator,
                reference_denominator,
                out=np.zeros_like(reference_numerator),
                where=reference_denominator > 0.0,
            )
            reference_values = np.sum(
                policy_weights * reference_seed_rates, axis=1
            ) / treatment_cost.shape[0]
            values = values - reference_values
            valid = valid & reference_valid

        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) >= remaining:
            cutoff = int(valid_indices[remaining - 1]) + 1
            prefix_valid = valid[:cutoff]
            output[accepted:] = values[:cutoff][prefix_valid]
            rejected += int(cutoff - remaining)
            draws += cutoff
            accepted = resamples
        else:
            output[accepted : accepted + len(valid_indices)] = values[valid]
            accepted += len(valid_indices)
            rejected += int(draw_count - len(valid_indices))
            draws += draw_count

    return output, rejected, draws


def _ratio_of_sums(cost: np.ndarray, opportunity: np.ndarray) -> float:
    denominator = float(opportunity.sum())
    if denominator <= 0.0:
        raise ValueError("decision-rate estimand has zero opportunity")
    return float(cost.sum() / denominator)


def _policy_seed_rates(cost: np.ndarray, opportunity: np.ndarray) -> np.ndarray:
    denominator = opportunity.sum(axis=1)
    if np.any(denominator <= 0.0):
        raise ValueError("a policy-seed decision-rate estimand has zero opportunity")
    return cost.sum(axis=1) / denominator


def _mean_policy_seed_rate(cost: np.ndarray, opportunity: np.ndarray) -> float:
    return float(_policy_seed_rates(cost, opportunity).mean())


def _exact_sign_flip(values: np.ndarray, method: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(POLICY_SEEDS):
        raise ValueError("sign-flip input must contain one value per policy seed")
    if not np.isfinite(values).all():
        raise ValueError("sign-flip values must be finite")
    observed = abs(float(values.mean()))
    assignments = np.arange(1 << len(values), dtype=np.uint64)[:, None]
    bit_positions = np.arange(len(values), dtype=np.uint64)
    signs = 2.0 * ((assignments >> bit_positions) & 1).astype(float) - 1.0
    randomized = np.abs((signs @ values) / len(values))
    scale = max(
        observed,
        float(np.max(np.abs(randomized))),
        np.finfo(float).tiny,
    )
    tolerance = np.finfo(float).eps * scale * 8.0
    extreme = int(np.count_nonzero(randomized >= observed - tolerance))
    ties = int(np.count_nonzero(np.abs(randomized - observed) <= tolerance))
    return {
        "raw_p_value": float(extreme / len(randomized)),
        "sign_flip_statistic": observed,
        "sign_flip_permutations": int(len(randomized)),
        "sign_flip_tail_tie_permutations": ties,
        "sign_flip_zero_difference_count": int(np.count_nonzero(values == 0.0)),
        "sign_flip_tail_ties_included": True,
        "p_value_method": method,
    }


def _holm_adjusted(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("Holm inputs must be finite p-values")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    adjusted_ordered = np.maximum.accumulate(
        ordered * np.arange(len(values), 0, -1, dtype=float)
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def _two_sided_interval(samples: np.ndarray) -> tuple[float, float]:
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)


def _delivery_effect(
    scenario: str,
    treatment_arm: str,
    reference_arm: str,
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    rng_seed: int,
    resamples: int,
    role: str,
) -> dict[str, Any]:
    bootstrap = _delivery_difference_bootstrap(
        treatment, reference, rng_seed=rng_seed, resamples=resamples
    )
    low, high = _two_sided_interval(bootstrap)
    one_sided_lower = float(
        np.quantile(bootstrap, 1.0 - CONFIDENCE_LEVEL)
    )
    seed_differences = (treatment - reference).mean(axis=1)
    return {
        "scenario": scenario,
        "endpoint": "delivery_ratio_difference",
        "treatment_arm": treatment_arm,
        "reference_arm": reference_arm,
        "role": role,
        "estimand": "equal_policy_and_workload_mean_paired_difference",
        "treatment_mean": float(treatment.mean()),
        "reference_mean": float(reference.mean()),
        "estimate": float((treatment - reference).mean()),
        "ci95_low": low,
        "ci95_high": high,
        "one_sided_95_lower_bound": one_sided_lower,
        "policy_seed_differences": [float(value) for value in seed_differences],
        "policy_seed_ids": list(POLICY_SEEDS),
        "policy_seed_count": len(POLICY_SEEDS),
        "workload_seed_count": len(VALIDATION_WORKLOAD_SEEDS),
        "bootstrap_rng_seed": rng_seed,
        "bootstrap_resamples": resamples,
        "ci_method": "crossed_pigeonhole_bootstrap_percentile",
        **_exact_sign_flip(
            seed_differences,
            "two_sided_exact_policy_seed_sign_flip_on_workload_means",
        ),
    }


def _rate_effect(
    scenario: str,
    treatment_arm: str,
    reference_arm: str,
    treatment_cost: np.ndarray,
    treatment_opportunity: np.ndarray,
    reference_cost: np.ndarray,
    reference_opportunity: np.ndarray,
    *,
    rng_seed: int,
    resamples: int,
    role: str,
) -> dict[str, Any]:
    bootstrap, rejected, total_draws = _rate_difference_bootstrap(
        treatment_cost,
        treatment_opportunity,
        reference_cost,
        reference_opportunity,
        rng_seed=rng_seed,
        resamples=resamples,
    )
    low, high = _two_sided_interval(bootstrap)
    one_sided_upper = float(np.quantile(bootstrap, CONFIDENCE_LEVEL))
    treatment_seed_rates = _policy_seed_rates(
        treatment_cost, treatment_opportunity
    )
    reference_seed_rates = _policy_seed_rates(reference_cost, reference_opportunity)
    seed_differences = treatment_seed_rates - reference_seed_rates
    treatment_rate = _mean_policy_seed_rate(
        treatment_cost, treatment_opportunity
    )
    reference_rate = _mean_policy_seed_rate(reference_cost, reference_opportunity)
    return {
        "scenario": scenario,
        "endpoint": "decision_avoidable_switch_rate_difference",
        "treatment_arm": treatment_arm,
        "reference_arm": reference_arm,
        "role": role,
        "estimand": (
            "difference_of_equal_weight_policy_seed_rates_within_seed_ratio_of_sums"
        ),
        "treatment_rate": treatment_rate,
        "reference_rate": reference_rate,
        "estimate": treatment_rate - reference_rate,
        "ci95_low": low,
        "ci95_high": high,
        "one_sided_95_upper_bound": one_sided_upper,
        "policy_seed_differences": [float(value) for value in seed_differences],
        "policy_seed_ids": list(POLICY_SEEDS),
        "policy_seed_count": len(POLICY_SEEDS),
        "workload_seed_count": len(VALIDATION_WORKLOAD_SEEDS),
        "bootstrap_rng_seed": rng_seed,
        "bootstrap_resamples": resamples,
        "bootstrap_total_draws": total_draws,
        "bootstrap_zero_denominator_draws_rejected": rejected,
        "ci_method": (
            "crossed_pigeonhole_bootstrap_recomputing_ratio_of_sums"
        ),
        **_exact_sign_flip(
            seed_differences,
            "two_sided_exact_policy_seed_sign_flip_on_ratio_of_sums_differences",
        ),
    }


def _arm_summary(
    scenario: str,
    arm: str,
    delivery: np.ndarray,
    cost: np.ndarray,
    opportunity: np.ndarray,
    *,
    rng_seed: int,
    resamples: int,
) -> dict[str, Any]:
    zeros = np.zeros_like(delivery)
    delivery_bootstrap = _delivery_difference_bootstrap(
        delivery, zeros, rng_seed=rng_seed, resamples=resamples
    )
    rate_bootstrap, rate_rejected, rate_total_draws = _rate_bootstrap(
        cost, opportunity, rng_seed=rng_seed + 1, resamples=resamples
    )
    delivery_low, delivery_high = _two_sided_interval(delivery_bootstrap)
    rate_low, rate_high = _two_sided_interval(rate_bootstrap)
    return {
        "scenario": scenario,
        "arm": arm,
        "role": "descriptive_independent_validation_gate_only",
        "delivery_ratio": float(delivery.mean()),
        "delivery_ci95_low": delivery_low,
        "delivery_ci95_high": delivery_high,
        "decision_avoidable_switches": int(cost.sum()),
        "decision_switch_opportunities": int(opportunity.sum()),
        "decision_avoidable_switch_rate": _mean_policy_seed_rate(
            cost, opportunity
        ),
        "pooled_decision_count_rate_for_audit_only": _ratio_of_sums(
            cost, opportunity
        ),
        "decision_rate_ci95_low": rate_low,
        "decision_rate_ci95_high": rate_high,
        "rate_aggregation": "equal_weight_mean_of_policy_seed_ratio_of_sums",
        "bootstrap_rng_seeds": [rng_seed, rng_seed + 1],
        "bootstrap_resamples": resamples,
        "rate_bootstrap_total_draws": rate_total_draws,
        "rate_bootstrap_zero_denominator_draws_rejected": rate_rejected,
    }


def _analyze_validation_rows(
    rows: Sequence[Any], *, bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Implementation hook; the public formal entry fixes 5,000 resamples."""

    if isinstance(bootstrap_resamples, bool) or not isinstance(
        bootstrap_resamples, Integral
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    bootstrap_resamples = int(bootstrap_resamples)
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")

    source_rows = tuple(rows)
    normalized = _normalize_rows(source_rows)
    full_input_rows = _full_input_rows(source_rows, normalized)
    matrices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        (scenario, arm): _arm_matrices(normalized, scenario, arm)
        for scenario in SCENARIOS
        for arm in ARMS
    }

    descriptive: list[dict[str, Any]] = []
    primary_differences: list[dict[str, Any]] = []
    constrained_rates: list[dict[str, Any]] = []
    validation_gates: list[dict[str, Any]] = []
    secondary_effects: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        scenario_seed = PRIMARY_BOOTSTRAP_SEED_BASE + 100 * scenario_index
        baseline = matrices[(scenario, ARM_QOS_ONLY_BASELINE)]
        constrained = matrices[(scenario, ARM_QOS_ONLY_CONSTRAINED)]

        delivery_difference = _delivery_effect(
            scenario,
            ARM_QOS_ONLY_CONSTRAINED,
            ARM_QOS_ONLY_BASELINE,
            constrained[0],
            baseline[0],
            rng_seed=scenario_seed,
            resamples=bootstrap_resamples,
            role="primary_independent_validation_gate_difference",
        )
        delivery_difference["noninferiority_margin"] = (
            DELIVERY_NONINFERIORITY_MARGIN
        )
        delivery_difference["noninferiority_threshold"] = (
            -DELIVERY_NONINFERIORITY_MARGIN
        )
        delivery_difference["one_sided_lower_meets_noninferiority"] = (
            delivery_difference["one_sided_95_lower_bound"]
            >= -DELIVERY_NONINFERIORITY_MARGIN
        )
        primary_differences.append(delivery_difference)

        rate_difference = _rate_effect(
            scenario,
            ARM_QOS_ONLY_CONSTRAINED,
            ARM_QOS_ONLY_BASELINE,
            constrained[1],
            constrained[2],
            baseline[1],
            baseline[2],
            rng_seed=scenario_seed + 1,
            resamples=bootstrap_resamples,
            role="primary_independent_validation_gate_difference",
        )
        rate_difference["superiority_threshold"] = 0.0
        rate_difference["one_sided_upper_supports_reduction"] = (
            rate_difference["one_sided_95_upper_bound"] < 0.0
        )
        primary_differences.append(rate_difference)

        (
            constrained_bootstrap,
            constrained_rejected,
            constrained_total_draws,
        ) = _rate_bootstrap(
            constrained[1],
            constrained[2],
            rng_seed=scenario_seed + 2,
            resamples=bootstrap_resamples,
        )
        rate_low, rate_high = _two_sided_interval(constrained_bootstrap)
        upper = float(np.quantile(constrained_bootstrap, CONFIDENCE_LEVEL))
        constrained_seed_rates = _policy_seed_rates(constrained[1], constrained[2])
        constrained_rate = float(constrained_seed_rates.mean())
        constrained_record = {
            "scenario": scenario,
            "arm": ARM_QOS_ONLY_CONSTRAINED,
            "role": "primary_independent_validation_gate_budget_assessment",
            "estimand": (
                "equal_weight_mean_policy_seed_rate_within_seed_ratio_of_sums"
            ),
            "decision_avoidable_switches": int(constrained[1].sum()),
            "decision_switch_opportunities": int(constrained[2].sum()),
            "estimate": constrained_rate,
            "pooled_decision_count_rate_for_audit_only": _ratio_of_sums(
                constrained[1], constrained[2]
            ),
            "policy_seed_ids": list(POLICY_SEEDS),
            "policy_seed_rates": [
                float(value) for value in constrained_seed_rates
            ],
            "budget": SWITCH_BUDGET,
            "point_estimate_within_budget": (
                constrained_rate <= SWITCH_BUDGET
            ),
            "ci95_low": rate_low,
            "ci95_high": rate_high,
            "one_sided_95_upper_bound": upper,
            "upper_bound_within_budget": (
                upper <= SWITCH_BUDGET
            ),
            "maximum_policy_seed_rate": float(constrained_seed_rates.max()),
            "every_policy_seed_rate_within_budget": bool(
                np.all(constrained_seed_rates <= SWITCH_BUDGET)
            ),
            "one_sided_confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_rng_seed": scenario_seed + 2,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_total_draws": constrained_total_draws,
            "bootstrap_zero_denominator_draws_rejected": constrained_rejected,
            "ci_method": (
                "crossed_pigeonhole_bootstrap_recomputing_ratio_of_sums"
            ),
        }
        constrained_rates.append(constrained_record)

        validation_gates.extend(
            (
                {
                    "scenario": scenario,
                    "gate": "delivery_noninferiority_one_sided_95_lower",
                    "value": delivery_difference["one_sided_95_lower_bound"],
                    "operator": ">=",
                    "threshold": -DELIVERY_NONINFERIORITY_MARGIN,
                    "passed": delivery_difference[
                        "one_sided_lower_meets_noninferiority"
                    ],
                },
                {
                    "scenario": scenario,
                    "gate": "avoidable_rate_reduction_one_sided_95_upper",
                    "value": rate_difference["one_sided_95_upper_bound"],
                    "operator": "<",
                    "threshold": 0.0,
                    "passed": rate_difference[
                        "one_sided_upper_supports_reduction"
                    ],
                },
                {
                    "scenario": scenario,
                    "gate": "constrained_rate_one_sided_95_upper",
                    "value": constrained_record["one_sided_95_upper_bound"],
                    "operator": "<=",
                    "threshold": SWITCH_BUDGET,
                    "passed": constrained_record["upper_bound_within_budget"],
                },
                {
                    "scenario": scenario,
                    "gate": "every_constrained_policy_seed_rate",
                    "value": constrained_record["maximum_policy_seed_rate"],
                    "operator": "<=",
                    "threshold": SWITCH_BUDGET,
                    "passed": constrained_record[
                        "every_policy_seed_rate_within_budget"
                    ],
                },
            )
        )

        for arm_index, arm in enumerate(ARMS):
            descriptive.append(
                _arm_summary(
                    scenario,
                    arm,
                    *matrices[(scenario, arm)],
                    rng_seed=(
                        SECONDARY_BOOTSTRAP_SEED_BASE
                        + 100 * scenario_index
                        + 10 * arm_index
                    ),
                    resamples=bootstrap_resamples,
                )
            )

        secondary_contrasts = (
            (
                "reward_shaped_control_minus_qos_only_baseline",
                ARM_REWARD_SHAPED_CONTROL,
                ARM_QOS_ONLY_BASELINE,
            ),
            (
                "qos_only_constrained_minus_reward_shaped_control",
                ARM_QOS_ONLY_CONSTRAINED,
                ARM_REWARD_SHAPED_CONTROL,
            ),
        )
        for contrast_index, (contrast, treatment_arm, reference_arm) in enumerate(
            secondary_contrasts
        ):
            treatment = matrices[(scenario, treatment_arm)]
            reference = matrices[(scenario, reference_arm)]
            seed = (
                SECONDARY_BOOTSTRAP_SEED_BASE
                + 1000
                + 100 * scenario_index
                + 10 * contrast_index
            )
            delivery_effect = _delivery_effect(
                scenario,
                treatment_arm,
                reference_arm,
                treatment[0],
                reference[0],
                rng_seed=seed,
                resamples=bootstrap_resamples,
                role="secondary_descriptive_unadjusted",
            )
            delivery_effect["contrast"] = contrast
            rate_effect = _rate_effect(
                scenario,
                treatment_arm,
                reference_arm,
                treatment[1],
                treatment[2],
                reference[1],
                reference[2],
                rng_seed=seed + 1,
                resamples=bootstrap_resamples,
                role="secondary_descriptive_unadjusted",
            )
            rate_effect["contrast"] = contrast
            secondary_effects.extend((delivery_effect, rate_effect))

    adjusted = _holm_adjusted(
        [record["raw_p_value"] for record in primary_differences]
    )
    for record, adjusted_p in zip(primary_differences, adjusted):
        record["holm_adjusted_p"] = float(adjusted_p)
        record["multiplicity_family"] = (
            "four_constrained_vs_qos_only_baseline_validation_differences"
        )
        record["multiplicity_family_size"] = len(primary_differences)

    core_rows = [dict(row) for row in normalized]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "analysis_scope": "formal_independent_validation_gate_only",
        "inferential_status": (
            "independent_validation_gate_only_not_test_or_paper_evidence"
        ),
        "validation_only": True,
        "independent_validation_gate": True,
        "checkpoint_selection_rows_included": False,
        "final_test_evidence": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "sealed_test_access_count": 0,
        "sealed_test_access": 0,
        "test_access_count": 0,
        "test_panel_consulted": False,
        "sealed_test_workloads_instantiated": False,
        "analysis_input_sha256": sha256_json(full_input_rows),
        "analysis_input_core_sha256": sha256_json(core_rows),
        "grid_audit": {
            "grid_complete": True,
            "expected_rows": len(core_rows),
            "observed_rows": len(core_rows),
            "scenarios": list(SCENARIOS),
            "arms": list(ARMS),
            "policy_seeds": list(POLICY_SEEDS),
            "selection_workload_seeds_excluded": list(SELECTION_WORKLOAD_SEEDS),
            "gate_workload_seeds": list(GATE_WORKLOAD_SEEDS),
            "all_values_finite": True,
            "all_costs_not_greater_than_opportunities": True,
            "all_forced_decision_constraint_costs_zero": True,
            "episode_rates_recomputed_and_matched": True,
            "complete_pairing_verified": True,
        },
        "statistical_contract": {
            "algorithm_replication_unit": "independently_trained_policy_seed",
            "workloads": (
                "independent_gate_workloads_paired_as_common_random_numbers"
            ),
            "checkpoint_selection_workloads_included": False,
            "episode_rows_treated_as_independent": False,
            "bootstrap": {
                "method": "crossed_pigeonhole_bootstrap",
                "resamples": bootstrap_resamples,
                "maximum_total_draws": max(
                    BOOTSTRAP_MINIMUM_MAXIMUM_DRAWS,
                    BOOTSTRAP_MAXIMUM_DRAW_MULTIPLIER * bootstrap_resamples,
                ),
                "insufficient_defined_draws": "fail_closed",
                "confidence_level": CONFIDENCE_LEVEL,
                "rate_reduction": "recompute_ratio_of_sums_in_every_resample",
                "between_policy_seed_weighting": "equal",
                "zero_opportunity_resamples": (
                    "deterministically_reject_and_redraw_until_resample_count"
                ),
            },
            "hypothesis_test": (
                "two_sided_exact_sign_flip_on_eight_policy_seed_effects"
            ),
            "primary_multiplicity": "Holm_FWER_over_four_paired_differences",
            "reward_shaped_control_role": "secondary_descriptive_only",
        },
        "primary": {
            "paired_differences": primary_differences,
            "constrained_rates": constrained_rates,
        },
        "validation_gates": validation_gates,
        "validation_gate_scope": (
            "independent_gate_performance_only_runner_must_combine_integrity_and_selection_gates"
        ),
        "validation_gate_passed": all(
            bool(record["passed"]) for record in validation_gates
        ),
        "descriptive_arm_summaries": descriptive,
        "secondary_reward_shaped_control": {
            "confirmatory": False,
            "multiplicity_adjustment": None,
            "paired_effects": secondary_effects,
        },
    }
    report = dict(body)
    report["report_sha256"] = sha256_json(body)
    return report


def analyze_validation_rows(rows: Sequence[Any]) -> dict[str, Any]:
    """Run the frozen 5,000-resample independent validation-gate analysis."""

    return _analyze_validation_rows(
        rows,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )


__all__ = [
    "ARMS",
    "ARM_QOS_ONLY_BASELINE",
    "ARM_QOS_ONLY_CONSTRAINED",
    "ARM_REWARD_SHAPED_CONTROL",
    "BOOTSTRAP_MAXIMUM_DRAW_MULTIPLIER",
    "BOOTSTRAP_MINIMUM_MAXIMUM_DRAWS",
    "BOOTSTRAP_RESAMPLES",
    "DELIVERY_NONINFERIORITY_MARGIN",
    "EXPECTED_GATE_ROW_COUNT",
    "GATE_WORKLOAD_SEEDS",
    "POLICY_SEEDS",
    "POLICY_SEED_NAMESPACE",
    "SCENARIOS",
    "SELECTION_WORKLOAD_SEEDS",
    "STUDY_NAME",
    "SWITCH_BUDGET",
    "TRAIN_WORKLOAD_SEEDS",
    "VALIDATION_WORKLOAD_SEEDS",
    "analyze_validation_rows",
    "canonical_json_bytes",
    "sha256_json",
    "validate_complete_grid",
    "validate_report_hash",
]
