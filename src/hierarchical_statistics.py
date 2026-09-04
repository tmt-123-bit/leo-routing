"""Statistics for crossed policy-seed and workload-seed experiments.

The evaluation design is crossed rather than nested: every trained policy seed
is evaluated on the same held-out workload seeds.  Episode rows therefore are
not independent replicates.  This module keeps the two sampling axes explicit:

* confidence intervals use a crossed (pigeonhole) bootstrap that independently
  resamples whole policy-seed rows and whole workload-seed columns; and
* confirmatory tests first average paired differences over workloads, then use
  an exact sign-flip test across independently trained policy seeds.

The latter makes the training run, not the episode, the unit of algorithmic
replication.  The workload panel remains paired common-random-number data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


DEFAULT_BOOTSTRAP_RESAMPLES = 5000
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Frozen indexing contract for the confirmatory 50k ablation.  These orders are
# recorded in every statistical manifest because the finite Monte Carlo
# bootstrap is reproducible only when both the RNG seed and matrix indexing are
# specified.
ABLATION_BOOTSTRAP_RNG_SEED_FORMULA = "18000 + 1000*s + 100*c + m"
ABLATION_SCENARIO_ORDER = (
    "low_load",
    "medium_load",
    "hotspot_high_load",
    "frequent_break",
    "fault_links",
)
ABLATION_CONTRAST_ORDER = (
    "remove_queue_mechanism_package",
    "remove_centered_local_credit",
    "remove_packet_context",
    "replace_graph_critic_with_flat_critic",
    "remove_ppo_protection_package",
    "add_lifetime_feature",
    "add_lifetime_reward",
    "add_hard_lifetime_mask",
)
ABLATION_ENDPOINT_ORDER = (
    "delivery_ratio",
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "routing_switches",
    "global_control_overhead_ratio",
)


@dataclass(frozen=True)
class CrossedMatrix:
    """A complete policy-seed by workload-seed response matrix."""

    policy_seeds: tuple[Any, ...]
    workload_seeds: tuple[Any, ...]
    values: np.ndarray


@dataclass(frozen=True)
class PairedCrossedMatrices:
    """Aligned treatment, reference, and treatment-minus-reference matrices."""

    policy_seeds: tuple[Any, ...]
    workload_seeds: tuple[Any, ...]
    treatment: np.ndarray
    reference: np.ndarray
    difference: np.ndarray
    policy_seed_pairing: str


def _field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _stable_sorted(values: Iterable[Any]) -> tuple[Any, ...]:
    values = tuple(values)
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(
            sorted(values, key=lambda value: (type(value).__name__, repr(value)))
        )


def build_crossed_matrix(rows: Sequence[Any], metric: str) -> CrossedMatrix:
    """Build and validate a complete crossed response matrix.

    Missing cells, duplicate cells, and non-finite responses are rejected.  A
    silent complete-case analysis could otherwise change the estimand or give
    unusually successful runs more weight.
    """

    if not rows:
        raise ValueError(f"cannot build {metric!r} matrix from zero rows")

    cells: dict[tuple[Any, Any], float] = {}
    for row in rows:
        policy_seed = _field(row, "policy_seed")
        workload_seed = _field(row, "workload_seed")
        key = (policy_seed, workload_seed)
        if key in cells:
            raise ValueError(
                f"duplicate policy/workload cell {key!r} for metric {metric!r}"
            )
        value = float(_field(row, metric))
        if not np.isfinite(value):
            raise ValueError(f"non-finite {metric!r} value in cell {key!r}")
        cells[key] = value

    policy_seeds = _stable_sorted(key[0] for key in cells)
    workload_seeds = _stable_sorted(key[1] for key in cells)
    # The generators above contain repetitions, so retain unique levels.
    policy_seeds = _stable_sorted(set(policy_seeds))
    workload_seeds = _stable_sorted(set(workload_seeds))

    expected = {
        (policy_seed, workload_seed)
        for policy_seed in policy_seeds
        for workload_seed in workload_seeds
    }
    missing = expected.difference(cells)
    unexpected = set(cells).difference(expected)
    if missing or unexpected:
        examples = _stable_sorted(missing)[:5]
        raise ValueError(
            f"incomplete crossed design for {metric!r}: "
            f"expected {len(expected)} cells, observed {len(cells)}, "
            f"missing examples={examples!r}"
        )

    values = np.asarray(
        [
            [cells[(policy_seed, workload_seed)] for workload_seed in workload_seeds]
            for policy_seed in policy_seeds
        ],
        dtype=float,
    )
    return CrossedMatrix(policy_seeds, workload_seeds, values)


def pair_crossed_matrices(
    treatment_rows: Sequence[Any],
    reference_rows: Sequence[Any],
    metric: str,
) -> PairedCrossedMatrices:
    """Align paired observations without treating repeated episodes as IID.

    Learned references are paired by identical policy seed.  A deterministic
    reference, represented by one sentinel policy seed, is broadcast across
    treatment policy seeds while preserving workload pairing.
    """

    treatment = build_crossed_matrix(treatment_rows, metric)
    reference = build_crossed_matrix(reference_rows, metric)
    if treatment.workload_seeds != reference.workload_seeds:
        missing_from_reference = set(treatment.workload_seeds).difference(
            reference.workload_seeds
        )
        missing_from_treatment = set(reference.workload_seeds).difference(
            treatment.workload_seeds
        )
        raise ValueError(
            "treatment and reference must use exactly the same workload seeds; "
            f"missing from reference={_stable_sorted(missing_from_reference)!r}, "
            f"missing from treatment={_stable_sorted(missing_from_treatment)!r}"
        )

    if treatment.policy_seeds == reference.policy_seeds:
        policy_seeds = treatment.policy_seeds
        treatment_values = treatment.values
        reference_values = reference.values
        pairing = "matched_policy_seed"
    elif len(reference.policy_seeds) == 1:
        policy_seeds = treatment.policy_seeds
        treatment_values = treatment.values
        reference_values = np.repeat(
            reference.values, len(policy_seeds), axis=0
        )
        pairing = "deterministic_reference_broadcast"
    elif len(treatment.policy_seeds) == 1:
        policy_seeds = reference.policy_seeds
        treatment_values = np.repeat(
            treatment.values, len(policy_seeds), axis=0
        )
        reference_values = reference.values
        pairing = "deterministic_treatment_broadcast"
    else:
        raise ValueError(
            "policy seeds must match exactly unless one side is deterministic: "
            f"treatment={treatment.policy_seeds!r}, "
            f"reference={reference.policy_seeds!r}"
        )

    return PairedCrossedMatrices(
        policy_seeds=policy_seeds,
        workload_seeds=treatment.workload_seeds,
        treatment=treatment_values,
        reference=reference_values,
        difference=treatment_values - reference_values,
        policy_seed_pairing=pairing,
    )


def crossed_bootstrap_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> np.ndarray:
    """Return pigeonhole-bootstrap means for a complete crossed matrix."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size:
        raise ValueError("values must be a non-empty two-dimensional matrix")
    if not np.isfinite(values).all():
        raise ValueError("values must all be finite")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    policy_count, workload_count = values.shape
    output = np.empty(resamples, dtype=float)
    batch_size = min(2048, resamples)
    policy_probability = np.full(policy_count, 1.0 / policy_count)
    workload_probability = np.full(workload_count, 1.0 / workload_count)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        size = stop - start
        policy_weights = rng.multinomial(
            policy_count, policy_probability, size=size
        ) / policy_count
        workload_weights = rng.multinomial(
            workload_count, workload_probability, size=size
        ) / workload_count
        policy_weighted = policy_weights @ values
        output[start:stop] = np.sum(policy_weighted * workload_weights, axis=1)
    return output


def crossed_mean_summary(
    matrix: CrossedMatrix,
    *,
    rng_seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Summarize a response with crossed-bootstrap uncertainty."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    bootstrap = crossed_bootstrap_mean(
        matrix.values,
        rng=np.random.default_rng(rng_seed),
        resamples=resamples,
    )
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap, [tail, 1.0 - tail])
    policy_means = matrix.values.mean(axis=1)
    return {
        "n": int(matrix.values.size),
        "policy_seed_count": len(matrix.policy_seeds),
        "workload_seed_count": len(matrix.workload_seeds),
        "mean": float(matrix.values.mean()),
        "std": (
            float(matrix.values.std(ddof=1)) if matrix.values.size > 1 else 0.0
        ),
        "policy_seed_mean_std": (
            float(policy_means.std(ddof=1)) if len(policy_means) > 1 else 0.0
        ),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_standard_error": (
            float(bootstrap.std(ddof=1)) if len(bootstrap) > 1 else 0.0
        ),
        "ci_method": "crossed_pigeonhole_bootstrap_percentile",
        "ci_resamples": resamples,
        "confidence_level": confidence_level,
    }


def exact_sign_flip_test(values: Sequence[float]) -> dict[str, Any]:
    """Two-sided exact randomization test of a paired mean difference.

    This enumerates all sign assignments and is intended for the small number
    of independent training seeds used in the experiments.  It remains exact
    with ties and zero differences, unlike automatic asymptotic fallbacks in
    signed-rank implementations.
    """

    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or not len(sample):
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(sample).all():
        raise ValueError("values must all be finite")
    if len(sample) > 20:
        raise ValueError(
            "exact sign-flip enumeration is limited to 20 independent units"
        )

    observed = abs(float(sample.mean()))
    permutation_count = 1 << len(sample)
    extreme = 0
    tail_ties = 0
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8.0
    bit_positions = np.arange(len(sample), dtype=np.uint64)
    for start in range(0, permutation_count, 65536):
        stop = min(start + 65536, permutation_count)
        assignments = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 2.0 * ((assignments >> bit_positions) & 1).astype(float) - 1.0
        randomized = np.abs((signs @ sample) / len(sample))
        extreme += int(np.count_nonzero(randomized >= observed - tolerance))
        tail_ties += int(
            np.count_nonzero(np.abs(randomized - observed) <= tolerance)
        )
    return {
        "statistic": observed,
        "p_value": float(extreme / permutation_count),
        "permutation_count": permutation_count,
        "tail_tie_permutation_count": tail_ties,
        "zero_difference_count": int(np.count_nonzero(sample == 0.0)),
        "tail_ties_included": True,
        "method": "exact_policy_seed_sign_flip_on_workload_means",
    }


def _rank_biserial_correlation(values: np.ndarray) -> float:
    nonzero = values[values != 0.0]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    signed_rank_sum = float(ranks[nonzero > 0.0].sum() - ranks[nonzero < 0.0].sum())
    return signed_rank_sum / float(ranks.sum())


def _wilcoxon_sensitivity(values: np.ndarray) -> dict[str, Any]:
    zero_method = "wilcox"
    requested_method = "auto"
    nonzero_absolute = np.abs(values[values != 0.0])
    if len(nonzero_absolute):
        _, absolute_counts = np.unique(nonzero_absolute, return_counts=True)
    else:
        absolute_counts = np.asarray([], dtype=int)
    tied_groups = absolute_counts[absolute_counts > 1]
    metadata = {
        "requested_method": requested_method,
        "zero_method": zero_method,
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "absolute_difference_tie_group_count": int(len(tied_groups)),
        "absolute_difference_tied_value_count": int(tied_groups.sum()),
    }
    if np.all(values == 0.0):
        return {"statistic": 0.0, "p_value": 1.0, **metadata}
    result = stats.wilcoxon(
        values,
        zero_method=zero_method,
        correction=False,
        alternative="two-sided",
        method=requested_method,
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        **metadata,
    }


def paired_crossed_summary(
    paired: PairedCrossedMatrices,
    *,
    rng_seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Compute paired estimates, crossed CI, exact test, and effect sizes."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    bootstrap = crossed_bootstrap_mean(
        paired.difference,
        rng=np.random.default_rng(rng_seed),
        resamples=resamples,
    )
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap, [tail, 1.0 - tail])

    policy_seed_differences = paired.difference.mean(axis=1)
    test = exact_sign_flip_test(policy_seed_differences)
    wilcoxon = _wilcoxon_sensitivity(policy_seed_differences)
    policy_std = (
        float(policy_seed_differences.std(ddof=1))
        if len(policy_seed_differences) > 1
        else 0.0
    )
    cohen_dz = None
    hedges_gz = None
    if policy_std > 0.0:
        cohen_dz = float(policy_seed_differences.mean() / policy_std)
        degrees_of_freedom = len(policy_seed_differences) - 1
        correction = (
            1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)
            if degrees_of_freedom >= 1
            else 1.0
        )
        hedges_gz = float(correction * cohen_dz)

    return {
        "policy_seed_count": len(paired.policy_seeds),
        "workload_seed_count": len(paired.workload_seeds),
        "episode_pair_count": int(paired.difference.size),
        "bootstrap_rng_seed": int(rng_seed),
        "policy_seed_ids": [
            seed.item() if isinstance(seed, np.generic) else seed
            for seed in paired.policy_seeds
        ],
        "policy_seed_mean_differences": [
            float(value) for value in policy_seed_differences
        ],
        "policy_seed_pairing": paired.policy_seed_pairing,
        "treatment_mean": float(paired.treatment.mean()),
        "reference_mean": float(paired.reference.mean()),
        "mean_difference": float(paired.difference.mean()),
        "median_policy_seed_mean_difference": float(
            np.median(policy_seed_differences)
        ),
        "policy_seed_mean_difference_std": policy_std,
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "bootstrap_standard_error": (
            float(bootstrap.std(ddof=1)) if len(bootstrap) > 1 else 0.0
        ),
        "difference_ci_method": "crossed_pigeonhole_bootstrap_percentile",
        "ci_resamples": resamples,
        "confidence_level": confidence_level,
        "paired_cohen_dz": cohen_dz,
        "paired_hedges_gz": hedges_gz,
        "rank_biserial_correlation": _rank_biserial_correlation(
            policy_seed_differences
        ),
        "positive_policy_seed_fraction": float(
            np.mean(policy_seed_differences > 0.0)
        ),
        "sign_flip_statistic": test["statistic"],
        "sign_flip_permutations": test["permutation_count"],
        "sign_flip_tail_tie_permutations": test[
            "tail_tie_permutation_count"
        ],
        "sign_flip_zero_difference_count": test["zero_difference_count"],
        "sign_flip_tail_ties_included": test["tail_ties_included"],
        "raw_p_value": test["p_value"],
        "p_value_method": test["method"],
        "seed_level_wilcoxon_statistic": wilcoxon["statistic"],
        "seed_level_wilcoxon_p": wilcoxon["p_value"],
        "seed_level_wilcoxon_requested_method": wilcoxon[
            "requested_method"
        ],
        "seed_level_wilcoxon_zero_method": wilcoxon["zero_method"],
        "seed_level_wilcoxon_zero_count": wilcoxon["zero_count"],
        "seed_level_wilcoxon_absolute_tie_group_count": wilcoxon[
            "absolute_difference_tie_group_count"
        ],
        "seed_level_wilcoxon_absolute_tied_value_count": wilcoxon[
            "absolute_difference_tied_value_count"
        ],
        "seed_level_wilcoxon_method": (
            "scipy_wilcoxon_two_sided_on_workload_averaged_policy_seeds"
        ),
    }


def _holm_adjusted(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    adjusted_ordered = np.maximum.accumulate(
        ordered * np.arange(count, 0, -1, dtype=float)
    )
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def _benjamini_hochberg_adjusted(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    adjusted_ordered = np.minimum.accumulate(
        (ordered * count / np.arange(1, count + 1, dtype=float))[::-1]
    )[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def add_multiplicity_corrections(
    rows: list[dict[str, Any]],
    *,
    p_value_key: str = "raw_p_value",
    family_keys: Sequence[str] = (),
    holm_key: str = "holm_adjusted_p",
    benjamini_hochberg_key: str = "benjamini_hochberg_p",
    family_size_key: str = "multiplicity_family_size",
) -> list[dict[str, Any]]:
    """Add Holm-FWER and BH-FDR adjusted p-values within explicit families."""

    families: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(rows):
        family = tuple(row[key] for key in family_keys)
        families.setdefault(family, []).append(index)

    for indices in families.values():
        p_values = np.asarray(
            [float(rows[index][p_value_key]) for index in indices], dtype=float
        )
        if (
            not np.isfinite(p_values).all()
            or np.any(p_values < 0.0)
            or np.any(p_values > 1.0)
        ):
            raise ValueError("p-values must be finite and lie in [0, 1]")
        holm = _holm_adjusted(p_values)
        bh = _benjamini_hochberg_adjusted(p_values)
        for local_index, row_index in enumerate(indices):
            rows[row_index][holm_key] = float(holm[local_index])
            rows[row_index][benjamini_hochberg_key] = float(bh[local_index])
            rows[row_index][family_size_key] = len(indices)
    return rows


def statistical_analysis_manifest(
    *, resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Machine-readable declaration of the inferential design."""

    return {
        "design": "complete_crossed_policy_seed_by_workload_seed",
        "algorithm_replication_unit": "independently_trained_policy_seed",
        "workloads": "paired_common_random_numbers_repeated_within_policy_seed",
        "confidence_interval": {
            "method": "crossed_pigeonhole_bootstrap_percentile",
            "resamples": resamples,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "resampled_axes": ["policy_seed", "workload_seed"],
            "confirmatory_ablation_rng_contract": {
                "formula": ABLATION_BOOTSTRAP_RNG_SEED_FORMULA,
                "scenario_order": list(ABLATION_SCENARIO_ORDER),
                "contrast_order": list(ABLATION_CONTRAST_ORDER),
                "endpoint_order": list(ABLATION_ENDPOINT_ORDER),
                "indices_are_zero_based": True,
                "matrix_axis_ordering": (
                    "stable ascending policy-seed and workload-seed labels; "
                    "policy-seed order is emitted in every effect row"
                ),
            },
        },
        "hypothesis_test": {
            "method": "exact_policy_seed_sign_flip_on_workload_means",
            "two_sided": True,
            "tail_ties_included": True,
            "episode_rows_treated_as_independent": False,
            "sensitivity_test": (
                "two-sided Wilcoxon signed-rank on the same policy-seed means"
            ),
            "wilcoxon_requested_method": "auto",
            "wilcoxon_zero_method": "wilcox",
            "zero_and_absolute_rank_tie_counts_recorded_per_contrast": True,
        },
        "multiple_comparisons": {
            "primary_outcome": "delivery_ratio",
            "primary_confirmatory_family": (
                "all scenario-by-planned-contrast delivery_ratio tests"
            ),
            "primary_adjustment": "Holm family-wise error rate within metric",
            "secondary_metric_families": (
                "reported separately and not promoted to primary outcomes"
            ),
            "global_sensitivity": (
                "Holm family-wise error rate over every reported test"
            ),
            "global_exploratory": (
                "Benjamini-Hochberg false discovery rate over every reported test"
            ),
        },
        "effect_sizes": [
            "raw paired mean difference with crossed-bootstrap interval",
            "policy-seed-level paired Cohen dz and Hedges gz",
            "policy-seed-level matched-pairs rank-biserial correlation",
        ],
        "assumptions": [
            "training seeds are independent algorithm replications",
            "held-out workload seeds are representative and independent draws",
            "the evaluation grid is complete and all comparisons use paired workloads",
            "signs of policy-seed mean differences are exchangeable under the null",
        ],
    }
