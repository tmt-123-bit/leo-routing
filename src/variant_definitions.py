"""Canonical, composable method definitions for LEO MAPPO experiments.

The canonical configurations are deliberately built from ``proposed`` with
``dataclasses.replace``.  This keeps the paper method and every controlled
variant auditable as a set of explicit feature flags instead of relying on
mutually exclusive string checks scattered through the environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class VariantDefinition:
    """Environment-side components controlled by a named method variant."""

    name: str
    lifetime_feature: bool = False
    lifetime_reward: bool = False
    hard_lifetime_mask: bool = False
    queue_features: bool = True
    queue_reward: bool = True
    centered_local_credit: bool = True
    packet_context: bool = True
    graph_critic: bool = True
    queue_trend_feature: bool = False
    downstream_bottleneck_feature: bool = False
    avoidable_switch_cost_only: bool = False
    switch_reward: bool = True

    def as_dict(self) -> Dict[str, bool | str]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedContrast:
    """A predeclared treatment-minus-reference method comparison."""

    name: str
    family: str
    reference: str
    treatment: str
    changed_flags: Tuple[str, ...]
    component_kind: str = "atomic"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# L0, the adopted method: dynamic link failures remain part of the physical
# environment, but predictive remaining-lifetime signals are not used by the
# policy, local reward, or feasibility mask.
PROPOSED = VariantDefinition(name="proposed")

# Auditable cumulative lifetime chain.  Adjacent definitions add exactly one
# component, so the corresponding comparisons isolate feature, reward, and
# hard-mask effects in that order.
WITH_LIFETIME_FEATURE = replace(
    PROPOSED,
    name="with_lifetime_feature",
    lifetime_feature=True,
)
WITH_LIFETIME_REWARD = replace(
    WITH_LIFETIME_FEATURE,
    name="with_lifetime_reward",
    lifetime_reward=True,
)
WITH_HARD_LIFETIME_MASK = replace(
    WITH_LIFETIME_REWARD,
    name="with_hard_lifetime_mask",
    hard_lifetime_mask=True,
)

# Controlled component removals all inherit the same L0 lifetime semantics as
# the proposed method.  Queue capacity remains a physical feasibility rule;
# ``no_queue`` removes learned queue observations and queue reward terms only.
NO_QUEUE = replace(
    PROPOSED,
    name="no_queue",
    queue_features=False,
    queue_reward=False,
)
NO_CREDIT = replace(
    PROPOSED,
    name="no_credit",
    centered_local_credit=False,
)
NO_PACKET_CONTEXT = replace(
    PROPOSED,
    name="no_packet_context",
    packet_context=False,
)
FLAT_CRITIC = replace(
    PROPOSED,
    name="flat_critic",
    graph_critic=False,
)

# Experimental hotspot candidate.  It is intentionally outside the frozen
# planned contrasts so the completed studies and their method matrix retain
# their original contract.
WITH_CONGESTION_CONTEXT = replace(
    PROPOSED,
    name="with_congestion_context",
    queue_trend_feature=True,
    downstream_bottleneck_feature=True,
)

# Experimental reward-alignment candidate.  It keeps the congestion-context
# observation schema while excluding forced reroutes from both local and team
# switch costs.  Switch accounting itself remains unchanged.
WITH_AVOIDABLE_SWITCH_REWARD = replace(
    WITH_CONGESTION_CONTEXT,
    name="with_avoidable_switch_reward",
    avoidable_switch_cost_only=True,
)

# Primary-objective control for constrained routing studies. Physical route
# changes are still counted, but neither local nor team reward contains a
# switch term; churn is controlled exclusively by the explicit constraint.
QOS_ONLY = replace(
    PROPOSED,
    name="qos_only",
    switch_reward=False,
)

CANONICAL_VARIANTS: Mapping[str, VariantDefinition] = MappingProxyType(
    {
        definition.name: definition
        for definition in (
            PROPOSED,
            WITH_LIFETIME_FEATURE,
            WITH_LIFETIME_REWARD,
            WITH_HARD_LIFETIME_MASK,
            NO_QUEUE,
            NO_CREDIT,
            NO_PACKET_CONTEXT,
            FLAT_CRITIC,
            WITH_CONGESTION_CONTEXT,
            WITH_AVOIDABLE_SWITCH_REWARD,
            QOS_ONLY,
        )
    }
)

# Compatibility is intentionally resolved at the boundary.  Internally the
# environment stores only canonical names.  ``no_ppo_protection`` changes the
# trainer, not the environment, so its environment definition is ``proposed``.
LEGACY_VARIANT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "no_lifetime": "proposed",
        "full": "with_hard_lifetime_mask",
        "no_ppo_protection": "proposed",
    }
)


# Machine-readable analysis contract.  Component removals share the proposed
# reference.  The lifetime ladder uses adjacent references so each contrast
# adds exactly one lifetime mechanism.  ``no_queue`` is intentionally labelled
# as a package because it jointly removes learned queue features and queue
# reward terms while retaining physical queue capacity constraints.
PLANNED_CONTRASTS: Tuple[PlannedContrast, ...] = (
    PlannedContrast(
        name="remove_queue_mechanism_package",
        family="component_removal",
        reference="proposed",
        treatment="no_queue",
        changed_flags=("queue_features", "queue_reward"),
        component_kind="mechanism_package",
    ),
    PlannedContrast(
        name="remove_centered_local_credit",
        family="component_removal",
        reference="proposed",
        treatment="no_credit",
        changed_flags=("centered_local_credit",),
    ),
    PlannedContrast(
        name="remove_packet_context",
        family="component_removal",
        reference="proposed",
        treatment="no_packet_context",
        changed_flags=("packet_context",),
    ),
    PlannedContrast(
        name="replace_graph_critic_with_flat_critic",
        family="component_removal",
        reference="proposed",
        treatment="flat_critic",
        changed_flags=("graph_critic",),
    ),
    PlannedContrast(
        name="remove_ppo_protection_package",
        family="training_safeguard_package",
        reference="proposed",
        treatment="no_ppo_protection",
        changed_flags=(
            "actor_gradient_clipping",
            "target_kl_early_stopping",
            "advantage_normalization",
        ),
        component_kind="trainer_safeguard_package",
    ),
    PlannedContrast(
        name="add_lifetime_feature",
        family="lifetime_ladder",
        reference="proposed",
        treatment="with_lifetime_feature",
        changed_flags=("lifetime_feature",),
    ),
    PlannedContrast(
        name="add_lifetime_reward",
        family="lifetime_ladder",
        reference="with_lifetime_feature",
        treatment="with_lifetime_reward",
        changed_flags=("lifetime_reward",),
    ),
    PlannedContrast(
        name="add_hard_lifetime_mask",
        family="lifetime_ladder",
        reference="with_lifetime_reward",
        treatment="with_hard_lifetime_mask",
        changed_flags=("hard_lifetime_mask",),
    ),
)
PLANNED_CONTRAST_BY_NAME: Mapping[str, PlannedContrast] = MappingProxyType(
    {contrast.name: contrast for contrast in PLANNED_CONTRASTS}
)


def resolve_variant(name: str) -> VariantDefinition:
    """Return a canonical definition, rejecting unknown or misspelled names."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("variant must be a non-empty string")
    requested = name.strip()
    canonical_name = LEGACY_VARIANT_ALIASES.get(requested, requested)
    try:
        return CANONICAL_VARIANTS[canonical_name]
    except KeyError as error:
        supported = sorted(set(CANONICAL_VARIANTS) | set(LEGACY_VARIANT_ALIASES))
        raise ValueError(
            f"unknown LEO variant {requested!r}; expected one of {supported}"
        ) from error


def canonical_variant_name(name: str) -> str:
    return resolve_variant(name).name


def planned_contrasts_manifest() -> list[Dict[str, object]]:
    """Return JSON-serializable planned contrasts for experiment manifests."""

    return [contrast.as_dict() for contrast in PLANNED_CONTRASTS]
