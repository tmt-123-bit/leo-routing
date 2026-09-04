"""Testable MAPPO building blocks used by the revised LEO integration."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


def route_hysteresis_residual_dtype_cap(
    cap: float,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> float:
    """Return the greatest value in ``dtype`` that does not exceed ``cap``."""

    if isinstance(cap, bool) or not isinstance(cap, (int, float)):
        raise ValueError("route hysteresis residual cap must be finite and non-negative")
    if not math.isfinite(float(cap)) or float(cap) < 0.0:
        raise ValueError("route hysteresis residual cap must be finite and non-negative")
    if not dtype.is_floating_point:
        raise ValueError("route hysteresis residual cap dtype must be floating point")
    rounded = torch.as_tensor(float(cap), dtype=dtype, device=device)
    if float(rounded.detach().cpu().item()) > float(cap):
        rounded = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf))
    return float(rounded.detach().cpu().item())


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (
            m_a
            + m_b
            + delta * delta * self.count * batch_count / total
        ) / total
        self.count = total


class SharedCandidateActor(nn.Module):
    """Permutation-equivariant scorer for a set of next-hop candidates."""

    def __init__(
        self,
        candidate_feature_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        route_switch_feature_index: int | None = None,
        route_hysteresis_beta: float = 0.0,
        route_hysteresis_mode: str = "legacy_additive",
        route_urgency_feature_index: int | None = None,
        route_class_2_feature_index: int | None = None,
        route_hysteresis_urgency_relief: float = 0.0,
        route_hysteresis_class_2_relief: float = 0.0,
        route_hysteresis_residual_init: float = 0.0,
        route_hysteresis_residual_cap: float = 0.0,
        route_hysteresis_residual_parameterization: str = "scalar",
    ):
        super().__init__()
        if not math.isfinite(route_hysteresis_beta):
            raise ValueError("route hysteresis beta must be finite")
        if route_hysteresis_beta < 0.0:
            raise ValueError("route hysteresis beta must be non-negative")
        if route_switch_feature_index is not None:
            if type(route_switch_feature_index) is not int:
                raise ValueError("route-switch feature index must be an integer")
            if not 0 <= route_switch_feature_index < candidate_feature_dim:
                raise ValueError(
                    "route-switch feature index is outside candidate schema"
                )
        if route_hysteresis_beta > 0.0 and route_switch_feature_index is None:
            raise ValueError("positive route hysteresis requires a switch feature")
        if route_hysteresis_mode not in {
            "legacy_additive",
            "decoupled_adaptive",
        }:
            raise ValueError("unsupported route hysteresis mode")
        for value, field in (
            (route_hysteresis_urgency_relief, "urgency relief"),
            (route_hysteresis_class_2_relief, "class-2 relief"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"route hysteresis {field} must be in [0, 1]")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"route hysteresis {field} must be in [0, 1]")
        if route_hysteresis_mode == "legacy_additive":
            if route_urgency_feature_index is not None or route_class_2_feature_index is not None:
                raise ValueError("legacy route hysteresis cannot use adaptive feature indices")
            if route_hysteresis_urgency_relief != 0.0 or route_hysteresis_class_2_relief != 0.0:
                raise ValueError("legacy route hysteresis cannot use adaptive relief")
        else:
            adaptive_indices = {
                "route urgency": route_urgency_feature_index,
                "route class-2": route_class_2_feature_index,
            }
            for field, index in adaptive_indices.items():
                if type(index) is not int:
                    raise ValueError(f"{field} feature index must be an integer")
                if not 0 <= index < candidate_feature_dim:
                    raise ValueError(f"{field} feature index is outside candidate schema")
            if route_switch_feature_index is None:
                raise ValueError("decoupled route hysteresis requires a switch feature")
            if len(
                {
                    route_switch_feature_index,
                    route_urgency_feature_index,
                    route_class_2_feature_index,
                }
            ) != 3:
                raise ValueError("adaptive route hysteresis feature indices must be distinct")
        for value, field in (
            (route_hysteresis_residual_init, "residual init"),
            (route_hysteresis_residual_cap, "residual cap"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"route hysteresis {field} must be finite and non-negative")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"route hysteresis {field} must be finite and non-negative")
        if route_hysteresis_residual_init > route_hysteresis_residual_cap:
            raise ValueError("route hysteresis residual init must not exceed its cap")
        if route_hysteresis_residual_parameterization not in {
            "scalar",
            "urgency_linear",
        }:
            raise ValueError(
                "unsupported route hysteresis residual parameterization"
            )
        if (
            route_hysteresis_residual_parameterization == "urgency_linear"
            and route_hysteresis_residual_cap <= 0.0
        ):
            raise ValueError(
                "urgency-linear route residual requires a positive cap"
            )
        if (
            route_hysteresis_residual_cap > 0.0
            and route_hysteresis_mode != "decoupled_adaptive"
        ):
            raise ValueError(
                "route hysteresis residual is supported only for decoupled_adaptive"
            )
        self.route_switch_feature_index = route_switch_feature_index
        self.route_hysteresis_beta = float(route_hysteresis_beta)
        self.route_hysteresis_mode = route_hysteresis_mode
        self.route_urgency_feature_index = route_urgency_feature_index
        self.route_class_2_feature_index = route_class_2_feature_index
        self.route_hysteresis_urgency_relief = float(
            route_hysteresis_urgency_relief
        )
        self.route_hysteresis_class_2_relief = float(
            route_hysteresis_class_2_relief
        )
        self.route_hysteresis_residual_cap = float(
            route_hysteresis_residual_cap
        )
        self.route_hysteresis_residual_parameterization = (
            route_hysteresis_residual_parameterization
        )
        if (
            self.route_hysteresis_residual_cap > 0.0
            and self.route_hysteresis_residual_parameterization == "scalar"
        ):
            self.route_hysteresis_residual_bias = nn.Parameter(
                torch.tensor(float(route_hysteresis_residual_init))
            )
        else:
            self.register_parameter("route_hysteresis_residual_bias", None)
        if self.route_hysteresis_residual_parameterization == "urgency_linear":
            self.route_hysteresis_residual_calm_bias = nn.Parameter(
                torch.tensor(float(route_hysteresis_residual_init))
            )
            self.route_hysteresis_residual_urgent_bias = nn.Parameter(
                torch.tensor(float(route_hysteresis_residual_init))
            )
        else:
            self.register_parameter("route_hysteresis_residual_calm_bias", None)
            self.register_parameter("route_hysteresis_residual_urgent_bias", None)
        self.project_route_hysteresis_residual_()
        encoder = [
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            encoder.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder)
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _candidate_quality_logits(
        self,
        candidates: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scorer_candidates = candidates
        if self.route_hysteresis_mode == "decoupled_adaptive":
            scorer_candidates = candidates.clone()
            scorer_candidates[..., self.route_switch_feature_index] = 0.0
        encoded = self.encoder(scorer_candidates)
        if action_mask is None:
            weights = torch.ones_like(encoded[..., :1])
        else:
            weights = action_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=-2) / weights.sum(
            dim=-2
        ).clamp_min(1.0)
        context = pooled.unsqueeze(-2).expand_as(encoded)
        return self.scorer(torch.cat([encoded, context], dim=-1)).squeeze(-1)

    def _route_hysteresis_terms(
        self,
        candidates: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | float, torch.Tensor, torch.Tensor | None]:
        switch_indicator = candidates[..., self.route_switch_feature_index]
        bonus_scale: torch.Tensor | float = 1.0
        urgency = None
        if self.route_hysteresis_mode == "decoupled_adaptive":
            urgency = candidates[
                ..., self.route_urgency_feature_index
            ].amax(dim=-1, keepdim=True).clamp(0.0, 1.0)
            class_2 = candidates[
                ..., self.route_class_2_feature_index
            ].amax(dim=-1, keepdim=True).clamp(0.0, 1.0)
            bonus_scale = (
                1.0 - self.route_hysteresis_urgency_relief * urgency
            ) * (
                1.0 - self.route_hysteresis_class_2_relief * class_2
            )
        stay_indicator = (switch_indicator < 0.5).to(dtype)
        return bonus_scale, stay_indicator, urgency

    def route_hysteresis_residual_parameters(
        self,
    ) -> tuple[nn.Parameter, ...]:
        """Return the active projected residual parameters in frozen order."""

        if self.route_hysteresis_residual_bias is not None:
            return (self.route_hysteresis_residual_bias,)
        if self.route_hysteresis_residual_calm_bias is not None:
            return (
                self.route_hysteresis_residual_calm_bias,
                self.route_hysteresis_residual_urgent_bias,
            )
        return ()

    def _route_hysteresis_residual(
        self,
        urgency: torch.Tensor | None,
        *,
        isolate_residual_gradient: bool,
    ) -> torch.Tensor | float:
        if self.route_hysteresis_residual_bias is not None:
            return self.route_hysteresis_residual_bias
        if self.route_hysteresis_residual_calm_bias is None:
            return 0.0
        if urgency is None:
            raise RuntimeError("urgency-linear route residual has no urgency context")
        interpolation_urgency = (
            urgency.detach() if isolate_residual_gradient else urgency
        )
        residual = (
            (1.0 - interpolation_urgency)
            * self.route_hysteresis_residual_calm_bias
            + interpolation_urgency
            * self.route_hysteresis_residual_urgent_bias
        )
        if not isolate_residual_gradient:
            residual = residual.detach()
        return residual

    def _apply_route_hysteresis(
        self,
        quality_logits: torch.Tensor,
        candidates: torch.Tensor,
        *,
        isolate_residual_gradient: bool,
    ) -> torch.Tensor:
        if isolate_residual_gradient:
            quality_logits = quality_logits.detach()
        bonus_scale, stay_indicator, urgency = self._route_hysteresis_terms(
            candidates, quality_logits.dtype
        )
        residual = self._route_hysteresis_residual(
            urgency,
            isolate_residual_gradient=isolate_residual_gradient,
        )
        if isolate_residual_gradient:
            if isinstance(bonus_scale, torch.Tensor):
                bonus_scale = bonus_scale.detach()
            stay_indicator = stay_indicator.detach()
        return quality_logits + (
            self.route_hysteresis_beta + residual
        ) * bonus_scale * stay_indicator

    @torch.no_grad()
    def project_route_hysteresis_residual_(self) -> None:
        """Project the trainable residual back onto its frozen interval."""

        for residual in self.route_hysteresis_residual_parameters():
            cap = torch.as_tensor(
                route_hysteresis_residual_dtype_cap(
                    self.route_hysteresis_residual_cap,
                    dtype=residual.dtype,
                ),
                dtype=residual.dtype,
                device=residual.device,
            )
            residual.clamp_(0.0, cap)

    def isolated_regularizer_logits(
        self,
        candidates: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return current logits while exposing only the residual gradient."""

        if not self.route_hysteresis_residual_parameters():
            raise ValueError(
                "isolated regularizer logits require a trainable route residual"
            )
        logits = self._candidate_quality_logits(candidates, action_mask)
        logits = self._apply_route_hysteresis(
            logits,
            candidates,
            isolate_residual_gradient=True,
        )
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits

    def forward(self, candidates: torch.Tensor, action_mask=None):
        logits = self._candidate_quality_logits(candidates, action_mask)
        if (
            self.route_hysteresis_beta > 0.0
            or self.route_hysteresis_residual_parameters()
        ):
            logits = self._apply_route_hysteresis(
                logits,
                candidates,
                isolate_residual_gradient=False,
            )
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


def _avoidable_switch_masks(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    action_mask: torch.Tensor,
    route_switch_feature_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidates.shape[:-1] != logits.shape:
        raise ValueError("candidate and logit shapes do not match")
    if action_mask.shape != logits.shape:
        raise ValueError("action-mask and logit shapes do not match")
    if type(route_switch_feature_index) is not int or not (
        0 <= route_switch_feature_index < candidates.shape[-1]
    ):
        raise ValueError("route-switch feature index is outside candidate schema")

    feasible = action_mask.bool().clone()
    if feasible.shape[-1] == 0:
        raise ValueError("action schema must contain the NO_OP slot")
    feasible[..., 0] = False
    switch_indicator = candidates[..., route_switch_feature_index] >= 0.5
    feasible_switch = feasible & switch_indicator
    feasible_stay = feasible & ~switch_indicator
    eligible = feasible_switch.any(dim=-1) & feasible_stay.any(dim=-1)
    return feasible_switch, feasible_stay, eligible


def avoidable_switch_probabilities(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    action_mask: torch.Tensor,
    route_switch_feature_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return switch probability and eligibility for each agent decision.

    A decision is eligible only when the cached next hop and at least one
    switching alternative are both feasible. Action zero is the LEO wrapper's
    NO_OP slot and never serves as a cached-route action.
    """

    feasible_switch, feasible_stay, eligible = _avoidable_switch_masks(
        logits,
        candidates,
        action_mask,
        route_switch_feature_index,
    )
    feasible = feasible_switch | feasible_stay
    masked_logits = logits.masked_fill(~feasible, -1e9)
    probabilities = torch.softmax(masked_logits, dim=-1) * feasible.to(logits.dtype)
    switch_probability = (
        probabilities * feasible_switch.to(probabilities.dtype)
    ).sum(dim=-1)
    return switch_probability, eligible


def avoidable_switch_decisions(
    candidates: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
    route_switch_feature_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return realized avoidable-switch costs and decision opportunities.

    The accounting is deliberately pre-contention: a cost is one when the
    cached next hop and a switching alternative are both feasible and the
    policy selects a switching action. First-use and forced reroutes therefore
    have zero opportunity and zero cost.
    """

    if actions.shape != candidates.shape[:-2]:
        raise ValueError("action and candidate decision shapes do not match")
    if actions.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("actions must use an integer tensor dtype")
    action_count = candidates.shape[-2]
    if bool(((actions < 0) | (actions >= action_count)).any()):
        raise ValueError("action index is outside the candidate schema")

    probe_logits = torch.zeros(
        candidates.shape[:-1],
        dtype=candidates.dtype,
        device=candidates.device,
    )
    feasible_switch, _, opportunities = _avoidable_switch_masks(
        probe_logits,
        candidates,
        action_mask,
        route_switch_feature_index,
    )
    selected_is_feasible = action_mask.bool().gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    if not bool(selected_is_feasible.all()):
        raise ValueError("selected action is infeasible")
    selected_switch = feasible_switch.gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    costs = opportunities & selected_switch
    return costs, opportunities


def projected_lagrange_multiplier_update(
    multiplier: float,
    empirical_rate: float,
    budget: float,
    learning_rate: float,
    *,
    minimum: float = 0.0,
    maximum: float,
) -> float:
    """Apply one projected dual-ascent update for a rate constraint."""

    values = {
        "multiplier": multiplier,
        "empirical rate": empirical_rate,
        "budget": budget,
        "learning rate": learning_rate,
        "minimum": minimum,
        "maximum": maximum,
    }
    for field, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"dual {field} must be finite")
        if not math.isfinite(float(value)):
            raise ValueError(f"dual {field} must be finite")
    if not 0.0 <= float(empirical_rate) <= 1.0:
        raise ValueError("dual empirical rate must be in [0, 1]")
    if not 0.0 <= float(budget) <= 1.0:
        raise ValueError("dual budget must be in [0, 1]")
    if float(learning_rate) <= 0.0:
        raise ValueError("dual learning rate must be positive")
    if float(minimum) < 0.0 or float(maximum) <= float(minimum):
        raise ValueError("dual projection interval is invalid")
    if not float(minimum) <= float(multiplier) <= float(maximum):
        raise ValueError("dual multiplier is outside its projection interval")

    updated = float(multiplier) + float(learning_rate) * (
        float(empirical_rate) - float(budget)
    )
    return min(float(maximum), max(float(minimum), updated))


def avoidable_switch_logit_margin_penalties(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    action_mask: torch.Tensor,
    route_switch_feature_index: int,
    *,
    route_urgency_feature_index: int,
    route_class_2_feature_index: int,
    route_hysteresis_beta: float,
    route_hysteresis_urgency_relief: float = 0.0,
    route_hysteresis_class_2_relief: float = 0.0,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize switch margins without undoing adaptive urgency relief.

    Actor logits already contain the relief-scaled cached-route bonus. The
    comparison restores the missing part of that bonus before applying the
    hinge, then scales the remaining penalty by the same relief factor. Thus
    urgency can relax hysteresis without creating regularization pressure that
    was absent in the corresponding non-urgent state.
    """

    for value, field in (
        (route_hysteresis_urgency_relief, "urgency relief"),
        (route_hysteresis_class_2_relief, "class-2 relief"),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"route hysteresis {field} must be finite and in [0, 1]")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"route hysteresis {field} must be finite and in [0, 1]")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        raise ValueError("avoidable-switch logit margin must be finite and non-negative")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("avoidable-switch logit margin must be finite and non-negative")
    if not isinstance(route_hysteresis_beta, (int, float)) or isinstance(
        route_hysteresis_beta, bool
    ):
        raise ValueError("route hysteresis beta must be finite and non-negative")
    if (
        not math.isfinite(float(route_hysteresis_beta))
        or float(route_hysteresis_beta) < 0.0
    ):
        raise ValueError("route hysteresis beta must be finite and non-negative")
    for index, field in (
        (route_urgency_feature_index, "route urgency feature index"),
        (route_class_2_feature_index, "route class-2 feature index"),
    ):
        if type(index) is not int or not 0 <= index < candidates.shape[-1]:
            raise ValueError(f"{field} is outside candidate schema")

    feasible_switch, feasible_stay, eligible = _avoidable_switch_masks(
        logits,
        candidates,
        action_mask,
        route_switch_feature_index,
    )
    best_switch = logits.masked_fill(~feasible_switch, -1e9).amax(dim=-1)
    best_stay = logits.masked_fill(~feasible_stay, -1e9).amax(dim=-1)
    urgency = candidates[..., route_urgency_feature_index].amax(dim=-1).clamp(0.0, 1.0)
    class_2 = candidates[..., route_class_2_feature_index].amax(dim=-1).clamp(0.0, 1.0)
    relief_scale = (
        1.0 - float(route_hysteresis_urgency_relief) * urgency
    ) * (
        1.0 - float(route_hysteresis_class_2_relief) * class_2
    )
    full_hysteresis_stay = best_stay + float(route_hysteresis_beta) * (
        1.0 - relief_scale
    )
    penalties = torch.relu(
        best_switch - full_hysteresis_stay + float(margin)
    ) * relief_scale
    return torch.where(eligible, penalties, torch.zeros_like(penalties)), eligible


def avoidable_switch_actual_logit_margin_penalties(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    action_mask: torch.Tensor,
    route_switch_feature_index: int,
    *,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a greedy hinge directly to the actor logits used for action choice."""

    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        raise ValueError("avoidable-switch logit margin must be finite and non-negative")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("avoidable-switch logit margin must be finite and non-negative")

    feasible_switch, feasible_stay, eligible = _avoidable_switch_masks(
        logits,
        candidates,
        action_mask,
        route_switch_feature_index,
    )
    best_switch = logits.masked_fill(~feasible_switch, -1e9).amax(dim=-1)
    best_stay = logits.masked_fill(~feasible_stay, -1e9).amax(dim=-1)
    penalties = torch.relu(best_switch - best_stay + float(margin))
    return torch.where(eligible, penalties, torch.zeros_like(penalties)), eligible


def select_leo_validation_record(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    delivery_tolerance: float = 0.0,
    class_2_tolerance: float = 0.0,
    switch_budget: float | None = None,
) -> Mapping[str, Any]:
    """Select a validation checkpoint without consulting the test panel."""

    if not records:
        raise ValueError("validation records must not be empty")
    for value, field in (
        (delivery_tolerance, "delivery tolerance"),
        (class_2_tolerance, "class-2 tolerance"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"validation {field} must be finite and non-negative")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"validation {field} must be finite and non-negative")

    if mode == "avoidable_switch_budget_constrained":
        if isinstance(switch_budget, bool) or not isinstance(
            switch_budget, (int, float)
        ):
            raise ValueError("validation switch budget must be finite and in [0, 1]")
        if not math.isfinite(float(switch_budget)) or not 0.0 <= float(
            switch_budget
        ) <= 1.0:
            raise ValueError("validation switch budget must be finite and in [0, 1]")

    base_fields = (
        "delivery_ratio",
        "mean_reward",
        "drop_rate",
        "average_delay_slots",
    )
    required_fields = set(base_fields)
    if mode in {
        "stability_constrained",
        "avoidable_stability_constrained",
    }:
        required_fields.update(
            {
                "routing_switches",
                "class_2_delivery_ratio",
                "environment_steps",
            }
        )
        if mode == "avoidable_stability_constrained":
            required_fields.add("avoidable_switch_rate")
    elif mode == "avoidable_switch_budget_constrained":
        required_fields.update(
            {
                "class_2_delivery_ratio",
                "decision_avoidable_switch_rate",
                "environment_steps",
            }
        )
    elif mode != "legacy_lexicographic":
        raise ValueError("unsupported validation selection mode")
    for record in records:
        missing = required_fields.difference(record)
        if missing:
            raise ValueError(f"validation record is missing fields: {sorted(missing)}")
        for field in required_fields:
            value = record[field]
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"validation field {field} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"validation field {field} must be finite")
        if (
            mode == "avoidable_switch_budget_constrained"
            and not 0.0
            <= float(record["decision_avoidable_switch_rate"])
            <= 1.0
        ):
            raise ValueError(
                "validation decision_avoidable_switch_rate must be in [0, 1]"
            )

    if mode == "legacy_lexicographic":
        return max(
            records,
            key=lambda record: (
                float(record["delivery_ratio"]),
                float(record["mean_reward"]),
                -float(record["drop_rate"]),
                -float(record["average_delay_slots"]),
            ),
        )

    if mode == "avoidable_switch_budget_constrained":
        feasible = [
            record
            for record in records
            if float(record["decision_avoidable_switch_rate"])
            <= float(switch_budget) + 1e-12
        ]
        if feasible:
            return max(
                feasible,
                key=lambda record: (
                    float(record["delivery_ratio"]),
                    float(record["class_2_delivery_ratio"]),
                    float(record["mean_reward"]),
                    -float(record["drop_rate"]),
                    -float(record["average_delay_slots"]),
                    -float(record["decision_avoidable_switch_rate"]),
                    -int(record["environment_steps"]),
                ),
            )
        return min(
            records,
            key=lambda record: (
                float(record["decision_avoidable_switch_rate"]),
                -float(record["delivery_ratio"]),
                -float(record["class_2_delivery_ratio"]),
                -float(record["mean_reward"]),
                float(record["drop_rate"]),
                float(record["average_delay_slots"]),
                int(record["environment_steps"]),
            ),
        )

    best_delivery = max(float(record["delivery_ratio"]) for record in records)
    delivery_eligible = [
        record
        for record in records
        if float(record["delivery_ratio"])
        >= best_delivery - float(delivery_tolerance)
    ]
    best_class_2 = max(
        float(record["class_2_delivery_ratio"])
        for record in delivery_eligible
    )
    safety_eligible = [
        record
        for record in delivery_eligible
        if float(record["class_2_delivery_ratio"])
        >= best_class_2 - float(class_2_tolerance)
    ]
    return min(
        safety_eligible,
        key=lambda record: (
            *(
                (float(record["avoidable_switch_rate"]),)
                if mode == "avoidable_stability_constrained"
                else ()
            ),
            float(record["routing_switches"]),
            -float(record["delivery_ratio"]),
            -float(record["class_2_delivery_ratio"]),
            -float(record["mean_reward"]),
            float(record["drop_rate"]),
            float(record["average_delay_slots"]),
            int(record["environment_steps"]),
        ),
    )


class PacketConditionedCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.network(state)


class GraphAttentionCritic(nn.Module):
    """Permutation-invariant centralized value network for a graph state."""

    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.state_size = (
            global_feature_dim
            + n_nodes * node_feature_dim
            + n_nodes * n_nodes * edge_feature_dim
        )
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_feature_dim),
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_feature_dim),
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.message_layers.append(
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            )
            self.attention_layers.append(
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
            )
            self.update_layers.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
            )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(global_feature_dim),
            nn.Linear(global_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def split_state(self, state: torch.Tensor):
        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"graph state has {state.shape[-1]} values, expected {self.state_size}"
            )
        global_end = self.global_feature_dim
        node_end = global_end + self.n_nodes * self.node_feature_dim
        global_features = state[..., :global_end]
        node_features = state[..., global_end:node_end].reshape(
            *state.shape[:-1], self.n_nodes, self.node_feature_dim
        )
        edge_features = state[..., node_end:].reshape(
            *state.shape[:-1],
            self.n_nodes,
            self.n_nodes,
            self.edge_feature_dim,
        )
        return global_features, node_features, edge_features

    def forward(self, state: torch.Tensor):
        leading_shape = state.shape[:-1]
        flat_state = state.reshape(-1, state.shape[-1])
        global_features, node_features, edge_features = self.split_state(
            flat_state
        )
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        edge_mask = edge_features[..., 0] > 0.5
        identity = torch.eye(
            self.n_nodes, dtype=torch.bool, device=state.device
        ).unsqueeze(0)
        message_mask = edge_mask | identity

        for message_layer, attention_layer, update_layer in zip(
            self.message_layers,
            self.attention_layers,
            self.update_layers,
        ):
            src = nodes.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
            dst = nodes.unsqueeze(1).expand(-1, self.n_nodes, -1, -1)
            context = torch.cat([src, dst, edges], dim=-1)
            messages = message_layer(context)
            attention_logits = attention_layer(context).squeeze(-1)
            attention_logits = attention_logits.masked_fill(
                ~message_mask, -1e9
            )
            attention = torch.softmax(attention_logits, dim=1)
            aggregated = (attention.unsqueeze(-1) * messages).sum(dim=1)
            nodes = nodes + update_layer(torch.cat([nodes, aggregated], dim=-1))

        pooled_mean = nodes.mean(dim=1)
        pooled_max = nodes.max(dim=1).values
        global_context = self.global_encoder(global_features)
        value = self.value_head(
            torch.cat([pooled_mean, pooled_max, global_context], dim=-1)
        )
        return value.reshape(*leading_shape, 1)


def permute_flat_graph_state(
    state: torch.Tensor,
    permutation: torch.Tensor,
    *,
    n_nodes: int,
    node_feature_dim: int,
    edge_feature_dim: int,
    global_feature_dim: int,
) -> torch.Tensor:
    """Reorder node labels while preserving the represented physical graph."""
    global_end = global_feature_dim
    node_end = global_end + n_nodes * node_feature_dim
    global_features = state[..., :global_end]
    node_features = state[..., global_end:node_end].reshape(
        *state.shape[:-1], n_nodes, node_feature_dim
    )
    edge_features = state[..., node_end:].reshape(
        *state.shape[:-1], n_nodes, n_nodes, edge_feature_dim
    )
    node_features = node_features[..., permutation, :]
    edge_features = edge_features[..., permutation, :, :]
    edge_features = edge_features[..., :, permutation, :]
    return torch.cat(
        [
            global_features,
            node_features.reshape(*state.shape[:-1], -1),
            edge_features.reshape(*state.shape[:-1], -1),
        ],
        dim=-1,
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    """GAE with zero bootstrap only for true terminal states.

    Time-limit truncation uses next_values for the one-step delta, then stops
    carrying sampled future advantages across the rollout boundary.
    """
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    for batch in range(values.shape[0]):
        length = int(valid[batch].sum().item())
        next_gae = torch.zeros_like(values[batch, 0])
        for t in reversed(range(length)):
            nonterminal = 1.0 - terminated[batch, t]
            delta = (
                rewards[batch, t]
                + gamma * nonterminal * next_values[batch, t]
                - values[batch, t]
            )
            carry = nonterminal * (1.0 - truncated[batch, t])
            next_gae = delta + gamma * gae_lambda * carry * next_gae
            advantages[batch, t] = next_gae
            returns[batch, t] = next_gae + values[batch, t]
    return advantages, returns


def masked_standardize(values, mask, epsilon: float = 1e-8):
    selected = values[mask]
    mean = selected.mean()
    std = selected.std(unbiased=False)
    result = values.clone()
    result[mask] = (selected - mean) / (std + epsilon)
    return result


def feasible_normalized_entropy(distribution, action_mask):
    feasible = action_mask.sum(dim=-1).clamp_min(1)
    denominator = torch.log(feasible.clamp_min(2).float())
    return distribution.entropy() / denominator


def shuffled_transition_minibatches(valid_mask, num_minibatches, generator=None):
    """Partition every valid (episode, time) index exactly once."""
    transitions = valid_mask.nonzero(as_tuple=False)
    if len(transitions) == 0:
        return []
    count = min(max(1, int(num_minibatches)), len(transitions))
    permutation = torch.randperm(
        len(transitions), device=transitions.device, generator=generator
    )
    return [transitions[index] for index in permutation.chunk(count)]
