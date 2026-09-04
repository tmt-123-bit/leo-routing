"""Relative-anchor margin cap for frozen proposed-actor hysteresis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from calm_selective_hysteresis_policy import (
    CalmSelectiveActorScoreHysteresisPolicy,
    load_calm_selective_hysteresis_policy,
)
from hysteresis_policy import SWITCH_FEATURE_INDEX


ANCHOR_URGENCY_RELIEF = 0.5
MAX_EQUIVALENT_MARGIN_CAP = 0.75


def _unit_interval(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return numeric


class MarginCappedActorScoreHysteresisPolicy(
    CalmSelectiveActorScoreHysteresisPolicy
):
    """Cap the v9 stay threshold relative to the failed v8 anchor.

    For clipped waiting age ``a`` and class-2 indicator ``z``:

        anchor = stay_bonus * (1 - 0.5 * a)
                            * (1 - class_2_relief * z)
        uncapped = stay_bonus * (1 - urgency_relief * a)
                              * (1 - class_2_relief * z)
                 + calm_bonus * (1 - z) * max(0, 1 - 2 * a)
        beta_eff = min(uncapped, anchor + margin_cap).

    The cap only releases raw-actor switches that the uncapped v9 policy
    would suppress.  It cannot create a new suppression.  First-use and
    forced-switch states remain raw-actor decisions.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        feature_dim: int,
        action_size: int,
        stay_bonus: float,
        urgency_relief: float,
        class_2_relief: float,
        calm_bonus: float,
        margin_cap: float,
        device: str,
        checkpoint_schema: Mapping[str, Any],
    ) -> None:
        super().__init__(
            actor,
            feature_dim=feature_dim,
            action_size=action_size,
            stay_bonus=stay_bonus,
            urgency_relief=urgency_relief,
            class_2_relief=class_2_relief,
            calm_bonus=calm_bonus,
            device=device,
            checkpoint_schema=checkpoint_schema,
        )
        self.margin_cap = _unit_interval(margin_cap, "margin cap")
        self._cap_active_choices = 0
        self._uncapped_would_suppress_choices = 0
        self._released_by_cap_choices = 0
        self._cap_reduction_sum = 0.0
        self._cap_reduction_min: float | None = None
        self._cap_reduction_max: float | None = None

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        expected_shape = (
            observation.shape[0],
            self.action_size * self.feature_dim,
        )
        if observation.shape != expected_shape:
            raise ValueError(
                f"observation shape {observation.shape} != {expected_shape}"
            )
        if mask.shape != (observation.shape[0], self.action_size):
            raise ValueError("action-mask shape does not match the checkpoint")

        candidate_array = observation.reshape(
            observation.shape[0], self.action_size, self.feature_dim
        )
        urgency_array, class_2_array = self._validated_packet_state(
            candidate_array, mask
        )
        candidates = torch.from_numpy(candidate_array).to(self.device)
        action_mask = torch.from_numpy(mask).to(self.device)
        with torch.no_grad():
            raw_logits = self.actor(candidates, action_mask)
            stay_mask = (
                candidates[..., SWITCH_FEATURE_INDEX] < 0.5
            ) & action_mask
            raw_actions = raw_logits.argmax(dim=-1)

        raw_logits_array = raw_logits.detach().cpu().numpy()
        raw_actions_array = raw_actions.detach().cpu().numpy()
        stay_mask_array = stay_mask.detach().cpu().numpy()
        base_bonus = self.stay_bonus * (
            1.0 - self.urgency_relief * urgency_array
        ) * (1.0 - self.class_2_relief * class_2_array)
        calm_addon = self.calm_bonus * (1.0 - class_2_array) * np.clip(
            1.0 - 2.0 * urgency_array, 0.0, 1.0
        )
        uncapped_bonus = base_bonus + calm_addon
        anchor_bonus = self.stay_bonus * (
            1.0 - ANCHOR_URGENCY_RELIEF * urgency_array
        ) * (1.0 - self.class_2_relief * class_2_array)
        effective_bonus = np.minimum(
            uncapped_bonus, anchor_bonus + self.margin_cap
        )
        effective_calm_addon = effective_bonus - base_bonus
        if np.any(effective_calm_addon < -1e-6):
            raise RuntimeError("margin cap reduced the multiplicative base bonus")
        effective_calm_addon = np.maximum(effective_calm_addon, 0.0)
        actions = self._apply_adaptive_hysteresis(
            raw_logits_array,
            raw_actions_array,
            mask,
            stay_mask_array,
            effective_bonus,
        )
        self._record_diagnostics(
            raw_logits_array,
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
        )
        self._record_adaptive_diagnostics(
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
            urgency_array,
            class_2_array,
            effective_bonus,
        )
        self._record_calm_diagnostics(
            raw_logits_array,
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
            urgency_array,
            class_2_array,
            effective_calm_addon,
        )
        self._record_margin_cap_diagnostics(
            raw_logits_array,
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
            uncapped_bonus,
            effective_bonus,
        )
        return actions.astype(np.int64, copy=False)

    def _record_margin_cap_diagnostics(
        self,
        raw_logits: np.ndarray,
        raw_actions: np.ndarray,
        actions: np.ndarray,
        action_mask: np.ndarray,
        stay_mask: np.ndarray,
        uncapped_bonus: np.ndarray,
        effective_bonus: np.ndarray,
    ) -> None:
        for index in range(action_mask.shape[0]):
            feasible_next = np.flatnonzero(action_mask[index, 1:]) + 1
            if feasible_next.size == 0:
                continue
            staying = feasible_next[stay_mask[index, feasible_next]]
            if staying.size != 1 or int(raw_actions[index]) == int(staying[0]):
                continue
            incumbent = int(staying[0])
            raw_action = int(raw_actions[index])
            raw_advantage = float(
                raw_logits[index, raw_action] - raw_logits[index, incumbent]
            )
            uncapped = float(uncapped_bonus[index])
            effective = float(effective_bonus[index])
            reduction = uncapped - effective
            if reduction < -1e-6:
                raise RuntimeError("margin cap increased the uncapped bonus")
            reduction = max(0.0, reduction)
            self._cap_reduction_sum += reduction
            self._cap_reduction_min = (
                reduction
                if self._cap_reduction_min is None
                else min(self._cap_reduction_min, reduction)
            )
            self._cap_reduction_max = (
                reduction
                if self._cap_reduction_max is None
                else max(self._cap_reduction_max, reduction)
            )
            cap_active = reduction > 1e-7
            uncapped_would_suppress = raw_advantage < uncapped
            released = (
                uncapped_would_suppress and int(actions[index]) == raw_action
            )
            if int(actions[index]) == incumbent and not uncapped_would_suppress:
                raise RuntimeError(
                    "margin cap created a suppression absent from v9"
                )
            self._cap_active_choices += int(cap_active)
            self._uncapped_would_suppress_choices += int(
                uncapped_would_suppress
            )
            self._released_by_cap_choices += int(released)

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        raw_switch_choices = int(result["raw_switch_choices"])
        suppressed = int(result["suppressed_switch_choices"])
        if self._released_by_cap_choices + suppressed != (
            self._uncapped_would_suppress_choices
        ):
            raise RuntimeError(
                "margin-cap counterfactual outcomes do not reconcile"
            )
        result.update(
            {
                "adaptive_mode": "posthoc_relative_anchor_margin_cap",
                "anchor_urgency_relief": ANCHOR_URGENCY_RELIEF,
                "margin_cap": self.margin_cap,
                "uncapped_policy": (
                    "v9_u0p625_c0p750_k0p750"
                ),
                "cap_definition": (
                    "min(v9_uncapped,(1-0.5*a)*(1-0.75*z)+margin_cap)"
                ),
                "cap_active_raw_switch_choices": self._cap_active_choices,
                "cap_inactive_raw_switch_choices": (
                    raw_switch_choices - self._cap_active_choices
                ),
                "uncapped_would_suppress_switch_choices": (
                    self._uncapped_would_suppress_choices
                ),
                "released_by_cap_switch_choices": (
                    self._released_by_cap_choices
                ),
                "cap_reduction_sum": self._cap_reduction_sum,
                "cap_reduction_min": self._cap_reduction_min,
                "cap_reduction_max": self._cap_reduction_max,
                "cap_reduction_mean": (
                    self._cap_reduction_sum / raw_switch_choices
                    if raw_switch_choices
                    else None
                ),
            }
        )
        return result


def load_margin_capped_hysteresis_policy(
    checkpoint_path: str | Path,
    *,
    stay_bonus: float,
    urgency_relief: float,
    class_2_relief: float,
    calm_bonus: float,
    margin_cap: float,
    device: str = "cpu",
) -> tuple[MarginCappedActorScoreHysteresisPolicy, dict[str, Any]]:
    base, checkpoint = load_calm_selective_hysteresis_policy(
        checkpoint_path,
        stay_bonus=stay_bonus,
        urgency_relief=urgency_relief,
        class_2_relief=class_2_relief,
        calm_bonus=calm_bonus,
        device=device,
    )
    return (
        MarginCappedActorScoreHysteresisPolicy(
            base.actor,
            feature_dim=base.feature_dim,
            action_size=base.action_size,
            stay_bonus=stay_bonus,
            urgency_relief=urgency_relief,
            class_2_relief=class_2_relief,
            calm_bonus=calm_bonus,
            margin_cap=margin_cap,
            device=device,
            checkpoint_schema=base.checkpoint_schema,
        ),
        checkpoint,
    )
