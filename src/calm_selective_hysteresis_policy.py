"""Calm-flow selective post-hoc hysteresis for frozen proposed actors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from adaptive_hysteresis_policy import (
    AdaptiveActorScoreHysteresisPolicy,
    load_adaptive_hysteresis_policy,
)
from hysteresis_policy import SWITCH_FEATURE_INDEX


AGE_BIN_EDGES = (0.0, 0.125, 0.25, 0.375, 0.5, 1.0)
AGE_BIN_LABELS = (
    "0p000_to_0p125",
    "0p125_to_0p250",
    "0p250_to_0p375",
    "0p375_to_0p500",
    "0p500_to_1p000",
)
ANCHOR_MARGIN_UPPER_EDGES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75)
ANCHOR_MARGIN_BIN_LABELS = (
    "lt_0p000",
    "0p000_to_0p125",
    "0p125_to_0p250",
    "0p250_to_0p375",
    "0p375_to_0p500",
    "0p500_to_0p750",
    "ge_0p750",
)
ANCHOR_STAY_BONUS = 1.0
ANCHOR_URGENCY_RELIEF = 0.5
ANCHOR_CLASS_2_RELIEF = 0.75
CALM_AGE_CUTOFF = 0.5


def _unit_interval(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return numeric


class CalmSelectiveActorScoreHysteresisPolicy(
    AdaptiveActorScoreHysteresisPolicy
):
    """Add hysteresis only for calm, non-class-2 cached-route choices.

    For clipped waiting age ``a`` and class-2 indicator ``z`` the threshold is

        stay_bonus * (1 - urgency_relief * a)
                   * (1 - class_2_relief * z)
        + calm_bonus * (1 - z) * max(0, 1 - 2 * a).

    The added term is exactly zero for class-2 packets and for ``a >= 0.5``.
    As in the base policy, first-use and forced-switch states are untouched.
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
            device=device,
            checkpoint_schema=checkpoint_schema,
        )
        self.calm_bonus = _unit_interval(calm_bonus, "calm bonus")
        self._calm_addon_sum = 0.0
        self._calm_addon_min: float | None = None
        self._calm_addon_max: float | None = None
        self._calm_raw_switch_choices = 0
        self._calm_suppressed_switch_choices = 0
        self._calm_allowed_switch_choices = 0
        self._age_choice_bins = {
            label: {"suppressed": 0, "allowed": 0}
            for label in AGE_BIN_LABELS
        }
        self._anchor_margin_choice_bins = {
            label: {"suppressed": 0, "allowed": 0}
            for label in ANCHOR_MARGIN_BIN_LABELS
        }
        self._anchor_margin_sum = 0.0
        self._anchor_margin_min: float | None = None
        self._anchor_margin_max: float | None = None

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
            urgency = torch.from_numpy(urgency_array).to(self.device)
            class_2 = torch.from_numpy(class_2_array).to(self.device)
            base_bonus = self.stay_bonus * (
                1.0 - self.urgency_relief * urgency
            ) * (1.0 - self.class_2_relief * class_2)
            calm_weight = (1.0 - class_2) * torch.clamp(
                1.0 - 2.0 * urgency, min=0.0, max=1.0
            )
            calm_addon = self.calm_bonus * calm_weight
            effective_bonus = base_bonus + calm_addon

        raw_logits_array = raw_logits.detach().cpu().numpy()
        raw_actions_array = raw_actions.detach().cpu().numpy()
        stay_mask_array = stay_mask.detach().cpu().numpy()
        effective_bonus_array = effective_bonus.detach().cpu().numpy()
        calm_addon_array = calm_addon.detach().cpu().numpy()
        actions = self._apply_adaptive_hysteresis(
            raw_logits_array,
            raw_actions_array,
            mask,
            stay_mask_array,
            effective_bonus_array,
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
            effective_bonus_array,
        )
        self._record_calm_diagnostics(
            raw_logits_array,
            raw_actions_array,
            actions,
            mask,
            stay_mask_array,
            urgency_array,
            class_2_array,
            calm_addon_array,
        )
        return actions.astype(np.int64, copy=False)

    def _record_calm_diagnostics(
        self,
        raw_logits: np.ndarray,
        raw_actions: np.ndarray,
        actions: np.ndarray,
        action_mask: np.ndarray,
        stay_mask: np.ndarray,
        urgency: np.ndarray,
        class_2: np.ndarray,
        calm_addon: np.ndarray,
    ) -> None:
        for index in range(action_mask.shape[0]):
            feasible_next = np.flatnonzero(action_mask[index, 1:]) + 1
            if feasible_next.size == 0:
                continue
            staying = feasible_next[stay_mask[index, feasible_next]]
            if staying.size != 1 or int(raw_actions[index]) == int(staying[0]):
                continue
            addon = float(calm_addon[index])
            self._calm_addon_sum += addon
            self._calm_addon_min = (
                addon
                if self._calm_addon_min is None
                else min(self._calm_addon_min, addon)
            )
            self._calm_addon_max = (
                addon
                if self._calm_addon_max is None
                else max(self._calm_addon_max, addon)
            )
            incumbent = int(staying[0])
            raw_action = int(raw_actions[index])
            if int(actions[index]) == incumbent:
                outcome = "suppressed"
            elif int(actions[index]) == raw_action:
                outcome = "allowed"
            else:
                raise RuntimeError(
                    "calm-selective hysteresis produced an invalid action"
                )

            age = float(urgency[index])
            age_bin = int(
                np.searchsorted(AGE_BIN_EDGES[1:-1], age, side="right")
            )
            self._age_choice_bins[AGE_BIN_LABELS[age_bin]][outcome] += 1
            anchor_bonus = ANCHOR_STAY_BONUS * (
                1.0 - ANCHOR_URGENCY_RELIEF * age
            ) * (
                1.0 - ANCHOR_CLASS_2_RELIEF * float(class_2[index])
            )
            raw_advantage = float(
                raw_logits[index, raw_action]
                - raw_logits[index, incumbent]
            )
            anchor_margin = raw_advantage - anchor_bonus
            margin_bin = int(
                np.searchsorted(
                    ANCHOR_MARGIN_UPPER_EDGES,
                    anchor_margin,
                    side="right",
                )
            )
            self._anchor_margin_choice_bins[
                ANCHOR_MARGIN_BIN_LABELS[margin_bin]
            ][outcome] += 1
            self._anchor_margin_sum += anchor_margin
            self._anchor_margin_min = (
                anchor_margin
                if self._anchor_margin_min is None
                else min(self._anchor_margin_min, anchor_margin)
            )
            self._anchor_margin_max = (
                anchor_margin
                if self._anchor_margin_max is None
                else max(self._anchor_margin_max, anchor_margin)
            )
            is_calm_non_class_2 = (
                float(class_2[index]) < 0.5 and age < CALM_AGE_CUTOFF
            )
            if not is_calm_non_class_2:
                continue
            self._calm_raw_switch_choices += 1
            if outcome == "suppressed":
                self._calm_suppressed_switch_choices += 1
            else:
                self._calm_allowed_switch_choices += 1

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        raw_switch_choices = int(result["raw_switch_choices"])
        if (
            self._calm_suppressed_switch_choices
            + self._calm_allowed_switch_choices
            != self._calm_raw_switch_choices
        ):
            raise RuntimeError("calm diagnostic outcomes do not reconcile")
        for name, bins in (
            ("age", self._age_choice_bins),
            ("anchor-margin", self._anchor_margin_choice_bins),
        ):
            total = sum(sum(counts.values()) for counts in bins.values())
            if total != raw_switch_choices:
                raise RuntimeError(f"{name} diagnostic bins do not reconcile")
        result.update(
            {
                "adaptive_mode": "posthoc_calm_non_class_2_additive",
                "calm_bonus": self.calm_bonus,
                "calm_age_cutoff": CALM_AGE_CUTOFF,
                "calm_addon_sum": self._calm_addon_sum,
                "calm_addon_min": self._calm_addon_min,
                "calm_addon_max": self._calm_addon_max,
                "calm_addon_mean": (
                    self._calm_addon_sum / raw_switch_choices
                    if raw_switch_choices
                    else None
                ),
                "calm_non_class_2_raw_switch_choices": (
                    self._calm_raw_switch_choices
                ),
                "calm_non_class_2_suppressed_switch_choices": (
                    self._calm_suppressed_switch_choices
                ),
                "calm_non_class_2_allowed_switch_choices": (
                    self._calm_allowed_switch_choices
                ),
                "age_bin_edges": list(AGE_BIN_EDGES),
                "age_choice_bins": {
                    label: dict(counts)
                    for label, counts in self._age_choice_bins.items()
                },
                "relative_anchor_definition": (
                    "raw_switch_advantage_minus_beta_anchor; "
                    "beta_anchor=(1-0.5*a)*(1-0.75*z)"
                ),
                "relative_anchor_margin_upper_edges": list(
                    ANCHOR_MARGIN_UPPER_EDGES
                ),
                "relative_anchor_margin_choice_bins": {
                    label: dict(counts)
                    for label, counts in self._anchor_margin_choice_bins.items()
                },
                "relative_anchor_margin_sum": self._anchor_margin_sum,
                "relative_anchor_margin_min": self._anchor_margin_min,
                "relative_anchor_margin_max": self._anchor_margin_max,
                "relative_anchor_margin_mean": (
                    self._anchor_margin_sum / raw_switch_choices
                    if raw_switch_choices
                    else None
                ),
            }
        )
        return result


def load_calm_selective_hysteresis_policy(
    checkpoint_path: str | Path,
    *,
    stay_bonus: float,
    urgency_relief: float,
    class_2_relief: float,
    calm_bonus: float,
    device: str = "cpu",
) -> tuple[CalmSelectiveActorScoreHysteresisPolicy, dict[str, Any]]:
    base, checkpoint = load_adaptive_hysteresis_policy(
        checkpoint_path,
        stay_bonus=stay_bonus,
        urgency_relief=urgency_relief,
        class_2_relief=class_2_relief,
        device=device,
    )
    return (
        CalmSelectiveActorScoreHysteresisPolicy(
            base.actor,
            feature_dim=base.feature_dim,
            action_size=base.action_size,
            stay_bonus=stay_bonus,
            urgency_relief=urgency_relief,
            class_2_relief=class_2_relief,
            calm_bonus=calm_bonus,
            device=device,
            checkpoint_schema=base.checkpoint_schema,
        ),
        checkpoint,
    )
