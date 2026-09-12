"""Frozen-policy diagnostics; local sensitivity is not a training evaluation."""

from __future__ import annotations

from copy import deepcopy
import math

import torch

from mappo_design import avoidable_switch_decisions, avoidable_switch_probabilities


def _decision_tensors(logits, candidates, mask, switch_index):
    mask = mask.bool()
    probability, eligible = avoidable_switch_probabilities(
        logits, candidates, mask, switch_index
    )
    actions = logits.masked_fill(~mask, -1e9).argmax(dim=-1)
    costs, opportunities = avoidable_switch_decisions(
        candidates, mask, actions, switch_index
    )
    if not torch.equal(eligible, opportunities):
        raise RuntimeError("surrogate and decision opportunities disagree")
    feasible = mask.clone()
    feasible[..., 0] = False
    switches = feasible & (candidates[..., switch_index] >= 0.5)
    stays = feasible & ~switches
    margin = (
        logits.masked_fill(~switches, -1e9).amax(dim=-1)
        - logits.masked_fill(~stays, -1e9).amax(dim=-1)
    )
    return probability, eligible, actions, costs, feasible.any(dim=-1), margin, switches


def _eligible_summary(probability, costs, margin):
    count = probability.numel()
    probability_sum = float(probability.double().sum().item())
    greedy_switches = int(costs.sum().item())
    greedy_stays = ~costs
    stay_count = int(greedy_stays.sum().item())
    stay_probability_sum = float(probability[greedy_stays].double().sum().item())
    positive = margin > 0.0
    sensitivity = probability * (1.0 - probability)
    return {
        "eligible_decisions": count,
        "greedy_switches": greedy_switches,
        "greedy_switch_rate": greedy_switches / count if count else None,
        "conditional_probability_sum": probability_sum,
        "conditional_probability_rate": probability_sum / count if count else None,
        "greedy_stay_decisions": stay_count,
        "probability_mass_on_greedy_stay_sum": stay_probability_sum,
        "conditional_probability_on_greedy_stay_mean": (
            stay_probability_sum / stay_count if stay_count else None
        ),
        "mass_above_half_with_greedy_stay_count": int(
            ((probability > 0.5) & greedy_stays).sum().item()
        ),
        "positive_switch_margin_count": int(positive.sum().item()),
        "positive_switch_margin_sum": float(margin[positive].double().sum().item()),
        "switch_margin_sum": float(margin.double().sum().item()),
        "switch_margin_mean": float(margin.double().mean().item()) if count else None,
        "zero_switch_margin_count": int((margin == 0.0).sum().item()),
        "saturation_threshold": 0.001,
        "saturation_count": int(
            ((probability <= 0.001) | (probability >= 0.999)).sum().item()
        ),
        "probability_sensitivity_sum": float(sensitivity.double().sum().item()),
        "probability_sensitivity_mean": (
            float(sensitivity.double().mean().item()) if count else None
        ),
        "probability_sensitivity_definition": "dP_switch/d(common_switch_logit_shift) = P_switch*(1-P_switch)",
    }


@torch.no_grad()
def summarize_decisions(logits, candidates, mask, switch_index=17):
    """Summarize frozen decisions using the production pre-contention ledger.

    Probability is summed over switching actions. Greedy cost depends on the
    largest individual logit, including PyTorch's first-index tie breaking.
    """
    probability, eligible, _, costs, active, margin, switches = _decision_tensors(
        logits, candidates, mask, switch_index
    )
    summary = _eligible_summary(probability[eligible], costs[eligible], margin[eligible])
    summary.update(
        decisions=eligible.numel(),
        active_decisions=int(active.sum().item()),
        non_opportunity_active_decisions=int((active & ~eligible).sum().item()),
        inactive_decisions=int((~active).sum().item()),
    )
    switch_count = switches.sum(dim=-1)
    summary["by_switch_candidate_count"] = {
        str(int(count)): _eligible_summary(
            probability[eligible & (switch_count == count)],
            costs[eligible & (switch_count == count)],
            margin[eligible & (switch_count == count)],
        )
        for count in torch.unique(switch_count[eligible]).tolist()
    }
    return summary


def gradient_probe(
    actor,
    candidates,
    mask,
    switch_index=17,
    relative_step=1e-3,
    dual_multiplier=1.0,
):
    """Take one fixed normalized constraint-gradient step on a deep copy.

    The step has L2 norm relative_step * max(parameter L2 norm, 1). Its
    normalization removes positive multiplier magnitude. This only measures
    local sensitivity on fixed observations; it is not PPO or a QoS estimate.
    """
    for value, name in ((relative_step, "relative_step"), (dual_multiplier, "dual_multiplier")):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    probe = deepcopy(actor).eval()
    named_parameters = [(name, p) for name, p in probe.named_parameters() if p.requires_grad]
    if not named_parameters:
        raise ValueError("actor has no trainable parameters")
    parameters = tuple(p for _, p in named_parameters)
    candidates = candidates.detach().to(device=parameters[0].device, dtype=parameters[0].dtype)
    mask = mask.detach().to(device=parameters[0].device, dtype=torch.bool)
    logits = probe(candidates, mask)
    probability, eligible, actions, _, active, _, _ = _decision_tensors(
        logits, candidates, mask, switch_index
    )
    before = summarize_decisions(logits, candidates, mask, switch_index)
    parameter_norm = math.sqrt(sum(float(p.detach().double().square().sum()) for p in parameters))
    result = {
        "diagnostic_only": True,
        "interpretation": "one normalized local sensitivity step on fixed observations; not PPO or QoS",
        "status": "no_opportunities",
        "relative_step": float(relative_step),
        "dual_multiplier": float(dual_multiplier),
        "positive_multiplier_magnitude_removed_by_normalization": True,
        "before": before,
        "after": None,
        "parameter_norm": parameter_norm,
        "parameter_step_norm": None,
        "gradient_norm": None,
        "weighted_gradient_norm": None,
        "parameter_gradient_norms": None,
        "conditional_probability_change": None,
        "greedy_switch_rate_change": None,
        "non_opportunity_action_changes": None,
        "opportunity_action_changes": None,
        "constraint_probability_decreased": None,
    }
    if not bool(eligible.any()):
        return result

    gradients = torch.autograd.grad(probability[eligible].mean(), parameters, allow_unused=True)
    gradient_norms = {
        name: float(gradient.detach().double().norm()) if gradient is not None else 0.0
        for (name, _), gradient in zip(named_parameters, gradients)
    }
    gradient_norm = math.sqrt(sum(value * value for value in gradient_norms.values()))
    step_norm = relative_step * max(parameter_norm, 1.0)
    can_step = gradient_norm > 0.0 and dual_multiplier > 0.0 and step_norm > 0.0
    if can_step:
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                if gradient is not None:
                    parameter.add_(gradient, alpha=-step_norm / gradient_norm)
    with torch.no_grad():
        after_logits = probe(candidates, mask)
        after = summarize_decisions(after_logits, candidates, mask, switch_index)
        after_actions = after_logits.masked_fill(~mask, -1e9).argmax(dim=-1)
        changed = actions != after_actions
    probability_change = after["conditional_probability_rate"] - before["conditional_probability_rate"]
    result.update(
        status="probed" if can_step else "zero_step",
        after=after,
        parameter_step_norm=float(step_norm) if can_step else 0.0,
        gradient_norm=gradient_norm,
        weighted_gradient_norm=gradient_norm * dual_multiplier,
        parameter_gradient_norms=gradient_norms,
        conditional_probability_change=probability_change,
        greedy_switch_rate_change=after["greedy_switch_rate"] - before["greedy_switch_rate"],
        non_opportunity_action_changes=int((changed & active & ~eligible).sum().item()),
        opportunity_action_changes=int((changed & eligible).sum().item()),
        constraint_probability_decreased=probability_change < 0.0,
    )
    return result
