import torch
import datetime
from dataclasses import asdict
import hashlib
import json
import os
import random
import shutil
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from mappo_design import (
    GraphAttentionCritic,
    PacketConditionedCritic,
    RunningMeanStd as DesignRunningMeanStd,
    SharedCandidateActor,
    avoidable_switch_decisions,
    avoidable_switch_actual_logit_margin_penalties,
    avoidable_switch_logit_margin_penalties,
    avoidable_switch_probabilities,
    compute_gae,
    feasible_normalized_entropy,
    masked_standardize,
    projected_lagrange_multiplier_update,
    route_hysteresis_residual_dtype_cap,
    select_leo_validation_record,
    shuffled_transition_minibatches,
)


@dataclass
class Args:
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"
    """ Name of the environment"""
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 3
    """ Number of episodes to collect in each rollout"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 64
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    lr_decay: bool = True
    """If True, hold the base LR for the first half of training then linearly
    decay it to 12.5% over the second half. Stabilizes late training: 13/15
    prior runs peaked then degraded 1-9% as the policy memorized 20 cycled
    traffic seeds. The legacy linear_schedule() helper exists but was never
    wired in; this gates the now-connected schedule."""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD(λ) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = True
    """ Normalize the advantage if True"""
    normalize_return: bool = True
    """ Normalize the returns if True. DEFAULT CHANGED True: the team reward sums
    terms up to +-2.0 per slot, so raw return targets drift to -10..-100 and the
    MSE critic loss produced median raw gradients ~36-42 (max ~679), clipped on
    ~99.98% of updates. Normalizing the value target/prediction by the per-update-
    round (ret_mu, ret_std) is the root-cause fix for critic instability."""
    epochs: int = 3
    """ Number of training epochs"""
    num_minibatches: int = 4
    """Number of shuffled transition minibatches per PPO epoch."""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.01
    """ Entropy coefficient """
    log_every: int = 10
    """ Logging steps """
    clip_gradients: float = 1.0
    """Actor gradient norm limit; <=0 disables clipping. Raised 0.5->1.0: actor
    median pre-clip norm was ~0.54, so the old 0.5 threshold clipped 42-69% of
    updates, silently distorting the PPO trust region (mean KL ~0.0018 vs target
    0.02 was already over-conservative)."""
    critic_clip_gradients: float = 10.0
    """Critic gradient norm limit; <=0 disables clipping. Separate from the actor
    because critic gradients are far larger: pre-normalization median ~36-42 (max
    ~679), and ~1-10 after normalize_return=True (verified on CURRENT-20260717 and
    600-step smoke). 10.0 bounds true spikes while rarely clipping the healthy
    gradient. The old shared 0.5 threshold clipped 99.98% of critic updates."""
    normalization_epsilon: float = 1e-8
    """Numerical floor used by reward/advantage/return normalization."""
    target_kl: float = 0.02
    """Stop remaining PPO epochs when the measured KL exceeds this value."""
    candidate_shared_actor: bool = False
    """Use a permutation-equivariant shared candidate scorer."""
    leo_project_path: str = "F:/leo-routing-preliminary-matlab"
    """Directory containing cleanmarl_leo_wrapper.py for env_type=leo."""
    leo_variant: str = "proposed"
    """Canonical LEO method variant; full/no_lifetime are legacy aliases."""
    route_hysteresis_beta: float = 0.0
    """Additive cached-route actor-logit prior for leo_multi policies."""
    route_hysteresis_mode: str = "legacy_additive"
    """legacy_additive or the non-compensable decoupled_adaptive policy."""
    route_urgency_feature_index: int = 20
    """Frozen candidate feature index for packet waiting urgency."""
    route_class_2_feature_index: int = 23
    """Frozen candidate feature index for the reliability-priority class."""
    route_hysteresis_urgency_relief: float = 0.0
    """Fraction of hysteresis removed as normalized packet urgency reaches one."""
    route_hysteresis_class_2_relief: float = 0.0
    """Fraction of hysteresis removed for class-2 packets."""
    route_hysteresis_residual_init: float = 0.0
    """Initial learned cached-route residual bias for schema-v4 actors."""
    route_hysteresis_residual_cap: float = 0.0
    """Projected upper bound for the learned cached-route residual bias."""
    route_hysteresis_residual_parameterization: str = "scalar"
    """scalar (schema v4) or urgency_linear endpoint controller (schema v5)."""
    avoidable_switch_probability_coef: float = 0.0
    """Actor-loss weight on avoidable route-switch probability."""
    avoidable_switch_regularization_mode: str = "conditional_probability"
    """conditional_probability, greedy_logit_margin, or its isolated variant."""
    avoidable_switch_logit_margin: float = 0.0
    """Required stay-over-switch logit margin for either greedy mode."""
    avoidable_switch_reduction: str = "minibatch_conditional_mean"
    """Legacy minibatch mean or rollout_micro_mean for partition-stable weighting."""
    avoidable_switch_constraint_enabled: bool = False
    """Optimize QoS subject to a decision-level avoidable-switch-rate budget."""
    avoidable_switch_budget: float = 0.12
    """Maximum avoidable switches per pre-contention switch opportunity."""
    avoidable_switch_dual_learning_rate: float = 0.05
    """Projected dual-ascent step size, applied once per rollout."""
    avoidable_switch_dual_initial: float = 0.0
    """Initial non-negative Lagrange multiplier."""
    avoidable_switch_dual_max: float = 5.0
    """Upper projection bound for the Lagrange multiplier."""
    log_loss_component_gradients: bool = False
    """Log separate actor-primary and weighted-regularizer gradient norms."""
    eval_steps: int = 10
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 50
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 1
    """ Random seed"""
    checkpoint_dir: str = "checkpoints"
    """Directory used for periodic and final checkpoints."""
    save_every_steps: int = 5000
    """Save one periodic checkpoint after this many new environment steps."""
    train_seed_start: int = 9001
    """First fixed workload seed used during training."""
    train_seed_count: int = 200
    """Number of training workload seeds cycled independently of policy seed.
    Raised 20->200: with batch_size=4 and 30-slot episodes a 50K-step run sees
    ~1668 episodes, so 20 seeds were each cycled ~83x (severe overfitting -- 13/15
    runs peaked then degraded). 200 seeds gives ~8 reps each."""
    validation_seed_start: int = 10001
    """First held-out workload seed used for checkpoint selection."""
    validation_selection_mode: str = "legacy_lexicographic"
    """legacy_lexicographic or validation-only stability_constrained selection."""
    validation_delivery_tolerance: float = 0.0
    """Delivery loss allowed before stability breaks validation checkpoint ties."""
    validation_class_2_tolerance: float = 0.0
    """Class-2 delivery loss allowed before routing stability breaks ties."""
    run_tag: str = ""
    """Optional stable experiment label appended to the run directory."""
    resume_checkpoint: str = ""
    """Checkpoint from this trainer schema to resume at an episode boundary."""


class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values):
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
        self.var = (m_a + m_b + delta * delta * self.count * batch_count / total) / total
        self.count = total


def rollout_micro_minibatch_mean(
    values: torch.Tensor,
    *,
    rollout_denominator: int,
    active_minibatch_count: int,
) -> torch.Tensor:
    """Scale one minibatch sum so update-mean equals the rollout micro mean."""

    if rollout_denominator <= 0:
        raise ValueError("rollout denominator must be positive")
    if active_minibatch_count <= 0:
        raise ValueError("active minibatch count must be positive")
    return values.sum() * (
        float(active_minibatch_count) / float(rollout_denominator)
    )


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        state_space,
        action_space,
        normalize_reward=False,
        normalization_epsilon=1e-8,
        device="cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.normalization_epsilon = normalization_epsilon
        self.reward_rms = DesignRunningMeanStd()
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.obs_space)
        ).to(self.device)
        avail_actions = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.action_space)
        ).to(self.device)
        actions = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        log_probs = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        reward = torch.zeros(
            (self.buffer_size, max_length, self.num_agents)
        ).to(self.device)
        states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        next_states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        terminated = torch.zeros((self.buffer_size, max_length)).to(self.device)
        truncated = torch.zeros((self.buffer_size, max_length)).to(self.device)
        policy_active = torch.zeros(
            (self.buffer_size, max_length, self.num_agents), dtype=torch.bool
        ).to(self.device)
        avoidable_switch_cost = torch.zeros(
            (self.buffer_size, max_length, self.num_agents), dtype=torch.bool
        ).to(self.device)
        switch_opportunity = torch.zeros(
            (self.buffer_size, max_length, self.num_agents), dtype=torch.bool
        ).to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(
            self.device
        )
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            avail_actions[i, :length] = self.episodes[i]["avail_actions"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            next_states[i, :length] = self.episodes[i]["next_states"]
            terminated[i, :length] = self.episodes[i]["terminated"]
            truncated[i, :length] = self.episodes[i]["truncated"]
            policy_active[i, :length] = self.episodes[i]["policy_active"].bool()
            avoidable_switch_cost[i, :length] = self.episodes[i][
                "avoidable_switch_cost"
            ].bool()
            switch_opportunity[i, :length] = self.episodes[i][
                "switch_opportunity"
            ].bool()
            mask[i, :length] = 1
        if self.normalize_reward:
            reward_mask = mask.unsqueeze(-1).expand_as(reward)
            valid_reward = reward[reward_mask].detach().cpu().numpy()
            self.reward_rms.update(valid_reward)
            reward[reward_mask] = (reward[reward_mask] - self.reward_rms.mean) / np.sqrt(
                self.reward_rms.var + self.normalization_epsilon
            )
        self.episodes = [None] * self.buffer_size
        return (
            obs.float(),
            actions.long(),
            log_probs.float(),
            reward.float(),
            states.float(),
            next_states.float(),
            avail_actions.bool(),
            terminated.float(),
            truncated.float(),
            policy_active,
            avoidable_switch_cost,
            switch_opportunity,
            mask,
        )


class Actor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layer,
        output_dim,
        candidate_feature_dim=None,
        candidate_actor_spec=None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.layers = nn.ModuleList()
        if candidate_feature_dim is not None:
            if input_dim != output_dim * candidate_feature_dim:
                raise ValueError(
                    "candidate actor expects input_dim == output_dim * candidate_feature_dim"
                )
            self.shared_candidate_actor = SharedCandidateActor(
                candidate_feature_dim=candidate_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layer,
                route_switch_feature_index=(
                    candidate_actor_spec or {}
                ).get("route_switch_feature_index"),
                route_hysteresis_beta=(
                    candidate_actor_spec or {}
                ).get("route_hysteresis_beta", 0.0),
                route_hysteresis_mode=(
                    candidate_actor_spec or {}
                ).get("route_hysteresis_mode", "legacy_additive"),
                route_urgency_feature_index=(
                    candidate_actor_spec or {}
                ).get("route_urgency_feature_index"),
                route_class_2_feature_index=(
                    candidate_actor_spec or {}
                ).get("route_class_2_feature_index"),
                route_hysteresis_urgency_relief=(
                    candidate_actor_spec or {}
                ).get("route_hysteresis_urgency_relief", 0.0),
                route_hysteresis_class_2_relief=(
                    candidate_actor_spec or {}
                ).get("route_hysteresis_class_2_relief", 0.0),
                route_hysteresis_residual_init=(
                    candidate_actor_spec or {}
                ).get("route_hysteresis_residual_init", 0.0),
                route_hysteresis_residual_cap=(
                    candidate_actor_spec or {}
                ).get("route_hysteresis_residual_cap", 0.0),
                route_hysteresis_residual_parameterization={
                    "projected_nonnegative_scalar": "scalar",
                    "projected_nonnegative_urgency_linear_endpoints": (
                        "urgency_linear"
                    ),
                }.get(
                    (candidate_actor_spec or {}).get(
                        "route_hysteresis_residual_parameterization"
                    ),
                    "scalar",
                ),
            )
        else:
            self.layers.append(
                nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            )
            for _ in range(num_layer):
                self.layers.append(
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                )
            self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))

    def act(self, x, avail_action=None):
        logits = self.logits(x, avail_action)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action)

    def greedy(self, x, avail_action=None):
        return self.logits(x, avail_action).argmax(dim=-1)

    def logits(self, x, avail_action=None):
        if self.candidate_feature_dim is not None:
            candidates = x.reshape(
                *x.shape[:-1], self.output_dim, self.candidate_feature_dim
            )
            return self.shared_candidate_actor(candidates, avail_action)
        for layer in self.layers:
            x = layer(x)
        if avail_action is not None:
            x = x.masked_fill(~avail_action, -1e9)
        return x


class Critic(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layer, graph_spec=None
    ) -> None:
        super().__init__()
        if graph_spec and graph_spec.get("type") == "graph_attention":
            self.network = GraphAttentionCritic(
                n_nodes=graph_spec["n_nodes"],
                node_feature_dim=graph_spec["node_feature_dim"],
                edge_feature_dim=graph_spec["edge_feature_dim"],
                global_feature_dim=graph_spec["global_feature_dim"],
                hidden_dim=hidden_dim,
                num_layers=num_layer,
            )
            if self.network.state_size != input_dim:
                raise ValueError("graph critic spec does not match environment state size")
        else:
            self.network = PacketConditionedCritic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layer,
            )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    kwargs = dict(kwargs)
    if env_type == "pz":
        from env.pettingzoo_wrapper import PettingZooWrapper

        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    elif env_type == "smaclite":
        from env.smaclite_wrapper import SMACliteWrapper

        env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "lbf":
        from env.lbf import LBFWrapper

        env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "leo":
        project_path = kwargs.pop(
            "project_path",
            os.environ.get("LEO_ROUTING_PROJECT", "F:/leo-routing-preliminary-matlab"),
        )
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
        from cleanmarl_leo_wrapper import CleanMARLLeoWrapper

        env = CleanMARLLeoWrapper(
            scenario=env_name, seed=kwargs.pop("seed", 11)
        )
    elif env_type == "leo_multi":
        project_path = kwargs.pop(
            "project_path",
            os.environ.get("LEO_ROUTING_PROJECT", "F:/leo-routing-preliminary-matlab"),
        )
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
        from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper

        env = CleanMARLLeoMultiAgentWrapper(
            scenario=env_name,
            seed=kwargs.pop("seed", 11),
            variant=kwargs.pop("variant", "proposed"),
        )
    else:
        raise ValueError(f"unknown env_type: {env_type}")

    return env


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads if g is not None]
    if not norms:
        return torch.tensor(0.0)
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


def reset_with_seed(env, seed=None):
    if seed is None:
        return env.reset()
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    """Write a checkpoint atomically so interruption cannot corrupt the target."""

    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def restore_jsonl_checkpoint_boundary(path, expected_size_bytes):
    """Validate a saved JSONL prefix and discard records beyond that boundary."""

    if isinstance(expected_size_bytes, bool) or not isinstance(
        expected_size_bytes, (int, np.integer)
    ):
        raise ValueError("metrics checkpoint offset must be an integer")
    expected_size_bytes = int(expected_size_bytes)
    if expected_size_bytes < 0:
        raise ValueError("metrics checkpoint offset must be non-negative")
    if not os.path.exists(path):
        if expected_size_bytes != 0:
            raise ValueError("metrics file is shorter than its checkpoint boundary")
        return
    if not os.path.isfile(path):
        raise ValueError("metrics path is not a regular file")

    actual_size_bytes = os.path.getsize(path)
    if actual_size_bytes < expected_size_bytes:
        raise ValueError("metrics file is shorter than its checkpoint boundary")
    with open(path, "rb") as file:
        checkpoint_prefix = file.read(expected_size_bytes)
    if len(checkpoint_prefix) != expected_size_bytes:
        raise ValueError("could not read the complete metrics checkpoint prefix")
    if checkpoint_prefix and not checkpoint_prefix.endswith(b"\n"):
        raise ValueError("metrics checkpoint offset is not at a JSONL boundary")
    try:
        decoded_prefix = checkpoint_prefix.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("metrics checkpoint prefix is not valid UTF-8") from exc
    for line_number, line in enumerate(decoded_prefix.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"metrics checkpoint contains blank line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"metrics checkpoint contains invalid JSON on line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"metrics checkpoint record {line_number} is not a mapping"
            )

    if actual_size_bytes > expected_size_bytes:
        with open(path, "r+b") as file:
            file.truncate(expected_size_bytes)
            file.flush()
            os.fsync(file.fileno())


def validate_resume_total_timesteps(
    saved_args,
    current_args,
    *,
    saved_step,
):
    """Validate a behavior-preserving extension of a stopped training run."""

    saved_total = saved_args.get("total_timesteps")
    current_total = current_args.get("total_timesteps")
    for value, context in (
        (saved_total, "checkpoint"),
        (current_total, "current invocation"),
        (saved_step, "checkpoint step"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"resume {context} total_timesteps must be an integer")
    saved_total = int(saved_total)
    current_total = int(current_total)
    saved_step = int(saved_step)
    if saved_total == current_total:
        return
    if (
        saved_args.get("lr_decay") is not False
        or current_args.get("lr_decay") is not False
    ):
        raise ValueError(
            "resume total_timesteps cannot change while learning-rate decay is enabled"
        )
    if current_total <= saved_total:
        raise ValueError("resume total_timesteps may only be extended")
    if current_total <= saved_step:
        raise ValueError("resume total_timesteps must exceed the saved step")


def advance_avoidable_switch_dual(
    *,
    multiplier,
    update_count,
    skipped_zero_opportunity_count,
    cost_count,
    opportunity_count,
    budget,
    learning_rate,
    maximum,
):
    """Return one immutable rollout-level dual transition.

    A rollout without switch opportunities carries no rate observation. It
    therefore increments only the skip counter; in particular it does not call
    the projected update and cannot decay the multiplier toward zero.
    """

    integer_fields = {
        "update_count": update_count,
        "skipped_zero_opportunity_count": skipped_zero_opportunity_count,
        "cost_count": cost_count,
        "opportunity_count": opportunity_count,
    }
    for field, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"dual {field} must be an integer")
        if int(value) < 0:
            raise ValueError(f"dual {field} must be non-negative")
    cost_count = int(cost_count)
    opportunity_count = int(opportunity_count)
    if cost_count > opportunity_count:
        raise ValueError("dual cost count exceeds opportunity count")

    numeric_fields = {
        "multiplier": multiplier,
        "budget": budget,
        "learning_rate": learning_rate,
        "maximum": maximum,
    }
    for field, value in numeric_fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"dual {field} must be finite")
        if not np.isfinite(value):
            raise ValueError(f"dual {field} must be finite")
    multiplier = float(multiplier)
    budget = float(budget)
    learning_rate = float(learning_rate)
    maximum = float(maximum)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("dual budget must be in [0, 1]")
    if learning_rate <= 0.0:
        raise ValueError("dual learning rate must be positive")
    if maximum <= 0.0:
        raise ValueError("dual maximum must be positive")
    if not 0.0 <= multiplier <= maximum:
        raise ValueError("dual multiplier is outside its projection")

    if opportunity_count == 0:
        return {
            "multiplier": multiplier,
            "update_count": int(update_count),
            "skipped_zero_opportunity_count": (
                int(skipped_zero_opportunity_count) + 1
            ),
            "empirical_rate": None,
            "violation": None,
            "update_skipped": True,
            "projection_hit": False,
        }

    empirical_rate = cost_count / opportunity_count
    next_multiplier = projected_lagrange_multiplier_update(
        multiplier,
        empirical_rate,
        budget,
        learning_rate,
        maximum=maximum,
    )
    return {
        "multiplier": float(next_multiplier),
        "update_count": int(update_count) + 1,
        "skipped_zero_opportunity_count": int(
            skipped_zero_opportunity_count
        ),
        "empirical_rate": float(empirical_rate),
        "violation": float(empirical_rate - budget),
        "update_skipped": False,
        "projection_hit": float(next_multiplier) in {0.0, maximum},
    }


def promote_validation_checkpoint(source_path, destination_path, expected_step):
    """Atomically promote one audited validation candidate to the best path."""

    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("step") != expected_step:
        raise ValueError("selected validation checkpoint step mismatch")
    source_hash = checkpoint_sha256(source_path)
    temporary_path = f"{destination_path}.tmp-{os.getpid()}"
    try:
        shutil.copy2(source_path, temporary_path)
        if checkpoint_sha256(temporary_path) != source_hash:
            raise RuntimeError("validation checkpoint copy hash mismatch")
        os.replace(temporary_path, destination_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    if checkpoint_sha256(destination_path) != source_hash:
        raise RuntimeError("promoted validation checkpoint hash mismatch")
    return source_hash


if __name__ == "__main__":
    import tyro
    from torch.utils.tensorboard import SummaryWriter

    args = tyro.cli(Args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ## import the environment
    kwargs = {}
    if args.env_type in {"leo", "leo_multi"}:
        kwargs["project_path"] = args.leo_project_path
        kwargs["seed"] = args.seed
        kwargs["variant"] = args.leo_variant
    env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    eval_kwargs = dict(kwargs)
    if args.env_type in {"leo", "leo_multi"}:
        eval_kwargs["seed"] = args.validation_seed_start
    eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=eval_kwargs,
    )

    ## Initialize the actor, critic and target-critic networks
    candidate_feature_dim = None
    candidate_actor_spec = None
    if not np.isfinite(args.route_hysteresis_beta) or args.route_hysteresis_beta < 0.0:
        raise ValueError("route_hysteresis_beta must be finite and non-negative")
    if args.route_hysteresis_mode not in {
        "legacy_additive",
        "decoupled_adaptive",
    }:
        raise ValueError("unsupported route_hysteresis_mode")
    if args.validation_selection_mode not in {
        "legacy_lexicographic",
        "stability_constrained",
        "avoidable_stability_constrained",
        "avoidable_switch_budget_constrained",
    }:
        raise ValueError("unsupported validation_selection_mode")
    for value, field in (
        (args.validation_delivery_tolerance, "validation_delivery_tolerance"),
        (args.validation_class_2_tolerance, "validation_class_2_tolerance"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    if (
        args.route_hysteresis_mode == "decoupled_adaptive"
        and args.route_hysteresis_beta <= 0.0
    ):
        raise ValueError("decoupled_adaptive route hysteresis requires positive beta")
    if (
        args.validation_selection_mode in {
            "stability_constrained",
            "avoidable_stability_constrained",
            "avoidable_switch_budget_constrained",
        }
        and args.env_type != "leo_multi"
    ):
        raise ValueError("stability-constrained validation is supported only for leo_multi")
    if (
        args.validation_selection_mode in {
            "stability_constrained",
            "avoidable_stability_constrained",
        }
        and args.route_hysteresis_mode != "decoupled_adaptive"
    ):
        raise ValueError(
            "stability-constrained validation requires schema-v2 adaptive hysteresis"
        )
    if (
        not np.isfinite(args.avoidable_switch_probability_coef)
        or args.avoidable_switch_probability_coef < 0.0
    ):
        raise ValueError(
            "avoidable_switch_probability_coef must be finite and non-negative"
        )
    if args.avoidable_switch_regularization_mode not in {
        "conditional_probability",
        "greedy_logit_margin",
        "isolated_greedy_logit_margin",
    }:
        raise ValueError("unsupported avoidable-switch regularization mode")
    if args.avoidable_switch_reduction not in {
        "minibatch_conditional_mean",
        "rollout_micro_mean",
    }:
        raise ValueError("unsupported avoidable-switch reduction")
    if (
        not np.isfinite(args.avoidable_switch_logit_margin)
        or args.avoidable_switch_logit_margin < 0.0
    ):
        raise ValueError(
            "avoidable_switch_logit_margin must be finite and non-negative"
        )
    if args.avoidable_switch_probability_coef == 0.0 and (
        args.avoidable_switch_regularization_mode != "conditional_probability"
        or args.avoidable_switch_logit_margin != 0.0
        or (
            not args.avoidable_switch_constraint_enabled
            and args.avoidable_switch_reduction != "minibatch_conditional_mean"
        )
    ):
        raise ValueError(
            "disabled avoidable-switch regularization requires default contracts"
        )
    if (
        args.avoidable_switch_regularization_mode == "conditional_probability"
        and args.avoidable_switch_logit_margin != 0.0
    ):
        raise ValueError(
            "conditional-probability regularization cannot use a logit margin"
        )
    if (
        args.avoidable_switch_regularization_mode
        in {"greedy_logit_margin", "isolated_greedy_logit_margin"}
        and args.avoidable_switch_logit_margin <= 0.0
    ):
        raise ValueError(
            "greedy-logit-margin regularization requires a positive margin"
        )
    if (
        args.avoidable_switch_probability_coef > 0.0
        and args.route_hysteresis_mode != "decoupled_adaptive"
    ):
        raise ValueError(
            "avoidable switch regularization requires decoupled_adaptive hysteresis"
        )
    if (
        args.validation_selection_mode == "avoidable_stability_constrained"
        and args.avoidable_switch_probability_coef <= 0.0
    ):
        raise ValueError(
            "avoidable-stability validation requires positive switch regularization"
        )
    for value, field in (
        (args.route_hysteresis_urgency_relief, "route_hysteresis_urgency_relief"),
        (args.route_hysteresis_class_2_relief, "route_hysteresis_class_2_relief"),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite and in [0, 1]")
    for value, field in (
        (args.route_hysteresis_residual_init, "route_hysteresis_residual_init"),
        (args.route_hysteresis_residual_cap, "route_hysteresis_residual_cap"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    if args.route_hysteresis_residual_init > args.route_hysteresis_residual_cap:
        raise ValueError(
            "route_hysteresis_residual_init cannot exceed its cap"
        )
    residual_enabled = args.route_hysteresis_residual_cap > 0.0
    if args.route_hysteresis_residual_parameterization not in {
        "scalar",
        "urgency_linear",
    }:
        raise ValueError("unsupported route hysteresis residual parameterization")
    if (
        args.route_hysteresis_residual_parameterization == "urgency_linear"
        and not residual_enabled
    ):
        raise ValueError("urgency-linear residual controller requires a positive cap")
    if (
        args.route_hysteresis_residual_parameterization == "urgency_linear"
        and args.avoidable_switch_reduction != "rollout_micro_mean"
    ):
        raise ValueError(
            "urgency-linear residual controller requires rollout micro reduction"
        )
    if (
        args.avoidable_switch_reduction == "rollout_micro_mean"
        and args.route_hysteresis_residual_parameterization != "urgency_linear"
        and not args.avoidable_switch_constraint_enabled
    ):
        raise ValueError(
            "rollout micro reduction requires the urgency-linear residual controller"
        )
    for value, field in (
        (args.avoidable_switch_budget, "avoidable_switch_budget"),
        (
            args.avoidable_switch_dual_learning_rate,
            "avoidable_switch_dual_learning_rate",
        ),
        (args.avoidable_switch_dual_initial, "avoidable_switch_dual_initial"),
        (args.avoidable_switch_dual_max, "avoidable_switch_dual_max"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be finite")
        if not np.isfinite(value):
            raise ValueError(f"{field} must be finite")
    if not 0.0 <= args.avoidable_switch_budget <= 1.0:
        raise ValueError("avoidable_switch_budget must be in [0, 1]")
    if args.avoidable_switch_dual_learning_rate <= 0.0:
        raise ValueError("avoidable_switch_dual_learning_rate must be positive")
    if args.avoidable_switch_dual_max <= 0.0:
        raise ValueError("avoidable_switch_dual_max must be positive")
    if not 0.0 <= args.avoidable_switch_dual_initial <= args.avoidable_switch_dual_max:
        raise ValueError("avoidable_switch_dual_initial is outside its projection")
    if not isinstance(args.resume_checkpoint, str):
        raise ValueError("resume_checkpoint must be a string")
    if args.avoidable_switch_constraint_enabled:
        if args.env_type != "leo_multi":
            raise ValueError("avoidable-switch constraint is supported only for leo_multi")
        if getattr(env.variant_definition, "switch_reward", True):
            raise ValueError(
                "avoidable-switch constraint requires a QoS-only reward objective"
            )
        if not getattr(env.variant_definition, "packet_context", False):
            raise ValueError("avoidable-switch constraint requires route-cache context")
        if args.avoidable_switch_probability_coef != 0.0:
            raise ValueError(
                "adaptive constraint and fixed avoidable-switch regularization are mutually exclusive"
            )
        if (
            args.avoidable_switch_regularization_mode != "conditional_probability"
            or args.avoidable_switch_logit_margin != 0.0
            or args.avoidable_switch_reduction != "rollout_micro_mean"
        ):
            raise ValueError(
                "avoidable-switch constraint requires the probability micro surrogate"
            )
        if (
            args.route_hysteresis_beta != 0.0
            or args.route_hysteresis_mode != "legacy_additive"
            or residual_enabled
        ):
            raise ValueError(
                "avoidable-switch constraint cannot be combined with route hysteresis"
            )
        if args.validation_selection_mode != "avoidable_switch_budget_constrained":
            raise ValueError(
                "avoidable-switch constraint requires budget-constrained validation"
            )
    elif args.validation_selection_mode == "avoidable_switch_budget_constrained":
        raise ValueError(
            "budget-constrained validation requires the avoidable-switch constraint"
        )
    isolated_regularizer = (
        args.avoidable_switch_regularization_mode
        == "isolated_greedy_logit_margin"
    )
    if residual_enabled and not isolated_regularizer:
        raise ValueError(
            "positive route hysteresis residual cap requires isolated regularization"
        )
    if isolated_regularizer and not residual_enabled:
        raise ValueError(
            "isolated switch regularization requires a positive residual cap"
        )
    if residual_enabled and args.route_hysteresis_mode != "decoupled_adaptive":
        raise ValueError(
            "route hysteresis residual is supported only for decoupled_adaptive"
        )
    if residual_enabled and args.avoidable_switch_probability_coef <= 0.0:
        raise ValueError(
            "route hysteresis residual requires positive switch regularization"
        )
    if args.route_hysteresis_mode == "legacy_additive" and (
        args.route_hysteresis_urgency_relief != 0.0
        or args.route_hysteresis_class_2_relief != 0.0
    ):
        raise ValueError("legacy_additive route hysteresis cannot use adaptive relief")
    if args.route_hysteresis_mode == "legacy_additive" and (
        args.route_urgency_feature_index != 20
        or args.route_class_2_feature_index != 23
    ):
        raise ValueError(
            "legacy_additive requires the default unused adaptive feature indices"
        )
    if args.route_hysteresis_beta > 0.0 and args.env_type != "leo_multi":
        raise ValueError("positive route hysteresis is supported only for leo_multi")
    if args.route_hysteresis_beta > 0.0 and not getattr(
        getattr(env, "variant_definition", None), "packet_context", False
    ):
        raise ValueError(
            "positive route hysteresis requires the packet-context route cache"
        )
    if args.env_type in {"leo", "leo_multi"} or args.candidate_shared_actor:
        getter = getattr(env, "get_candidate_feature_dim", None)
        if getter is None:
            raise ValueError(
                "candidate_shared_actor requires env.get_candidate_feature_dim()"
            )
        candidate_feature_dim = getter()
        if args.env_type == "leo_multi":
            switch_index_getter = getattr(env, "get_route_switch_feature_index", None)
            if switch_index_getter is None:
                raise ValueError("leo_multi environment has no route-switch schema")
            candidate_actor_spec = {
                "schema_version": 1,
                "type": "shared_candidate_actor",
                "route_switch_feature_index": int(switch_index_getter()),
                "route_hysteresis_beta": float(args.route_hysteresis_beta),
            }
            if args.route_hysteresis_mode == "decoupled_adaptive":
                urgency_index_getter = getattr(
                    env, "get_route_urgency_feature_index", None
                )
                class_2_index_getter = getattr(
                    env, "get_route_class_2_feature_index", None
                )
                if urgency_index_getter is None or class_2_index_getter is None:
                    raise ValueError("leo_multi environment has no adaptive route schema")
                feature_schema_getter = getattr(
                    env, "get_candidate_feature_schema", None
                )
                if feature_schema_getter is None:
                    raise ValueError("leo_multi environment has no candidate feature schema")
                feature_schema = feature_schema_getter()
                urgency_index = int(urgency_index_getter())
                class_2_index = int(class_2_index_getter())
                if args.route_urgency_feature_index != urgency_index:
                    raise ValueError("route urgency feature index disagrees with environment")
                if args.route_class_2_feature_index != class_2_index:
                    raise ValueError("route class-2 feature index disagrees with environment")
                candidate_actor_spec = {
                    **candidate_actor_spec,
                    "schema_version": 2,
                    "route_hysteresis_mode": "decoupled_adaptive",
                    "route_urgency_feature_index": urgency_index,
                    "route_class_2_feature_index": class_2_index,
                    "route_hysteresis_urgency_relief": float(
                        args.route_hysteresis_urgency_relief
                    ),
                    "route_hysteresis_class_2_relief": float(
                        args.route_hysteresis_class_2_relief
                    ),
                    "candidate_feature_schema_id": feature_schema["schema_id"],
                    "candidate_feature_schema_sha256": feature_schema["sha256"],
                }
                if args.avoidable_switch_probability_coef > 0.0:
                    candidate_actor_spec = {
                        **candidate_actor_spec,
                        "schema_version": 3,
                        "avoidable_switch_probability_coef": float(
                            args.avoidable_switch_probability_coef
                        ),
                    }
                    if residual_enabled:
                        candidate_actor_spec = {
                            **candidate_actor_spec,
                            "schema_version": (
                                5
                                if args.route_hysteresis_residual_parameterization
                                == "urgency_linear"
                                else 4
                            ),
                            "route_hysteresis_residual_parameterization": (
                                "projected_nonnegative_urgency_linear_endpoints"
                                if args.route_hysteresis_residual_parameterization
                                == "urgency_linear"
                                else "projected_nonnegative_scalar"
                            ),
                            "route_hysteresis_residual_init": float(
                                args.route_hysteresis_residual_init
                            ),
                            "route_hysteresis_residual_cap": float(
                                args.route_hysteresis_residual_cap
                            ),
                        }
    switch_regularizer_spec = None
    if args.avoidable_switch_probability_coef > 0.0:
        endpoint_controller = (
            candidate_actor_spec.get("schema_version") == 5
        )
        rollout_micro_reduction = (
            args.avoidable_switch_reduction == "rollout_micro_mean"
        )
        switch_regularizer_spec = {
            "schema_version": (
                3
                if endpoint_controller
                else (
                    2
                    if args.avoidable_switch_regularization_mode
                    == "isolated_greedy_logit_margin"
                    else 1
                )
            ),
            "mode": args.avoidable_switch_regularization_mode,
            "coefficient": float(args.avoidable_switch_probability_coef),
            "logit_margin": float(args.avoidable_switch_logit_margin),
            "reduction": (
                "rollout_micro_mean_over_eligible_pre_contention_decisions"
                if rollout_micro_reduction
                else "conditional_mean_over_eligible_active_decisions"
            ),
            "no_op_action_index": 0,
            "route_switch_feature_index": int(
                candidate_actor_spec["route_switch_feature_index"]
            ),
            "route_urgency_feature_index": int(
                candidate_actor_spec["route_urgency_feature_index"]
            ),
            "route_class_2_feature_index": int(
                candidate_actor_spec["route_class_2_feature_index"]
            ),
            "route_hysteresis_beta": float(
                candidate_actor_spec["route_hysteresis_beta"]
            ),
            "route_hysteresis_urgency_relief": float(
                candidate_actor_spec["route_hysteresis_urgency_relief"]
            ),
            "route_hysteresis_class_2_relief": float(
                candidate_actor_spec["route_hysteresis_class_2_relief"]
            ),
            "relief_weighting": (
                "none"
                if args.avoidable_switch_regularization_mode
                == "conditional_probability"
                else (
                    "actual_policy_logits_no_extra_weighting"
                    if args.avoidable_switch_regularization_mode
                    == "isolated_greedy_logit_margin"
                    else "restore_full_hysteresis_before_hinge_then_multiply_adaptive_relief_scale"
                )
            ),
            **(
                {
                    "logit_source": "actual_policy_logits",
                    "gradient_scope": (
                        "cached_route_residual_endpoints_only"
                        if endpoint_controller
                        else "cached_route_residual_only"
                    ),
                    **(
                        {
                            "eligibility_stage": "pre_contention",
                            "minibatch_weighting": (
                                "eligible_sum_scaled_to_rollout_micro_mean"
                            ),
                        }
                        if endpoint_controller
                        else {}
                    ),
                }
                if args.avoidable_switch_regularization_mode
                == "isolated_greedy_logit_margin"
                else {}
            ),
        }
    switch_constraint_spec = None
    if args.avoidable_switch_constraint_enabled:
        feature_schema = env.get_candidate_feature_schema()
        switch_constraint_spec = {
            "schema_version": 1,
            "type": "projected_lagrangian_avoidable_switch_rate",
            "cost_definition": (
                "policy_selects_different_next_hop_while_cached_next_hop_and_"
                "at_least_one_alternative_are_feasible"
            ),
            "decision_stage": "pre_contention",
            "normalization": "rollout_micro_ratio_of_sums_over_opportunities",
            "forced_reroutes_counted": False,
            "first_route_counted": False,
            "zero_opportunity_update": "skip",
            "actor_surrogate": (
                "conditional_switch_probability_micro_mean_on_rollout_occupancy"
            ),
            "dual_update_timing": "once_after_each_completed_ppo_rollout",
            "budget": float(args.avoidable_switch_budget),
            "dual_learning_rate": float(
                args.avoidable_switch_dual_learning_rate
            ),
            "dual_initial": float(args.avoidable_switch_dual_initial),
            "dual_projection": [0.0, float(args.avoidable_switch_dual_max)],
            "no_op_action_index": 0,
            "route_switch_feature_index": int(
                candidate_actor_spec["route_switch_feature_index"]
            ),
            "candidate_feature_schema_id": feature_schema["schema_id"],
            "candidate_feature_schema_sha256": feature_schema["sha256"],
            "reward_objective": "qos_only_without_switch_reward",
            "leo_variant": env.variant,
        }
    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
        candidate_feature_dim=candidate_feature_dim,
        candidate_actor_spec=candidate_actor_spec,
    ).to(device)
    critic_spec_getter = getattr(env, "get_critic_spec", None)
    critic_spec = critic_spec_getter() if critic_spec_getter else None
    critic = Critic(
        input_dim=env.get_state_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
        graph_spec=critic_spec,
    ).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"__{args.run_tag}" if args.run_tag else ""
    resume_checkpoint = args.resume_checkpoint.strip()
    if resume_checkpoint:
        resume_checkpoint = os.path.abspath(resume_checkpoint)
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
        checkpoint_root = os.path.dirname(resume_checkpoint)
        run_name = os.path.basename(checkpoint_root)
    else:
        run_name = (
            f"{args.env_type}__{args.env_name}__seed-{args.seed}__{time_token}{tag}"
        )
        checkpoint_root = os.path.join(args.checkpoint_dir, run_name)
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MAPPO-{run_name}",
        )
    writer = SummaryWriter(f"runs/MAPPO-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    os.makedirs(checkpoint_root, exist_ok=True)
    metrics_path = os.path.join(checkpoint_root, "training_metrics.jsonl")
    if not resume_checkpoint:
        with open(
            os.path.join(checkpoint_root, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(asdict(args), file, indent=2, ensure_ascii=False)

    def candidate_actor_learned_state():
        shared_actor = getattr(actor, "shared_candidate_actor", None)
        schema_version = int((candidate_actor_spec or {}).get("schema_version", 0))
        parameter_names = {
            4: ("route_hysteresis_residual_bias",),
            5: (
                "route_hysteresis_residual_calm_bias",
                "route_hysteresis_residual_urgent_bias",
            ),
        }.get(schema_version, ())
        if not parameter_names:
            return None
        cap = float(args.route_hysteresis_residual_cap)
        learned_state = {}
        for parameter_name in parameter_names:
            residual = getattr(shared_actor, parameter_name, None)
            if not isinstance(residual, torch.Tensor) or residual.numel() != 1:
                raise RuntimeError("cached-route residual parameter is missing")
            value = float(residual.detach().cpu().item())
            dtype_cap = route_hysteresis_residual_dtype_cap(
                cap,
                dtype=residual.dtype,
            )
            if not np.isfinite(value) or not 0.0 <= value <= dtype_cap:
                raise RuntimeError("cached-route residual escaped its projection")
            learned_state[parameter_name] = value
        return learned_state

    def current_switch_constraint_state():
        if not args.avoidable_switch_constraint_enabled:
            return None
        return {
            "schema_version": 1,
            "multiplier": float(dual_multiplier),
            "update_count": int(dual_update_count),
            "skipped_zero_opportunity_count": int(
                dual_skipped_zero_opportunity_count
            ),
            "cumulative_cost": int(dual_cumulative_cost),
            "cumulative_opportunity": int(dual_cumulative_opportunity),
            "last_rollout_cost": int(dual_last_cost),
            "last_rollout_opportunity": int(dual_last_opportunity),
            "last_rollout_rate": (
                None if dual_last_rate is None else float(dual_last_rate)
            ),
            "last_rollout_violation": (
                None if dual_last_violation is None else float(dual_last_violation)
            ),
        }

    def current_trainer_state():
        return {
            "schema_version": 1,
            "step": int(step),
            "training_step": int(training_step),
            "update_round": int(update_round),
            "num_episodes": int(num_episodes),
            "next_save_step": int(next_save_step),
            "best_validation_score": best_validation_score,
            "best_validation_record": best_validation_record,
            "validation_records": list(validation_records),
            "reward_rms": {
                "mean": float(rb.reward_rms.mean),
                "var": float(rb.reward_rms.var),
                "count": float(rb.reward_rms.count),
            },
            "rollout_log_accumulator": {
                "episode_rewards": [float(value) for value in ep_rewards],
                "episode_lengths": [int(value) for value in ep_lengths],
                "episode_stats": list(ep_stats),
            },
            "metrics_file_size_bytes": (
                int(os.path.getsize(metrics_path))
                if os.path.exists(metrics_path)
                else 0
            ),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
        }

    def save_checkpoint(label, current_step, *, exact_resume_boundary=False):
        if int(current_step) != int(step):
            raise RuntimeError("checkpoint step disagrees with trainer state")
        path = os.path.join(checkpoint_root, f"{label}.pt")
        payload = {
            "step": current_step,
            "resume_boundary": (
                "completed_update"
                if exact_resume_boundary
                else "inference_snapshot_only"
            ),
            "args": asdict(args),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "obs_size": env.get_obs_size(),
            "state_size": env.get_state_size(),
            "action_size": env.get_action_size(),
            "n_agents": env.n_agents,
            "candidate_feature_dim": candidate_feature_dim,
            "candidate_actor_spec": candidate_actor_spec,
            "switch_regularizer_spec": switch_regularizer_spec,
            "switch_constraint_spec": switch_constraint_spec,
            "switch_constraint_state": current_switch_constraint_state(),
            "critic_spec": critic_spec,
            "validation_selection_spec": validation_selection_spec,
            "trainer_state": current_trainer_state(),
        }
        learned_state = candidate_actor_learned_state()
        if learned_state is not None:
            payload["candidate_actor_learned_state"] = learned_state
        atomic_torch_save(payload, path)
        return path

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=env.get_obs_size(),
        state_space=env.get_state_size(),
        action_space=env.get_action_size(),
        num_agents=env.n_agents,
        normalize_reward=args.normalize_reward,
        normalization_epsilon=args.normalization_epsilon,
        device=device,
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    training_step = 0
    update_round = 0
    num_episodes = 0
    step = 0
    next_save_step = args.save_every_steps
    dual_multiplier = float(args.avoidable_switch_dual_initial)
    dual_update_count = 0
    dual_skipped_zero_opportunity_count = 0
    dual_cumulative_cost = 0
    dual_cumulative_opportunity = 0
    dual_last_cost = 0
    dual_last_opportunity = 0
    dual_last_rate = None
    dual_last_violation = None
    best_validation_score = None
    best_validation_record = None
    validation_records = []
    deferred_validation_selection = bool(
        args.avoidable_switch_constraint_enabled
        or (
            candidate_actor_spec
            and candidate_actor_spec.get("schema_version") in {2, 3, 4, 5}
        )
    )
    validation_selection_spec = {
        "schema_version": int(
            candidate_actor_spec.get("schema_version", 0)
            if candidate_actor_spec
            else 0
        ),
        "mode": args.validation_selection_mode,
        "delivery_tolerance": float(args.validation_delivery_tolerance),
        "class_2_tolerance": float(args.validation_class_2_tolerance),
        "switch_budget": (
            float(args.avoidable_switch_budget)
            if args.avoidable_switch_constraint_enabled
            else None
        ),
        "source": "validation_only",
        "test_panel_consulted": False,
        "validation_seed_start": int(args.validation_seed_start),
        "validation_episodes": int(args.num_eval_ep),
        "metric_aggregation": {
            "delivery_ratio": "macro_mean_over_validation_episodes",
            "class_2_delivery_ratio": "macro_mean_over_validation_episodes",
            "routing_switches": "mean_of_episode_total_switch_counts",
        },
        "selection_timing": (
            "after_all_validation_candidates"
            if deferred_validation_selection
            else "online_legacy_compatible"
        ),
    }
    if (
        candidate_actor_spec
        and candidate_actor_spec.get("schema_version") in {3, 4, 5}
    ):
        validation_selection_spec["metric_aggregation"][
            "avoidable_switch_rate"
        ] = "micro_ratio_over_validation_switch_opportunities"
    if args.avoidable_switch_constraint_enabled:
        validation_selection_spec["metric_aggregation"].update(
            decision_avoidable_switch_rate=(
                "micro_ratio_over_pre_contention_validation_opportunities"
            ),
            decision_avoidable_switches=(
                "sum_over_pre_contention_validation_decisions"
            ),
            decision_switch_opportunities=(
                "sum_over_pre_contention_validation_decisions"
            ),
        )
    if resume_checkpoint:
        resume_payload = torch.load(
            resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(resume_payload, dict):
            raise ValueError("resume checkpoint payload must be a mapping")
        saved_args = resume_payload.get("args")
        if not isinstance(saved_args, dict):
            raise ValueError("resume checkpoint has no trainer args")
        mutable_resume_fields = {
            "checkpoint_dir",
            "resume_checkpoint",
            "run_tag",
            "use_wnb",
            "wnb_entity",
            "wnb_project",
        }
        current_args = asdict(args)
        mismatches = {
            field: (saved_args.get(field), current_args[field])
            for field in current_args
            if field not in mutable_resume_fields
            and field != "total_timesteps"
            and saved_args.get(field) != current_args[field]
        }
        if mismatches:
            raise ValueError(f"resume trainer args mismatch: {mismatches}")
        structural_contract = {
            "obs_size": env.get_obs_size(),
            "state_size": env.get_state_size(),
            "action_size": env.get_action_size(),
            "n_agents": env.n_agents,
            "candidate_feature_dim": candidate_feature_dim,
            "candidate_actor_spec": candidate_actor_spec,
            "switch_regularizer_spec": switch_regularizer_spec,
            "switch_constraint_spec": switch_constraint_spec,
            "critic_spec": critic_spec,
            "validation_selection_spec": validation_selection_spec,
        }
        structural_mismatches = {
            field: (resume_payload.get(field), expected)
            for field, expected in structural_contract.items()
            if resume_payload.get(field) != expected
        }
        if structural_mismatches:
            raise ValueError(
                f"resume checkpoint contract mismatch: {structural_mismatches}"
            )
        if resume_payload.get("resume_boundary") != "completed_update":
            raise ValueError(
                "checkpoint is not at an exact completed-update resume boundary"
            )
        trainer_state = resume_payload.get("trainer_state")
        trainer_schema_version = (
            trainer_state.get("schema_version")
            if isinstance(trainer_state, dict)
            else None
        )
        if (
            not isinstance(trainer_state, dict)
            or isinstance(trainer_schema_version, bool)
            or not isinstance(trainer_schema_version, (int, np.integer))
            or int(trainer_schema_version) != 1
        ):
            raise ValueError("checkpoint is not exact-resume capable")

        def saved_nonnegative_integer(container, field, context):
            value = container.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"resume {context} {field} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"resume {context} {field} must be non-negative")
            return value

        def saved_finite_number(container, field, context):
            value = container.get(field)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"resume {context} {field} must be finite")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"resume {context} {field} must be finite")
            return value

        payload_step = saved_nonnegative_integer(
            resume_payload, "step", "checkpoint"
        )
        saved_step = saved_nonnegative_integer(trainer_state, "step", "trainer state")
        if payload_step != saved_step:
            raise ValueError("resume checkpoint and trainer-state steps disagree")
        validate_resume_total_timesteps(
            saved_args,
            current_args,
            saved_step=saved_step,
        )
        saved_training_step = saved_nonnegative_integer(
            trainer_state, "training_step", "trainer state"
        )
        saved_update_round = saved_nonnegative_integer(
            trainer_state, "update_round", "trainer state"
        )
        saved_num_episodes = saved_nonnegative_integer(
            trainer_state, "num_episodes", "trainer state"
        )
        saved_next_save_step = saved_nonnegative_integer(
            trainer_state, "next_save_step", "trainer state"
        )
        if saved_num_episodes != saved_update_round * args.batch_size:
            raise ValueError(
                "resume episode count does not match completed rollout updates"
            )
        if saved_step < saved_num_episodes:
            raise ValueError("resume environment step count is smaller than episodes")
        if args.save_every_steps > 0:
            if (
                saved_next_save_step <= saved_step
                or saved_next_save_step % args.save_every_steps != 0
            ):
                raise ValueError("resume next checkpoint boundary is invalid")
        elif saved_next_save_step != 0:
            raise ValueError("disabled periodic saving has a nonzero next boundary")
        if saved_step >= args.total_timesteps:
            raise ValueError("resume total_timesteps must exceed the saved step")

        saved_validation_records = trainer_state.get("validation_records")
        if not isinstance(saved_validation_records, list) or not all(
            isinstance(record, dict) for record in saved_validation_records
        ):
            raise ValueError("resume validation records must be a list of mappings")
        saved_best_validation_score = trainer_state.get("best_validation_score")
        if saved_best_validation_score is not None:
            if not isinstance(saved_best_validation_score, (list, tuple)) or len(
                saved_best_validation_score
            ) != 4:
                raise ValueError("resume best validation score is invalid")
            if any(
                isinstance(value, bool)
                or not isinstance(
                    value, (int, float, np.integer, np.floating)
                )
                or not np.isfinite(value)
                for value in saved_best_validation_score
            ):
                raise ValueError("resume best validation score is invalid")
            saved_best_validation_score = tuple(
                float(value) for value in saved_best_validation_score
            )
        saved_best_validation_record = trainer_state.get("best_validation_record")
        if saved_best_validation_record is not None and not isinstance(
            saved_best_validation_record, dict
        ):
            raise ValueError("resume best validation record must be a mapping")
        if (saved_best_validation_score is None) != (
            saved_best_validation_record is None
        ):
            raise ValueError("resume best validation score/record disagree")

        reward_rms = trainer_state.get("reward_rms")
        if not isinstance(reward_rms, dict):
            raise ValueError("resume reward RMS state must be a mapping")
        reward_rms_mean = saved_finite_number(reward_rms, "mean", "reward RMS")
        reward_rms_var = saved_finite_number(reward_rms, "var", "reward RMS")
        reward_rms_count = saved_finite_number(reward_rms, "count", "reward RMS")
        if reward_rms_var < 0.0 or reward_rms_count <= 0.0:
            raise ValueError("resume reward RMS variance/count are invalid")

        rollout_log_accumulator = trainer_state.get("rollout_log_accumulator")
        if not isinstance(rollout_log_accumulator, dict):
            raise ValueError("resume rollout log accumulator must be a mapping")
        saved_ep_rewards = rollout_log_accumulator.get("episode_rewards")
        saved_ep_lengths = rollout_log_accumulator.get("episode_lengths")
        saved_ep_stats = rollout_log_accumulator.get("episode_stats")
        if (
            not isinstance(saved_ep_rewards, list)
            or not isinstance(saved_ep_lengths, list)
            or not isinstance(saved_ep_stats, list)
        ):
            raise ValueError("resume rollout log accumulator fields must be lists")
        restored_ep_rewards = []
        for value in saved_ep_rewards:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value, (int, float, np.integer, np.floating)
                )
                or not np.isfinite(value)
            ):
                raise ValueError("resume episode rewards must be finite numbers")
            restored_ep_rewards.append(float(value))
        restored_ep_lengths = []
        for value in saved_ep_lengths:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("resume episode lengths must be integers")
            if int(value) <= 0:
                raise ValueError("resume episode lengths must be positive")
            restored_ep_lengths.append(int(value))
        if len(restored_ep_rewards) != len(restored_ep_lengths):
            raise ValueError("resume episode reward/length counts disagree")
        if len(restored_ep_rewards) > min(
            max(0, args.log_every), saved_num_episodes
        ):
            raise ValueError("resume rollout log accumulator exceeds its boundary")
        if args.env_type == "smaclite":
            if len(saved_ep_stats) != len(restored_ep_rewards) or not all(
                isinstance(record, dict) for record in saved_ep_stats
            ):
                raise ValueError("resume SMAClite episode stats are inconsistent")
        elif saved_ep_stats:
            raise ValueError("resume non-SMAClite checkpoint contains episode stats")

        saved_metrics_size = saved_nonnegative_integer(
            trainer_state, "metrics_file_size_bytes", "trainer state"
        )

        constraint_state = resume_payload.get("switch_constraint_state")
        restored_constraint_state = None
        if args.avoidable_switch_constraint_enabled:
            constraint_schema_version = (
                constraint_state.get("schema_version")
                if isinstance(constraint_state, dict)
                else None
            )
            if (
                not isinstance(constraint_state, dict)
                or isinstance(constraint_schema_version, bool)
                or not isinstance(constraint_schema_version, (int, np.integer))
                or int(constraint_schema_version) != 1
            ):
                raise ValueError("resume checkpoint has no switch-constraint state")
            restored_dual_multiplier = saved_finite_number(
                constraint_state, "multiplier", "switch constraint"
            )
            if not 0.0 <= restored_dual_multiplier <= args.avoidable_switch_dual_max:
                raise ValueError("resume dual multiplier is outside its projection")
            restored_dual_update_count = saved_nonnegative_integer(
                constraint_state, "update_count", "switch constraint"
            )
            restored_dual_skipped_count = saved_nonnegative_integer(
                constraint_state,
                "skipped_zero_opportunity_count",
                "switch constraint",
            )
            restored_dual_cumulative_cost = saved_nonnegative_integer(
                constraint_state, "cumulative_cost", "switch constraint"
            )
            restored_dual_cumulative_opportunity = saved_nonnegative_integer(
                constraint_state, "cumulative_opportunity", "switch constraint"
            )
            restored_dual_last_cost = saved_nonnegative_integer(
                constraint_state, "last_rollout_cost", "switch constraint"
            )
            restored_dual_last_opportunity = saved_nonnegative_integer(
                constraint_state, "last_rollout_opportunity", "switch constraint"
            )
            if (
                restored_dual_update_count + restored_dual_skipped_count
                != saved_update_round
            ):
                raise ValueError(
                    "resume dual update/skip counts do not match rollout updates"
                )
            if not (
                0
                <= restored_dual_cumulative_cost
                <= restored_dual_cumulative_opportunity
            ):
                raise ValueError("resume switch-constraint totals are invalid")
            if (restored_dual_update_count == 0) != (
                restored_dual_cumulative_opportunity == 0
            ):
                raise ValueError(
                    "resume dual updates disagree with cumulative opportunities"
                )
            if not (
                0 <= restored_dual_last_cost <= restored_dual_last_opportunity
                and restored_dual_last_cost <= restored_dual_cumulative_cost
                and restored_dual_last_opportunity
                <= restored_dual_cumulative_opportunity
                and restored_dual_update_count
                <= restored_dual_cumulative_opportunity
            ):
                raise ValueError("resume switch-constraint rollout counts are invalid")
            restored_dual_last_rate = constraint_state.get("last_rollout_rate")
            restored_dual_last_violation = constraint_state.get(
                "last_rollout_violation"
            )
            if restored_dual_last_opportunity == 0:
                if (
                    restored_dual_last_cost != 0
                    or restored_dual_last_rate is not None
                    or restored_dual_last_violation is not None
                ):
                    raise ValueError(
                        "resume zero-opportunity rollout has observed dual statistics"
                    )
            else:
                restored_dual_last_rate = saved_finite_number(
                    constraint_state, "last_rollout_rate", "switch constraint"
                )
                restored_dual_last_violation = saved_finite_number(
                    constraint_state, "last_rollout_violation", "switch constraint"
                )
                expected_last_rate = (
                    restored_dual_last_cost / restored_dual_last_opportunity
                )
                if not np.isclose(
                    restored_dual_last_rate,
                    expected_last_rate,
                    rtol=0.0,
                    atol=1e-12,
                ) or not np.isclose(
                    restored_dual_last_violation,
                    expected_last_rate - args.avoidable_switch_budget,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError("resume switch-constraint rates are inconsistent")
            restored_constraint_state = (
                restored_dual_multiplier,
                restored_dual_update_count,
                restored_dual_skipped_count,
                restored_dual_cumulative_cost,
                restored_dual_cumulative_opportunity,
                restored_dual_last_cost,
                restored_dual_last_opportunity,
                restored_dual_last_rate,
                restored_dual_last_violation,
            )
        elif constraint_state is not None:
            raise ValueError("disabled constraint checkpoint contains dual state")

        saved_python_state = trainer_state.get("python_random_state")
        saved_numpy_state = trainer_state.get("numpy_random_state")
        saved_torch_state = trainer_state.get("torch_random_state")
        try:
            random.Random().setstate(saved_python_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("resume Python RNG state is invalid") from exc
        try:
            np.random.RandomState().set_state(saved_numpy_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("resume NumPy RNG state is invalid") from exc
        if (
            not isinstance(saved_torch_state, torch.Tensor)
            or saved_torch_state.dtype != torch.uint8
            or saved_torch_state.ndim != 1
        ):
            raise ValueError("resume CPU torch RNG state is invalid")
        saved_torch_state = saved_torch_state.cpu()
        try:
            torch.Generator(device="cpu").set_state(saved_torch_state)
        except RuntimeError as exc:
            raise ValueError("resume CPU torch RNG state is invalid") from exc
        saved_cuda_state = trainer_state.get("torch_cuda_random_state_all")
        if device.type == "cuda":
            if (
                not torch.cuda.is_available()
                or not isinstance(saved_cuda_state, (list, tuple))
                or len(saved_cuda_state) != torch.cuda.device_count()
            ):
                raise ValueError("resume checkpoint has invalid CUDA RNG state")
            if not all(
                isinstance(state, torch.Tensor)
                and state.dtype == torch.uint8
                and state.ndim == 1
                for state in saved_cuda_state
            ):
                raise ValueError("resume checkpoint has invalid CUDA RNG state")
            saved_cuda_state = [state.cpu() for state in saved_cuda_state]
        elif saved_cuda_state is not None:
            raise ValueError("non-CUDA resume checkpoint contains CUDA RNG state")

        actor.load_state_dict(resume_payload["actor"])
        critic.load_state_dict(resume_payload["critic"])
        actor_optimizer.load_state_dict(resume_payload["actor_optimizer"])
        critic_optimizer.load_state_dict(resume_payload["critic_optimizer"])
        step = saved_step
        training_step = saved_training_step
        update_round = saved_update_round
        num_episodes = saved_num_episodes
        next_save_step = saved_next_save_step
        best_validation_score = saved_best_validation_score
        best_validation_record = saved_best_validation_record
        validation_records = list(saved_validation_records)
        rb.reward_rms.mean = reward_rms_mean
        rb.reward_rms.var = reward_rms_var
        rb.reward_rms.count = reward_rms_count
        ep_rewards = restored_ep_rewards
        ep_lengths = restored_ep_lengths
        ep_stats = list(saved_ep_stats)
        if restored_constraint_state is not None:
            (
                dual_multiplier,
                dual_update_count,
                dual_skipped_zero_opportunity_count,
                dual_cumulative_cost,
                dual_cumulative_opportunity,
                dual_last_cost,
                dual_last_opportunity,
                dual_last_rate,
                dual_last_violation,
            ) = restored_constraint_state
        random.setstate(saved_python_state)
        np.random.set_state(saved_numpy_state)
        torch.set_rng_state(saved_torch_state)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved_cuda_state)
        restore_jsonl_checkpoint_boundary(metrics_path, saved_metrics_size)
        resume_record = {
            "record_type": "resume_event",
            "resumed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_checkpoint": resume_checkpoint,
            "source_checkpoint_sha256": checkpoint_sha256(resume_checkpoint),
            "environment_steps": step,
            "optimizer_updates": training_step,
            "update_round": update_round,
            "num_episodes": num_episodes,
            "target_total_timesteps": int(args.total_timesteps),
        }
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(resume_record, ensure_ascii=False) + "\n")
    while step < args.total_timesteps:
        num_episode = 0
        while num_episode < args.batch_size:
            episode = {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "states": [],
                "next_states": [],
                "terminated": [],
                "truncated": [],
                "policy_active": [],
                "avoidable_switch_cost": [],
                "switch_opportunity": [],
                "avail_actions": [],
            }
            workload_seed = None
            if args.env_type in {"leo", "leo_multi"}:
                workload_seed = args.train_seed_start + (
                    num_episodes + num_episode
                ) % max(1, args.train_seed_count)
            obs, _ = reset_with_seed(env, workload_seed)
            ep_reward, ep_length = 0, 0
            done, truncated = False, False
            while not done and not truncated:
                avail_action = env.get_avail_actions()
                active_getter = getattr(env, "get_policy_active_mask", None)
                if active_getter is None:
                    policy_active = np.ones(env.n_agents, dtype=np.float32)
                else:
                    policy_active = active_getter()
                state = env.get_state()
                with torch.no_grad():
                    actions, log_probs = actor.act(
                        torch.from_numpy(obs).float().to(device),
                        avail_action=torch.from_numpy(avail_action).bool().to(device),
                    )
                next_obs, reward, done, truncated, infos = env.step(
                    actions.cpu().numpy()
                )
                decision_cost = np.zeros(env.n_agents, dtype=np.float32)
                decision_opportunity = np.zeros(env.n_agents, dtype=np.float32)
                if args.env_type == "leo_multi":
                    decision_cost = env.get_last_avoidable_switch_costs().astype(
                        np.float32
                    )
                    decision_opportunity = env.get_last_switch_opportunities().astype(
                        np.float32
                    )
                    forced_switch = env.get_last_forced_switches()
                    if np.any(decision_cost > decision_opportunity):
                        raise RuntimeError("avoidable-switch cost exceeds opportunity")
                    if np.any(forced_switch & decision_cost.astype(bool)):
                        raise RuntimeError("forced reroute entered the avoidable-switch cost")
                    if args.avoidable_switch_constraint_enabled:
                        decision_candidates = torch.from_numpy(obs).float().reshape(
                            env.n_agents,
                            env.get_action_size(),
                            candidate_feature_dim,
                        )
                        expected_cost, expected_opportunity = (
                            avoidable_switch_decisions(
                                decision_candidates,
                                torch.from_numpy(avail_action).bool(),
                                actions.cpu().long(),
                                int(
                                    candidate_actor_spec[
                                        "route_switch_feature_index"
                                    ]
                                ),
                            )
                        )
                        if not np.array_equal(
                            decision_cost.astype(bool), expected_cost.numpy()
                        ) or not np.array_equal(
                            decision_opportunity.astype(bool),
                            expected_opportunity.numpy(),
                        ):
                            raise RuntimeError(
                                "environment and actor switch-constraint ledgers disagree"
                            )
                agent_reward_getter = getattr(
                    env, "get_last_agent_rewards", None
                )
                if agent_reward_getter is None:
                    agent_reward = np.full(
                        env.n_agents, float(reward), dtype=np.float32
                    )
                else:
                    agent_reward = agent_reward_getter()
                next_state = env.get_state()
                ep_reward += reward
                ep_length += 1
                step += 1
                episode["obs"].append(obs)
                episode["actions"].append(actions.cpu())
                episode["log_prob"].append(log_probs.cpu())
                episode["reward"].append(agent_reward)
                episode["next_states"].append(next_state)
                episode["terminated"].append(done)
                episode["truncated"].append(truncated)
                episode["policy_active"].append(policy_active)
                episode["avoidable_switch_cost"].append(decision_cost)
                episode["switch_opportunity"].append(decision_opportunity)
                episode["avail_actions"].append(avail_action)
                episode["states"].append(state)

                obs = next_obs

            rb.add(episode)
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            if args.env_type == "smaclite":
                ep_stats.append(infos)
            num_episode += 1
        num_episodes += args.batch_size
        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episodes, step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "rollout/battle_won",
                    np.mean([info["battle_won"] for info in ep_stats]),
                    step,
                )
            ep_rewards = []
            ep_lengths = []
            ep_stats = []
        ## Collate episodes in buffer into single batch
        (
            b_obs,
            b_actions,
            b_log_probs,
            b_reward,
            b_states,
            b_next_states,
            b_avail_actions,
            b_terminated,
            b_truncated,
            b_policy_active,
            b_avoidable_switch_cost,
            b_switch_opportunity,
            b_mask,
        ) = rb.get_batch()

        # Learning-rate schedule: hold base LR for the first half of training,
        # then linearly decay to 12.5% over the second half (reduces the late-
        # training validation degradation seen when the policy overfits the
        # cycled traffic seeds). Toggleable via args.lr_decay.
        if args.lr_decay:
            progress = step / max(1, args.total_timesteps)
            if progress <= 0.5:
                lr_frac = 1.0
            else:
                lr_frac = max(0.125, 1.0 - (progress - 0.5) * 1.75)
            for _opt, _base in (
                (actor_optimizer, args.learning_rate_actor),
                (critic_optimizer, args.learning_rate_critic),
            ):
                for _pg in _opt.param_groups:
                    _pg["lr"] = _base * lr_frac

        # GAE with explicit terminated/truncated semantics. Terminated states
        # zero-bootstrap; time-limit truncations bootstrap from next_state.
        with torch.no_grad():
            values = critic(b_states).squeeze(-1).unsqueeze(-1).expand(
                -1, -1, env.n_agents
            )
            next_values = critic(b_next_states).squeeze(-1).unsqueeze(-1).expand(
                -1, -1, env.n_agents
            )
            agent_rewards = b_reward
            terminated_agents = b_terminated.unsqueeze(-1).expand_as(values)
            truncated_agents = b_truncated.unsqueeze(-1).expand_as(values)
            advantages, return_lambda = compute_gae(
                rewards=agent_rewards,
                values=values,
                next_values=next_values,
                terminated=terminated_agents,
                truncated=truncated_agents,
                valid=b_mask,
                gamma=args.gamma,
                gae_lambda=args.td_lambda,
            )

        valid_agent_mask = (
            b_mask.unsqueeze(-1).expand_as(advantages) & b_policy_active
        )
        if args.normalize_advantage:
            advantages = masked_standardize(
                advantages,
                valid_agent_mask,
                epsilon=args.normalization_epsilon,
            )
        ret_mu = return_lambda[valid_agent_mask].mean()
        ret_std = return_lambda[valid_agent_mask].std(unbiased=False)
        if args.normalize_return:
            writer.add_scalar("train/return_normalization_mean", ret_mu.item(), step)
            writer.add_scalar("train/return_normalization_std", ret_std.item(), step)
        rollout_eligible_count = 0
        rollout_active_agent_count = int(b_policy_active.sum().item())
        rollout_constraint_cost = 0
        rollout_constraint_opportunity = 0
        rollout_constraint_rate = None
        rollout_constraint_violation = None
        dual_multiplier_used = float(dual_multiplier)
        if args.avoidable_switch_reduction == "rollout_micro_mean":
            with torch.no_grad():
                rollout_candidates = b_obs.reshape(
                    *b_obs.shape[:-1],
                    env.get_action_size(),
                    candidate_feature_dim,
                )
                _, rollout_switch_eligible = avoidable_switch_probabilities(
                    torch.zeros_like(b_avail_actions, dtype=b_obs.dtype),
                    rollout_candidates,
                    b_avail_actions,
                    int(candidate_actor_spec["route_switch_feature_index"]),
                )
                rollout_eligible_count = int(
                    (b_policy_active & rollout_switch_eligible).sum().item()
                )
            if rollout_active_agent_count <= 0:
                raise RuntimeError("rollout contains no active policy decisions")
        if args.avoidable_switch_constraint_enabled:
            decision_valid = b_mask.unsqueeze(-1) & b_policy_active
            realized_opportunity = decision_valid & b_switch_opportunity
            realized_cost = decision_valid & b_avoidable_switch_cost
            if bool((realized_cost & ~realized_opportunity).any()):
                raise RuntimeError("rollout switch cost exceeds its opportunity mask")
            expected_opportunity = decision_valid & rollout_switch_eligible
            if not torch.equal(realized_opportunity, expected_opportunity):
                raise RuntimeError(
                    "rollout switch-opportunity ledger disagrees with actor features"
                )
            rollout_constraint_cost = int(realized_cost.sum().item())
            rollout_constraint_opportunity = int(realized_opportunity.sum().item())
            if rollout_constraint_opportunity != rollout_eligible_count:
                raise RuntimeError("rollout switch-opportunity denominator mismatch")
            if rollout_constraint_opportunity > 0:
                rollout_constraint_rate = (
                    rollout_constraint_cost / rollout_constraint_opportunity
                )
                rollout_constraint_violation = (
                    rollout_constraint_rate - args.avoidable_switch_budget
                )
        # Shuffle valid time transitions into PPO minibatches. Actor losses use
        # active satellite-agent samples; the centralized critic uses one team
        # return target per graph state.
        actor_losses = []
        critic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        actor_gradients_post_clip = []
        critic_gradients_post_clip = []
        clipped_ratios = []
        actor_primary_losses = []
        avoidable_switch_probability_means = []
        avoidable_switch_regularization_penalties = []
        avoidable_switch_regularization_positive_fractions = []
        avoidable_switch_eligible_fractions = []
        actor_primary_gradient_norms = []
        avoidable_switch_regularization_gradient_norms = []
        residual_parameter_names = {
            4: ("shared_candidate_actor.route_hysteresis_residual_bias",),
            5: (
                "shared_candidate_actor.route_hysteresis_residual_calm_bias",
                "shared_candidate_actor.route_hysteresis_residual_urgent_bias",
            ),
        }.get(int((candidate_actor_spec or {}).get("schema_version", 0)), ())
        residual_primary_signed_gradients = {
            name: [] for name in residual_parameter_names
        }
        residual_regularizer_signed_gradients = {
            name: [] for name in residual_parameter_names
        }
        stop_for_kl = False
        for epoch_idx in range(args.epochs):
            epoch_kls = []
            transition_minibatches = shuffled_transition_minibatches(
                b_mask, args.num_minibatches
            )
            active_transition_minibatches = [
                indices
                for indices in transition_minibatches
                if b_policy_active[indices[:, 0], indices[:, 1]].any()
            ]
            active_minibatch_count = len(active_transition_minibatches)
            for transition_indices in active_transition_minibatches:
                batch_indices = transition_indices[:, 0]
                time_indices = transition_indices[:, 1]
                active_mask = b_policy_active[batch_indices, time_indices]

                current_logits = actor.logits(
                    x=b_obs[batch_indices, time_indices],
                    avail_action=b_avail_actions[batch_indices, time_indices],
                )
                current_dist = Categorical(logits=current_logits)
                current_logprob = current_dist.log_prob(
                    b_actions[batch_indices, time_indices]
                )
                log_ratio = current_logprob - b_log_probs[
                    batch_indices, time_indices
                ]
                ratio = torch.exp(log_ratio)
                minibatch_advantages = advantages[
                    batch_indices, time_indices
                ]
                pg_loss_unclipped = minibatch_advantages * ratio
                pg_loss_clipped = minibatch_advantages * torch.clamp(
                    ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
                )
                pg_loss = torch.min(
                    pg_loss_unclipped, pg_loss_clipped
                )[active_mask].mean()
                normalized_entropy = feasible_normalized_entropy(
                    current_dist,
                    b_avail_actions[batch_indices, time_indices],
                )
                entropy = normalized_entropy[active_mask].mean()
                switch_probability_mean = current_logits.sum() * 0.0
                switch_regularization_penalty = current_logits.sum() * 0.0
                switch_regularization_positive_fraction = current_logits.new_tensor(
                    0.0
                )
                eligible_fraction = current_logits.new_tensor(0.0)
                if (
                    args.avoidable_switch_probability_coef > 0.0
                    or args.avoidable_switch_constraint_enabled
                ):
                    minibatch_obs = b_obs[batch_indices, time_indices]
                    candidates = minibatch_obs.reshape(
                        *minibatch_obs.shape[:-1],
                        env.get_action_size(),
                        candidate_feature_dim,
                    )
                    switch_probability, switch_eligible = (
                        avoidable_switch_probabilities(
                            current_logits,
                            candidates,
                            b_avail_actions[batch_indices, time_indices],
                            int(candidate_actor_spec["route_switch_feature_index"]),
                        )
                    )
                    regularization_mask = active_mask & switch_eligible
                    if regularization_mask.any():
                        selected_switch_probability = switch_probability[
                            regularization_mask
                        ]
                        if args.avoidable_switch_reduction == "rollout_micro_mean":
                            switch_probability_mean = rollout_micro_minibatch_mean(
                                selected_switch_probability,
                                rollout_denominator=rollout_eligible_count,
                                active_minibatch_count=active_minibatch_count,
                            )
                        else:
                            switch_probability_mean = (
                                selected_switch_probability.mean()
                            )
                        if (
                            args.avoidable_switch_regularization_mode
                            == "conditional_probability"
                        ):
                            regularization_values = switch_probability[
                                regularization_mask
                            ]
                        elif (
                            args.avoidable_switch_regularization_mode
                            == "greedy_logit_margin"
                        ):
                            margin_penalties, margin_eligible = (
                                avoidable_switch_logit_margin_penalties(
                                    current_logits,
                                    candidates,
                                    b_avail_actions[
                                        batch_indices, time_indices
                                    ],
                                    int(
                                        switch_regularizer_spec[
                                            "route_switch_feature_index"
                                        ]
                                    ),
                                    route_urgency_feature_index=int(
                                        switch_regularizer_spec[
                                            "route_urgency_feature_index"
                                        ]
                                    ),
                                    route_class_2_feature_index=int(
                                        switch_regularizer_spec[
                                            "route_class_2_feature_index"
                                        ]
                                    ),
                                    route_hysteresis_beta=float(
                                        switch_regularizer_spec[
                                            "route_hysteresis_beta"
                                        ]
                                    ),
                                    route_hysteresis_urgency_relief=float(
                                        switch_regularizer_spec[
                                            "route_hysteresis_urgency_relief"
                                        ]
                                    ),
                                    route_hysteresis_class_2_relief=float(
                                        switch_regularizer_spec[
                                            "route_hysteresis_class_2_relief"
                                        ]
                                    ),
                                    margin=float(
                                        switch_regularizer_spec["logit_margin"]
                                    ),
                                )
                            )
                            if not torch.equal(margin_eligible, switch_eligible):
                                raise RuntimeError(
                                    "switch regularizer eligibility mismatch"
                                )
                            regularization_values = margin_penalties[
                                regularization_mask
                            ]
                        else:
                            isolated_logits = (
                                actor.shared_candidate_actor.isolated_regularizer_logits(
                                    candidates,
                                    b_avail_actions[
                                        batch_indices, time_indices
                                    ],
                                )
                            )
                            margin_penalties, margin_eligible = (
                                avoidable_switch_actual_logit_margin_penalties(
                                    isolated_logits,
                                    candidates,
                                    b_avail_actions[
                                        batch_indices, time_indices
                                    ],
                                    int(
                                        switch_regularizer_spec[
                                            "route_switch_feature_index"
                                        ]
                                    ),
                                    margin=float(
                                        switch_regularizer_spec["logit_margin"]
                                    ),
                                )
                            )
                            if not torch.equal(margin_eligible, switch_eligible):
                                raise RuntimeError(
                                    "switch regularizer eligibility mismatch"
                                )
                            regularization_values = margin_penalties[
                                regularization_mask
                            ]
                        if args.avoidable_switch_reduction == "rollout_micro_mean":
                            switch_regularization_penalty = (
                                rollout_micro_minibatch_mean(
                                    regularization_values,
                                    rollout_denominator=rollout_eligible_count,
                                    active_minibatch_count=active_minibatch_count,
                                )
                            )
                            switch_regularization_positive_fraction = (
                                rollout_micro_minibatch_mean(
                                    (regularization_values > 0.0).to(
                                        current_logits.dtype
                                    ),
                                    rollout_denominator=rollout_eligible_count,
                                    active_minibatch_count=active_minibatch_count,
                                )
                            )
                        else:
                            switch_regularization_penalty = (
                                regularization_values.mean()
                            )
                            switch_regularization_positive_fraction = (
                                (regularization_values > 0.0)
                                .to(current_logits.dtype)
                                .mean()
                            )
                    if args.avoidable_switch_reduction == "rollout_micro_mean":
                        eligible_fraction = rollout_micro_minibatch_mean(
                            regularization_mask.to(current_logits.dtype),
                            rollout_denominator=rollout_active_agent_count,
                            active_minibatch_count=active_minibatch_count,
                        )
                    else:
                        eligible_fraction = regularization_mask.float().sum() / (
                            active_mask.float().sum().clamp_min(1.0)
                        )
                actor_primary_loss = -pg_loss - args.entropy_coef * entropy
                switch_loss_weight = (
                    dual_multiplier_used
                    if args.avoidable_switch_constraint_enabled
                    else args.avoidable_switch_probability_coef
                )
                actor_loss = (
                    actor_primary_loss
                    + switch_loss_weight * switch_regularization_penalty
                )
                if args.log_loss_component_gradients:
                    actor_parameters = tuple(actor.parameters())
                    primary_gradients = torch.autograd.grad(
                        actor_primary_loss,
                        actor_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    regularizer_gradients = torch.autograd.grad(
                        switch_loss_weight * switch_regularization_penalty,
                        actor_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    actor_primary_gradient_norms.append(
                        norm_d(primary_gradients, 2).item()
                    )
                    avoidable_switch_regularization_gradient_norms.append(
                        norm_d(regularizer_gradients, 2).item()
                    )
                    for (name, _), primary_gradient, regularizer_gradient in zip(
                        actor.named_parameters(),
                        primary_gradients,
                        regularizer_gradients,
                    ):
                        if name not in residual_primary_signed_gradients:
                            continue
                        residual_primary_signed_gradients[name].append(
                            0.0
                            if primary_gradient is None
                            else float(primary_gradient.detach().reshape(-1)[0].item())
                        )
                        residual_regularizer_signed_gradients[name].append(
                            0.0
                            if regularizer_gradient is None
                            else float(
                                regularizer_gradient.detach().reshape(-1)[0].item()
                            )
                        )

                value_prediction = critic(
                    b_states[batch_indices, time_indices]
                ).squeeze(-1)
                value_target = return_lambda[
                    batch_indices, time_indices
                ].mean(dim=-1)
                if args.normalize_return:
                    scale = ret_std + args.normalization_epsilon
                    value_prediction = (value_prediction - ret_mu) / scale
                    value_target = (value_target - ret_mu) / scale
                # Huber (smooth_l1) instead of MSE: caps the per-sample gradient
                # contribution from outlier targets (e.g. a slot with many drops),
                # pairing with normalize_return for a robust value fit.
                critic_loss = F.smooth_l1_loss(value_prediction, value_target)

                kl_divergence = (
                    ((ratio - 1) - log_ratio)[active_mask].mean()
                )
                clipped_ratio = (
                    ((ratio - 1.0).abs() > args.ppo_clip)[active_mask]
                    .float()
                    .mean()
                )

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                actor_gradient = norm_d(
                    [p.grad for p in actor.parameters()], 2
                )
                critic_gradient = norm_d(
                    [p.grad for p in critic.parameters()], 2
                )
                if args.clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=args.clip_gradients
                    )
                if args.critic_clip_gradients > 0:
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), max_norm=args.critic_clip_gradients
                    )
                actor_gradient_post = norm_d(
                    [p.grad for p in actor.parameters()], 2
                )
                critic_gradient_post = norm_d(
                    [p.grad for p in critic.parameters()], 2
                )
                actor_optimizer.step()
                if (
                    candidate_actor_spec
                    and candidate_actor_spec.get("schema_version") in {4, 5}
                ):
                    actor.shared_candidate_actor.project_route_hysteresis_residual_()
                critic_optimizer.step()
                training_step += 1

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies_bonuses.append(entropy.item())
                kl_divergences.append(kl_divergence.item())
                epoch_kls.append(kl_divergence.item())
                actor_gradients.append(actor_gradient.item())
                critic_gradients.append(critic_gradient.item())
                actor_gradients_post_clip.append(actor_gradient_post.item())
                critic_gradients_post_clip.append(critic_gradient_post.item())
                clipped_ratios.append(clipped_ratio.item())
                actor_primary_losses.append(actor_primary_loss.item())
                avoidable_switch_probability_means.append(
                    switch_probability_mean.item()
                )
                avoidable_switch_regularization_penalties.append(
                    switch_regularization_penalty.item()
                )
                avoidable_switch_regularization_positive_fractions.append(
                    switch_regularization_positive_fraction.item()
                )
                avoidable_switch_eligible_fractions.append(
                    eligible_fraction.item()
                )

            if (
                args.target_kl > 0
                and epoch_kls
                and float(np.mean(epoch_kls)) > args.target_kl
            ):
                writer.add_scalar("train/early_stop_epoch", epoch_idx + 1, step)
                stop_for_kl = True
            if stop_for_kl:
                break
        update_round += 1
        dual_update_skipped = False
        dual_projection_hit = False
        if args.avoidable_switch_constraint_enabled:
            dual_cumulative_cost += rollout_constraint_cost
            dual_cumulative_opportunity += rollout_constraint_opportunity
            dual_last_cost = rollout_constraint_cost
            dual_last_opportunity = rollout_constraint_opportunity
            dual_transition = advance_avoidable_switch_dual(
                multiplier=dual_multiplier_used,
                update_count=dual_update_count,
                skipped_zero_opportunity_count=(
                    dual_skipped_zero_opportunity_count
                ),
                cost_count=rollout_constraint_cost,
                opportunity_count=rollout_constraint_opportunity,
                budget=args.avoidable_switch_budget,
                learning_rate=args.avoidable_switch_dual_learning_rate,
                maximum=args.avoidable_switch_dual_max,
            )
            dual_multiplier = dual_transition["multiplier"]
            dual_update_count = dual_transition["update_count"]
            dual_skipped_zero_opportunity_count = dual_transition[
                "skipped_zero_opportunity_count"
            ]
            dual_last_rate = dual_transition["empirical_rate"]
            dual_last_violation = dual_transition["violation"]
            dual_update_skipped = dual_transition["update_skipped"]
            dual_projection_hit = dual_transition["projection_hit"]
            if dual_last_rate != rollout_constraint_rate or (
                dual_last_violation != rollout_constraint_violation
            ):
                raise RuntimeError("rollout and dual constraint statistics disagree")

        with torch.no_grad():
            valid_states = b_states[b_mask]
            predicted_values = critic(valid_states).squeeze(-1)
            target_values = return_lambda[b_mask].mean(dim=-1)
            target_var = torch.var(target_values, unbiased=False)
            explained_variance = 1.0 - torch.var(
                target_values - predicted_values, unbiased=False
            ) / (target_var + args.normalization_epsilon)

        valid_actions = b_actions[valid_agent_mask]
        slot_counts = torch.bincount(
            valid_actions.reshape(-1), minlength=env.get_action_size()
        ).float()
        slot_freq = slot_counts / slot_counts.sum().clamp_min(1.0)

        writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar(
            "train/actor_primary_loss", np.mean(actor_primary_losses), step
        )
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
        writer.add_scalar(
            "train/avoidable_switch_probability",
            np.mean(avoidable_switch_probability_means),
            step,
        )
        writer.add_scalar(
            "train/avoidable_switch_regularization_penalty",
            np.mean(avoidable_switch_regularization_penalties),
            step,
        )
        writer.add_scalar(
            "train/avoidable_switch_regularization_loss",
            args.avoidable_switch_probability_coef
            * np.mean(avoidable_switch_regularization_penalties),
            step,
        )
        writer.add_scalar(
            "train/avoidable_switch_regularization_positive_fraction",
            np.mean(avoidable_switch_regularization_positive_fractions),
            step,
        )
        writer.add_scalar(
            "train/avoidable_switch_eligible_fraction",
            np.mean(avoidable_switch_eligible_fractions),
            step,
        )
        constraint_surrogate_mean = float(
            np.mean(avoidable_switch_regularization_penalties)
        )
        if args.avoidable_switch_constraint_enabled:
            writer.add_scalar(
                "constraint/surrogate_rate",
                constraint_surrogate_mean,
                step,
            )
            writer.add_scalar(
                "constraint/rollout_cost", rollout_constraint_cost, step
            )
            writer.add_scalar(
                "constraint/rollout_opportunities",
                rollout_constraint_opportunity,
                step,
            )
            if rollout_constraint_rate is not None:
                writer.add_scalar(
                    "constraint/empirical_rate", rollout_constraint_rate, step
                )
                writer.add_scalar(
                    "constraint/empirical_violation",
                    rollout_constraint_violation,
                    step,
                )
            writer.add_scalar(
                "constraint/dual_multiplier_used", dual_multiplier_used, step
            )
            writer.add_scalar(
                "constraint/dual_multiplier_next", dual_multiplier, step
            )
            writer.add_scalar(
                "constraint/lagrangian_term",
                dual_multiplier_used
                * (constraint_surrogate_mean - args.avoidable_switch_budget),
                step,
            )
        if args.log_loss_component_gradients:
            writer.add_scalar(
                "train/actor_primary_gradient_norm",
                np.mean(actor_primary_gradient_norms),
                step,
            )
            writer.add_scalar(
                "train/avoidable_switch_regularization_gradient_norm",
                np.mean(avoidable_switch_regularization_gradient_norms),
                step,
            )
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients), step)
        writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
        writer.add_scalar(
            "train/actor_gradients_post_clip",
            np.mean(actor_gradients_post_clip),
            step,
        )
        writer.add_scalar(
            "train/critic_gradients_post_clip",
            np.mean(critic_gradients_post_clip),
            step,
        )
        writer.add_scalar("train/explained_variance", explained_variance.item(), step)
        writer.add_scalar(
            "train/advantage_mean", advantages[valid_agent_mask].mean().item(), step
        )
        writer.add_scalar(
            "train/advantage_std",
            advantages[valid_agent_mask].std(unbiased=False).item(),
            step,
        )
        writer.add_scalar("train/return_mean", target_values.mean().item(), step)
        writer.add_scalar(
            "train/return_std", target_values.std(unbiased=False).item(), step
        )
        for slot, frequency in enumerate(slot_freq):
            writer.add_scalar(
                f"train/action_slot_frequency/{slot}", frequency.item(), step
            )
        writer.add_scalar("train/num_updates", training_step, step)
        learned_state = candidate_actor_learned_state()
        if learned_state is not None:
            for parameter_name, value in learned_state.items():
                writer.add_scalar(f"train/{parameter_name}", value, step)
        training_record = {
            "record_type": "training_update",
            "environment_steps": int(step),
            "update_round": int(update_round),
            "optimizer_updates": int(training_step),
            "actor_loss": float(np.mean(actor_losses)),
            "actor_primary_loss": float(np.mean(actor_primary_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "normalized_entropy": float(np.mean(entropies_bonuses)),
            "approx_kl": float(np.mean(kl_divergences)),
            "clip_fraction": float(np.mean(clipped_ratios)),
            "avoidable_switch_probability": float(
                np.mean(avoidable_switch_probability_means)
            ),
            "avoidable_switch_regularization_penalty": float(
                np.mean(avoidable_switch_regularization_penalties)
            ),
            "avoidable_switch_regularization_loss": float(
                args.avoidable_switch_probability_coef
                * np.mean(avoidable_switch_regularization_penalties)
            ),
            "avoidable_switch_regularization_positive_fraction": float(
                np.mean(avoidable_switch_regularization_positive_fractions)
            ),
            "avoidable_switch_eligible_fraction": float(
                np.mean(avoidable_switch_eligible_fractions)
            ),
            "actor_gradient_pre_clip": float(np.mean(actor_gradients)),
            "critic_gradient_pre_clip": float(np.mean(critic_gradients)),
            "actor_gradient_post_clip": float(np.mean(actor_gradients_post_clip)),
            "critic_gradient_post_clip": float(np.mean(critic_gradients_post_clip)),
            "explained_variance": float(explained_variance.item()),
            "advantage_mean": float(advantages[valid_agent_mask].mean().item()),
            "advantage_std": float(advantages[valid_agent_mask].std(unbiased=False).item()),
            "return_mean": float(target_values.mean().item()),
            "return_std": float(target_values.std(unbiased=False).item()),
            "action_slot_frequency": [float(x) for x in slot_freq.tolist()],
        }
        if args.avoidable_switch_reduction == "rollout_micro_mean":
            training_record.update(
                avoidable_switch_eligibility_stage="pre_contention",
                rollout_active_policy_decisions=rollout_active_agent_count,
                rollout_avoidable_switch_eligible_decisions=(
                    rollout_eligible_count
                ),
            )
        if args.avoidable_switch_constraint_enabled:
            training_record.update(
                switch_constraint_schema_version=1,
                decision_avoidable_switch_cost=rollout_constraint_cost,
                decision_switch_opportunities=rollout_constraint_opportunity,
                decision_avoidable_switch_rate=rollout_constraint_rate,
                decision_avoidable_switch_rate_defined=(
                    rollout_constraint_rate is not None
                ),
                decision_avoidable_switch_violation=rollout_constraint_violation,
                switch_constraint_budget=float(args.avoidable_switch_budget),
                switch_constraint_surrogate_rate=constraint_surrogate_mean,
                switch_constraint_actor_term=(
                    dual_multiplier_used * constraint_surrogate_mean
                ),
                switch_constraint_centered_lagrangian_term=(
                    dual_multiplier_used
                    * (constraint_surrogate_mean - args.avoidable_switch_budget)
                ),
                dual_multiplier_used=dual_multiplier_used,
                dual_multiplier_next=float(dual_multiplier),
                dual_update_count=int(dual_update_count),
                dual_update_skipped_zero_opportunity=dual_update_skipped,
                dual_skipped_zero_opportunity_count=int(
                    dual_skipped_zero_opportunity_count
                ),
                dual_projection_hit=dual_projection_hit,
                dual_cumulative_cost=int(dual_cumulative_cost),
                dual_cumulative_opportunity=int(dual_cumulative_opportunity),
            )
        if learned_state is not None:
            training_record["candidate_actor_learned_state"] = learned_state
            shared_actor = actor.shared_candidate_actor
            cap_hits = {}
            for parameter_name, value in learned_state.items():
                parameter = getattr(shared_actor, parameter_name)
                dtype_cap = route_hysteresis_residual_dtype_cap(
                    args.route_hysteresis_residual_cap,
                    dtype=parameter.dtype,
                )
                cap_hits[parameter_name] = value >= dtype_cap
            training_record["route_hysteresis_residual_cap_hit"] = cap_hits
        if args.log_loss_component_gradients:
            training_record.update(
                actor_primary_gradient_norm=float(
                    np.mean(actor_primary_gradient_norms)
                ),
                avoidable_switch_regularization_gradient_norm=float(
                    np.mean(avoidable_switch_regularization_gradient_norms)
                ),
            )
            if residual_parameter_names:
                prefix = "shared_candidate_actor."
                training_record[
                    "route_hysteresis_residual_primary_signed_gradient"
                ] = {
                    name.removeprefix(prefix): float(np.mean(values))
                    for name, values in residual_primary_signed_gradients.items()
                }
                training_record[
                    "route_hysteresis_residual_regularizer_signed_gradient"
                ] = {
                    name.removeprefix(prefix): float(np.mean(values))
                    for name, values in residual_regularizer_signed_gradients.items()
                }
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(training_record, ensure_ascii=False) + "\n")
        periodic_checkpoint_label = None
        if args.save_every_steps > 0 and step >= next_save_step:
            while next_save_step <= step:
                next_save_step += args.save_every_steps
            periodic_checkpoint_label = f"step_{step}"

        if update_round % max(1, args.eval_steps) == 0:
            first_eval_seed = (
                args.validation_seed_start
                if args.env_type in {"leo", "leo_multi"}
                else None
            )
            eval_obs, _ = reset_with_seed(eval_env, first_eval_seed)
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            current_decision_cost = 0
            current_decision_opportunity = 0
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    actions = actor.greedy(
                        torch.from_numpy(eval_obs).float().to(device),
                        avail_action=torch.from_numpy(eval_env.get_avail_actions())
                        .bool()
                        .to(device),
                    )
                next_obs_, reward, done, truncated, infos = eval_env.step(
                    actions.cpu().numpy()
                )
                if args.env_type == "leo_multi":
                    step_cost = np.asarray(
                        infos["decision_avoidable_switch_costs"], dtype=bool
                    )
                    step_opportunity = np.asarray(
                        infos["decision_switch_opportunities"], dtype=bool
                    )
                    step_forced = np.asarray(
                        infos["decision_forced_switches"], dtype=bool
                    )
                    if np.any(step_cost & ~step_opportunity) or np.any(
                        step_cost & step_forced
                    ):
                        raise RuntimeError(
                            "invalid validation switch-constraint ledger"
                        )
                    current_decision_cost += int(step_cost.sum())
                    current_decision_opportunity += int(step_opportunity.sum())
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    episode_info = dict(infos)
                    episode_info["decision_avoidable_switches"] = int(
                        current_decision_cost
                    )
                    episode_info["decision_switch_opportunities"] = int(
                        current_decision_opportunity
                    )
                    episode_info["decision_avoidable_switch_rate"] = (
                        current_decision_cost / current_decision_opportunity
                        if current_decision_opportunity > 0
                        else None
                    )
                    eval_ep_stats.append(episode_info)
                    current_reward = 0
                    current_ep_length = 0
                    current_decision_cost = 0
                    current_decision_opportunity = 0
                    eval_ep += 1
                    if eval_ep < args.num_eval_ep:
                        next_eval_seed = (
                            args.validation_seed_start + eval_ep
                            if args.env_type in {"leo", "leo_multi"}
                            else None
                        )
                        eval_obs, _ = reset_with_seed(
                            eval_env, next_eval_seed
                        )
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    np.mean([info["battle_won"] for info in eval_ep_stats]),
                    step,
                )
            elif args.env_type in {"leo", "leo_multi"}:
                mean_delivery = float(
                    np.mean(
                        [info.get("delivery_ratio", 0.0) for info in eval_ep_stats]
                    )
                )
                mean_drop = float(
                    np.mean([info.get("drop_rate", 0.0) for info in eval_ep_stats])
                )
                mean_delay = float(
                    np.mean(
                        [
                            info.get("average_delay_slots", 0.0)
                            for info in eval_ep_stats
                        ]
                    )
                )
                writer.add_scalar("eval/delivery_ratio", mean_delivery, step)
                writer.add_scalar("eval/drop_rate", mean_drop, step)
                writer.add_scalar("eval/average_delay_slots", mean_delay, step)
                validation_record = {
                    "record_type": "validation",
                    "environment_steps": int(step),
                    "episodes": int(args.num_eval_ep),
                    "seed_start": int(args.validation_seed_start),
                    "delivery_ratio": mean_delivery,
                    "mean_reward": float(np.mean(eval_ep_reward)),
                    "drop_rate": mean_drop,
                    "average_delay_slots": mean_delay,
                }
                if args.env_type == "leo_multi":
                    routing_switches_total = int(
                        sum(info["routing_switches"] for info in eval_ep_stats)
                    )
                    avoidable_switches_total = int(
                        sum(
                            info["avoidable_routing_switches"]
                            for info in eval_ep_stats
                        )
                    )
                    forced_switches_total = int(
                        sum(
                            info["forced_routing_switches"]
                            for info in eval_ep_stats
                        )
                    )
                    switch_opportunities = int(
                        sum(
                            info["switch_opportunities"]
                            for info in eval_ep_stats
                        )
                    )
                    decision_avoidable_switches = int(
                        sum(
                            info["decision_avoidable_switches"]
                            for info in eval_ep_stats
                        )
                    )
                    decision_switch_opportunities = int(
                        sum(
                            info["decision_switch_opportunities"]
                            for info in eval_ep_stats
                        )
                    )
                    if (
                        args.avoidable_switch_constraint_enabled
                        and decision_switch_opportunities <= 0
                    ):
                        raise RuntimeError(
                            "constraint validation has no switch opportunities"
                        )
                    decision_avoidable_switch_rate = (
                        decision_avoidable_switches
                        / decision_switch_opportunities
                        if decision_switch_opportunities > 0
                        else None
                    )
                    validation_episode_count = len(eval_ep_stats)
                    mean_switches = routing_switches_total / validation_episode_count
                    mean_avoidable_switches = (
                        avoidable_switches_total / validation_episode_count
                    )
                    mean_forced_switches = (
                        forced_switches_total / validation_episode_count
                    )
                    avoidable_switch_rate = avoidable_switches_total / max(
                        1, switch_opportunities
                    )
                    mean_class_2_delivery = float(
                        np.mean(
                            [
                                info["class_2_delivery_ratio"]
                                for info in eval_ep_stats
                            ]
                        )
                    )
                    writer.add_scalar("eval/routing_switches", mean_switches, step)
                    writer.add_scalar(
                        "eval/avoidable_routing_switches",
                        mean_avoidable_switches,
                        step,
                    )
                    writer.add_scalar(
                        "eval/forced_routing_switches",
                        mean_forced_switches,
                        step,
                    )
                    writer.add_scalar(
                        "eval/avoidable_switch_rate",
                        avoidable_switch_rate,
                        step,
                    )
                    if decision_avoidable_switch_rate is not None:
                        writer.add_scalar(
                            "eval/decision_avoidable_switch_rate",
                            decision_avoidable_switch_rate,
                            step,
                        )
                    writer.add_scalar(
                        "eval/class_2_delivery_ratio",
                        mean_class_2_delivery,
                        step,
                    )
                    validation_record.update(
                        {
                            "routing_switches": mean_switches,
                            "routing_switches_total": routing_switches_total,
                            "avoidable_routing_switches": mean_avoidable_switches,
                            "avoidable_routing_switches_total": (
                                avoidable_switches_total
                            ),
                            "forced_routing_switches": mean_forced_switches,
                            "forced_routing_switches_total": forced_switches_total,
                            "switch_opportunities": switch_opportunities,
                            "avoidable_switch_rate": float(
                                avoidable_switch_rate
                            ),
                            "class_2_delivery_ratio": mean_class_2_delivery,
                            "decision_avoidable_switches": (
                                decision_avoidable_switches
                            ),
                            "decision_switch_opportunities": (
                                decision_switch_opportunities
                            ),
                            "decision_avoidable_switch_rate": (
                                decision_avoidable_switch_rate
                            ),
                        }
                    )
                validation_learned_state = candidate_actor_learned_state()
                if validation_learned_state is not None:
                    validation_record["candidate_actor_learned_state"] = (
                        validation_learned_state
                    )
                if deferred_validation_selection:
                    validation_record["candidate_checkpoint"] = save_checkpoint(
                        f"validation_candidate_step_{step}", step
                    )
                    validation_record["selection_status"] = "pending"
                else:
                    validation_score = (
                        mean_delivery,
                        validation_record["mean_reward"],
                        -mean_drop,
                        -mean_delay,
                    )
                    if (
                        best_validation_score is None
                        or validation_score > best_validation_score
                    ):
                        best_validation_score = validation_score
                        best_validation_record = dict(validation_record)
                        save_checkpoint("validation_best", step)
                    validation_record["is_validation_best"] = (
                        validation_score == best_validation_score
                    )
                validation_records.append(dict(validation_record))
                with open(metrics_path, "a", encoding="utf-8") as file:
                    file.write(json.dumps(validation_record, ensure_ascii=False) + "\n")
        if periodic_checkpoint_label is not None:
            save_checkpoint(
                periodic_checkpoint_label,
                step,
                exact_resume_boundary=True,
            )
        save_checkpoint("latest", step, exact_resume_boundary=True)

    final_checkpoint = save_checkpoint("final", step, exact_resume_boundary=True)
    validation_best_checkpoint = None
    if deferred_validation_selection and validation_records:
        selected = select_leo_validation_record(
            validation_records,
            mode=args.validation_selection_mode,
            delivery_tolerance=args.validation_delivery_tolerance,
            class_2_tolerance=args.validation_class_2_tolerance,
            switch_budget=(
                args.avoidable_switch_budget
                if args.avoidable_switch_constraint_enabled
                else None
            ),
        )
        best_validation_record = dict(selected)
        validation_best_checkpoint = os.path.join(
            checkpoint_root, "validation_best.pt"
        )
        selected_checkpoint_hash = promote_validation_checkpoint(
            selected["candidate_checkpoint"],
            validation_best_checkpoint,
            int(selected["environment_steps"]),
        )
        best_validation_score = (
            float(selected["delivery_ratio"]),
            float(selected["mean_reward"]),
            -float(selected["drop_rate"]),
            -float(selected["average_delay_slots"]),
        )
        selection_record = {
            "record_type": "validation_selection",
            "selection_spec": validation_selection_spec,
            "candidate_count": len(validation_records),
            "selected_candidate_checkpoint": selected["candidate_checkpoint"],
            "selected_candidate_step": int(selected["environment_steps"]),
            "selected_checkpoint_sha256": selected_checkpoint_hash,
            "validation_best_checkpoint": validation_best_checkpoint,
            "selected_metrics": {
                key: selected[key]
                for key in (
                    "environment_steps",
                    "episodes",
                    "seed_start",
                    "delivery_ratio",
                    "mean_reward",
                    "drop_rate",
                    "average_delay_slots",
                    "routing_switches",
                    "routing_switches_total",
                    "avoidable_routing_switches",
                    "avoidable_routing_switches_total",
                    "forced_routing_switches",
                    "forced_routing_switches_total",
                    "switch_opportunities",
                    "avoidable_switch_rate",
                    "class_2_delivery_ratio",
                    "decision_avoidable_switches",
                    "decision_switch_opportunities",
                    "decision_avoidable_switch_rate",
                )
            },
        }
        if args.avoidable_switch_constraint_enabled:
            selection_record["constraint_feasible"] = bool(
                selected["decision_avoidable_switch_rate"]
                <= args.avoidable_switch_budget + 1e-12
            )
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(selection_record, ensure_ascii=False) + "\n")
    elif best_validation_score is not None:
        validation_best_checkpoint = os.path.join(
            checkpoint_root, "validation_best.pt"
        )
    selected_validation_metrics = None
    selected_validation_checkpoint_sha256 = None
    if best_validation_record is not None:
        selected_validation_metrics = {
            key: best_validation_record[key]
            for key in (
                "environment_steps",
                "episodes",
                "seed_start",
                "delivery_ratio",
                "mean_reward",
                "drop_rate",
                "average_delay_slots",
                "routing_switches",
                "routing_switches_total",
                "avoidable_routing_switches",
                "avoidable_routing_switches_total",
                "forced_routing_switches",
                "forced_routing_switches_total",
                "switch_opportunities",
                "avoidable_switch_rate",
                "class_2_delivery_ratio",
                "decision_avoidable_switches",
                "decision_switch_opportunities",
                "decision_avoidable_switch_rate",
            )
            if key in best_validation_record
        }
        if validation_best_checkpoint is not None:
            selected_validation_checkpoint_sha256 = checkpoint_sha256(
                validation_best_checkpoint
            )
    run_manifest = {
        "run_name": run_name,
        "final_checkpoint": final_checkpoint,
        "validation_best_checkpoint": validation_best_checkpoint,
        "candidate_actor_spec": candidate_actor_spec,
        "switch_regularizer_spec": switch_regularizer_spec,
        "switch_constraint_spec": switch_constraint_spec,
        "final_switch_constraint_state": current_switch_constraint_state(),
        "best_validation_score": best_validation_score,
        "validation_selection_spec": validation_selection_spec,
        "validation_candidate_count": len(validation_records),
        "selected_validation_metrics": selected_validation_metrics,
        "selected_validation_checkpoint_sha256": (
            selected_validation_checkpoint_sha256
        ),
        "environment_steps": int(step),
        "optimizer_updates": int(training_step),
        "resumed_from_checkpoint": resume_checkpoint or None,
    }
    final_learned_state = candidate_actor_learned_state()
    if final_learned_state is not None:
        run_manifest["final_candidate_actor_learned_state"] = final_learned_state
        selected_learned_state = (
            best_validation_record or {}
        ).get("candidate_actor_learned_state")
        if selected_learned_state is None:
            raise RuntimeError("selected residual checkpoint lacks learned actor state")
        run_manifest["selected_candidate_actor_learned_state"] = (
            selected_learned_state
        )
    with open(os.path.join(checkpoint_root, "run_manifest.json"), "w", encoding="utf-8") as file:
        json.dump(run_manifest, file, indent=2, ensure_ascii=False)
    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
