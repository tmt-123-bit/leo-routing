import torch
import datetime
from dataclasses import asdict
import os
import random
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from mappo_design import (
    PacketConditionedCritic,
    RunningMeanStd as DesignRunningMeanStd,
    SharedCandidateActor,
    compute_gae,
    feasible_normalized_entropy,
    masked_standardize,
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
    normalize_return: bool = False
    """ Normalize the returns if True"""
    epochs: int = 3
    """ Number of training epochs"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.01
    """ Entropy coefficient """
    log_every: int = 10
    """ Logging steps """
    clip_gradients: float = 0.5
    """Global actor/critic gradient norm limit; <=0 disables clipping."""
    normalization_epsilon: float = 1e-8
    """Numerical floor used by reward/advantage/return normalization."""
    target_kl: float = 0.02
    """Stop remaining PPO epochs when the measured KL exceeds this value."""
    candidate_shared_actor: bool = False
    """Use a permutation-equivariant shared candidate scorer."""
    leo_project_path: str = "F:/leo-routing-preliminary-matlab"
    """Directory containing cleanmarl_leo_wrapper.py for env_type=leo."""
    eval_steps: int = 10
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 10
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
        reward = torch.zeros((self.buffer_size, max_length)).to(self.device)
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
            mask[i, :length] = 1
        if self.normalize_reward:
            valid_reward = reward[mask].detach().cpu().numpy()
            self.reward_rms.update(valid_reward)
            reward[mask] = (reward[mask] - self.reward_rms.mean) / np.sqrt(
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
    def __init__(self, input_dim, hidden_dim, num_layer) -> None:
        super().__init__()
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

        env = CleanMARLLeoWrapper(scenario=env_name)
    elif env_type == "leo_multi":
        project_path = kwargs.pop(
            "project_path",
            os.environ.get("LEO_ROUTING_PROJECT", "F:/leo-routing-preliminary-matlab"),
        )
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
        from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper

        env = CleanMARLLeoMultiAgentWrapper(scenario=env_name)
    else:
        raise ValueError(f"unknown env_type: {env_type}")

    return env


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads if g is not None]
    if not norms:
        return torch.tensor(0.0)
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


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
    env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )
    eval_env = environment(
        env_type=args.env_type,
        env_name=args.env_name,
        env_family=args.env_family,
        agent_ids=args.agent_ids,
        kwargs=kwargs,
    )

    ## Initialize the actor, critic and target-critic networks
    candidate_feature_dim = None
    if args.env_type in {"leo", "leo_multi"} or args.candidate_shared_actor:
        getter = getattr(env, "get_candidate_feature_dim", None)
        if getter is None:
            raise ValueError(
                "candidate_shared_actor requires env.get_candidate_feature_dim()"
            )
        candidate_feature_dim = getter()
    actor = Actor(
        input_dim=env.get_obs_size(),
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=env.get_action_size(),
        candidate_feature_dim=candidate_feature_dim,
    ).to(device)
    critic = Critic(
        input_dim=env.get_state_size(),
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
    ).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
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
    checkpoint_root = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_root, exist_ok=True)

    def save_checkpoint(label, current_step):
        path = os.path.join(checkpoint_root, f"{label}.pt")
        torch.save(
            {
                "step": current_step,
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
            },
            path,
        )
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
    num_episodes = 0
    step = 0
    next_save_step = args.save_every_steps
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
                "avail_actions": [],
            }
            obs, _ = env.reset()
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
                next_state = env.get_state()
                ep_reward += reward
                ep_length += 1
                step += 1
                episode["obs"].append(obs)
                episode["actions"].append(actions.cpu())
                episode["log_prob"].append(log_probs.cpu())
                episode["reward"].append(reward)
                episode["next_states"].append(next_state)
                episode["terminated"].append(done)
                episode["truncated"].append(truncated)
                episode["policy_active"].append(policy_active)
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
            b_mask,
        ) = rb.get_batch()

        # GAE with explicit terminated/truncated semantics. Terminated states
        # zero-bootstrap; time-limit truncations bootstrap from next_state.
        with torch.no_grad():
            values = critic(b_states).squeeze(-1).unsqueeze(-1).expand(
                -1, -1, env.n_agents
            )
            next_values = critic(b_next_states).squeeze(-1).unsqueeze(-1).expand(
                -1, -1, env.n_agents
            )
            agent_rewards = b_reward.unsqueeze(-1).expand(-1, -1, env.n_agents)
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
        # training loop
        actor_losses = []
        critic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        actor_gradients_post_clip = []
        critic_gradients_post_clip = []
        clipped_ratios = []
        for epoch_idx in range(args.epochs):
            actor_loss = 0
            critic_loss = 0
            entropies = 0
            kl_divergence = 0
            clipped_ratio = 0
            for t in range(b_obs.size(1)):
                step_agent_mask = (
                    b_mask[:, t].unsqueeze(-1).expand(-1, env.n_agents)
                    & b_policy_active[:, t]
                )
                if not step_agent_mask.any():
                    continue
                # policy gradient (PG) loss
                ## PG: compute the ratio:
                current_logits = actor.logits(
                    x=b_obs[:, t], avail_action=b_avail_actions[:, t]
                )
                current_dist = Categorical(logits=current_logits)
                current_logprob = current_dist.log_prob(b_actions[:, t])

                log_ratio = current_logprob - b_log_probs[:, t]
                ratio = torch.exp(log_ratio)
                ## Compute PG the loss
                pg_loss1 = advantages[:, t] * ratio
                pg_loss2 = advantages[:, t] * torch.clamp(
                    ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
                )
                pg_loss = (
                    torch.min(pg_loss1, pg_loss2)[step_agent_mask].mean()
                )

                # Compute entropy bonus
                normalized_entropy = feasible_normalized_entropy(
                    current_dist, b_avail_actions[:, t]
                )
                entropy_loss = normalized_entropy[step_agent_mask].mean()
                entropies += entropy_loss
                actor_loss += -pg_loss - args.entropy_coef * entropy_loss

                # Compute the value loss
                current_values = critic(x=b_states[:, t]).expand(-1, env.n_agents)
                value_prediction = current_values[b_mask[:, t]]
                value_target = return_lambda[:, t][b_mask[:, t]]
                if args.normalize_return:
                    scale = ret_std + args.normalization_epsilon
                    value_prediction = (value_prediction - ret_mu) / scale
                    value_target = (value_target - ret_mu) / scale
                value_loss = F.mse_loss(
                    value_prediction, value_target
                ) * b_mask[:, t].sum()
                critic_loss += value_loss

                # track kl distance
                b_kl_divergence = (
                    ((ratio - 1) - log_ratio)[step_agent_mask].mean()
                )
                kl_divergence += b_kl_divergence
                clipped_ratio += (
                    ((ratio - 1.0).abs() > args.ppo_clip)[step_agent_mask]
                    .float()
                    .mean()
                )

            actor_loss /= max(1, b_mask.size(1))
            critic_loss /= b_mask.sum()
            entropies /= b_mask.sum()
            kl_divergence /= b_mask.sum()
            clipped_ratio /= b_mask.sum()

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
            critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=args.clip_gradients
                )
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), max_norm=args.clip_gradients
                )
            actor_gradient_post = norm_d([p.grad for p in actor.parameters()], 2)
            critic_gradient_post = norm_d([p.grad for p in critic.parameters()], 2)
            actor_optimizer.step()
            critic_optimizer.step()
            training_step += 1

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies_bonuses.append(entropies.item())
            kl_divergences.append(kl_divergence.item())
            actor_gradients.append(actor_gradient)
            critic_gradients.append(critic_gradient)
            actor_gradients_post_clip.append(actor_gradient_post)
            critic_gradients_post_clip.append(critic_gradient_post)
            clipped_ratios.append(clipped_ratio.cpu())
            if args.target_kl > 0 and kl_divergence.item() > args.target_kl:
                writer.add_scalar("train/early_stop_epoch", epoch_idx + 1, step)
                break

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
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
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
        if args.save_every_steps > 0 and step >= next_save_step:
            save_checkpoint(f"step_{step}", step)
            while next_save_step <= step:
                next_save_step += args.save_every_steps

        if (training_step / args.epochs) % args.eval_steps == 0:
            eval_obs, _ = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
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
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs, _ = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    np.mean([info["battle_won"] for info in eval_ep_stats]),
                    step,
                )

    save_checkpoint("final", step)
    writer.close()
    if args.use_wnb:
        wandb.finish()
    env.close()
    eval_env.close()
