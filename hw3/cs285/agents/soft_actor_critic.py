import copy
from typing import Callable, Optional, Sequence, Tuple

import cs285.infrastructure.pytorch_util as ptu
import numpy as np
import torch
from torch import nn


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
            "skip",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(
                observation
            )
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape

        assert num_critic_networks == self.num_critic_networks

        # (9) TODO(student): Implement the different backup strategies.
        if self.target_critic_backup_type == "doubleq":
            raise NotImplementedError
        elif self.target_critic_backup_type == "min":
            raise NotImplementedError
        elif self.target_critic_backup_type == "mean":
            next_qs = next_qs.mean(dim=0, keepdim=True).expand(
                (self.num_critic_networks, batch_size)
            )
        else:
            # Default, we don't need to do anything.
            pass

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = (
                next_qs[None]
                .expand((self.num_critic_networks, batch_size))
                .contiguous()
            )

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # (7) TODO(student)
            # Sample from the actor
            next_action_distribution: torch.distributions.Distribution = self.actor(
                next_obs
            )
            assert isinstance(
                next_action_distribution,
                torch.distributions.Distribution,
            ), type(next_action_distribution)

            next_action = next_action_distribution.sample()
            assert next_action.shape == (batch_size, self.action_dim), next_action.shape

            # (8) TODO Compute the next Q-values for the sampled actions
            next_qs = self.target_critic(next_obs, next_action)  # for given pairs.

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # (10) TODO(student): Add entropy bonus to the target values for SAC
                next_action_entropy = self.entropy(next_action_distribution)
                # As we want to maximize the entropy
                # $\mathbb{E}_{a \sim \pi}[-\log \pi(a|s)]$
                next_qs -= self.temperature * next_action_entropy

            # (8) TODO Compute the target Q-value
            target_values: torch.Tensor = (
                reward.unsqueeze(0)
                + self.discount * (1.0 - done.float().unsqueeze(0)) * next_qs
            )
            assert target_values.shape == (self.num_critic_networks, batch_size)

        # (8) TODO(student): Update the critic
        # Predict Q-values
        q_values = self.critic(obs, action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.

        Providing an entropy bonus to encourage the actor to have high entropy
        (i.e. to be "more random")

        Entropy is defined as:
        $\mathcal{H}(\pi) = \mathbb{E}_{a \sim \pi}[-\log \pi(a|s)]$
        """

        # (10) TODO(student): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        # rsample: differentiable reparameterized sampling

        """
        Sample from the distribution, calculate the log probability of the samples, 
        and return the negative mean.
        
        Normally we do not need the gradients to flow through the sampling process.
        However, maximizing the entropy requires differentiating through the 
        sampling distribution.
        """
        sample = action_distribution.rsample()
        log_probs = action_distribution.log_prob(sample)

        # If action space is multidimensional, sum over action dims
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(dim=-1)

        entropy = -log_probs  # per batch element
        return entropy

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # (11) TODO(student): Generate an action distribution
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # (11) TODO(student): draw num_actor_samples samples from the action distribution
        # for each batch element - NOTE: Don't use no_grad here as we need gradients through log_prob
        action = action_distribution.sample((self.num_actor_samples,))

        assert action.shape == (
            self.num_actor_samples,
            batch_size,
            self.action_dim,
        ), action.shape

        with torch.no_grad():
            # (11) TODO(student): Compute Q-values for the current state-action pair
            """
            Evaluate the critic network on each sampled (s,a) pair.
            """
            q_values_list = []
            for critic in self.critics:
                # obs has the shape (batch_size, obs_dim)
                # action has the shape (num_actor_samples, batch_size, action_dim)
                obs_unsqueeze = obs.unsqueeze(0)  # (1, batch_size, ob_dim)
                # Use expand to pretend to have num_actor_samples copies along the first dimension
                obs_expanded = obs_unsqueeze.expand(self.num_actor_samples, -1, -1)
                assert (
                    obs_expanded.shape[:2] == action.shape[:2]
                ), f"obs and action must have the same num_samples and batch_size, got {obs.shape} vs {action.shape}"

                align_dim = self.num_actor_samples * batch_size
                obs_flat = obs_expanded.reshape(align_dim, obs.shape[1])
                action_flat = action.reshape(align_dim, action.shape[2])
                q = critic(obs_flat, action_flat).view(
                    self.num_actor_samples, batch_size
                )

                assert q.shape == (self.num_actor_samples, batch_size), q.shape
                q_values_list.append(q)

            q_values = torch.stack(q_values_list)

            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, axis=0)

            # Normalize Q-values for REINFORCE to reduce variance and gradient scale issues
            # This is especially important when Q-values are large
            if self.num_actor_samples > 1:
                # For multiple samples, subtract sample mean and normalize by sample std
                q_mean = q_values.mean(dim=0, keepdim=True)
                advantage = q_values - q_mean
            else:
                # For single sample, just center the Q-values around their batch mean
                q_mean = q_values.mean()
                advantage = q_values - q_mean

        # Do REINFORCE: calculate log-probs and use the Q-values
        # (11) TODO(student)
        log_probs = action_distribution.log_prob(action)

        """
        Lets maximize the Q values
        $$\nabla_\theta \mathbb{E}[Q] = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q(s,a)]$$
        """
        loss = -log_probs * advantage
        loss = loss.mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        """
        Used to improve the gradient estimator to give lower variance and support
        few samples.
        Be careful to use the reparametrization trick for sampling.
        """
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # (12) TODO(student): Sample actions
        # Note: Think about whether to use .rsample() or .sample() here...
        action = ...

        # (12) TODO(student): Compute Q-values for the sampled state-action pair
        q_values = ...

        # (12) TODO(student): Compute the actor loss
        loss = ...

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)
        else:
            loss = torch.tensor(0.0, device=obs.device)
            entropy = self.entropy(self.actor(obs)).mean()

            assert isinstance(loss, torch.Tensor), "loss must be a tensor"
            assert isinstance(entropy, torch.Tensor), "entropy must be a tensor"
            assert (
                loss.shape == entropy.shape
            ), f"loss and entropy must have the same shape, got {loss.shape} vs {entropy.shape}"

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        # (9) TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        for _ in range(self.num_critic_updates):
            critic_info = self.update_critic(
                observations, actions, rewards, next_observations, dones
            )
            critic_infos.append(critic_info)

        # (9) TODO(student): Update the actor
        actor_info = self.update_actor(observations)

        # (9) TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        if (
            self.target_update_period is not None
            and step % self.target_update_period == 0
        ):
            # Hard target updates
            self.update_target_critic()

        elif self.soft_target_update_rate is not None:
            # Soft target updates
            self.soft_update_target_critic(self.soft_target_update_rate)

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
