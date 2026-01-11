from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

        self._debug_printed = False

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        """
        Compute the TD error. How to compute the next state value is different from DQN.
        For DQN, next Q-values would be calculated based on argmax Q(s',a'), while for
        AWAC, the actor gives the action distribution π(a'|s') to compute E_π[Q(s',a')].

        Critic provides Q estimation for each state-action pair.
        (outputs Q(s,a) for all actions)
        """
        with torch.no_grad():
            # (5) TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            next_qa_values = self.target_critic(next_observations)

            # Use the actor to compute a critic backup
            next_actions_dist = self.actor(next_observations)

            # Expected Q under policy: E_π[Q(s', a')] = Σ π(a'|s') * Q(s', a')
            next_q_values = (next_actions_dist.probs * next_qa_values).sum(dim=1)
            assert next_q_values.shape == rewards.shape, next_q_values.shape

            # (5) TODO(student): Compute the TD target
            target_values = rewards + self.discount * (1 - dones.float()) * next_q_values

        # (5) TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = qa_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        assert q_values.shape == target_values.shape

        # Qϕ​(s,a) ← r+ γVψ​(s′)
        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # (6) TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        qa_values = self.target_critic(observations)

        # instead of online critic in paper to decrease variance
        q_values = qa_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # E_π[Q(s, a)] = Σ π(a|s) * Q(s, a) sum over actions per sample
        # Value function is estimated as the expectation over actions in the dataset.
        if action_dist is None:
            action_dist = self.actor(observations)
        values = (action_dist.probs * qa_values).sum(dim=1)
        assert q_values.shape == values.shape

        # A(s,a) = Q(s,a) - V(s) will be computed separately for the actor update
        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        AWAC actor update: weighted behavior cloning based on advantages.

        The actor learns to imitate actions from the dataset, weighted by their advantages,
        which is predicted from the learned critic.
        High-advantage actions get larger weights, encouraging the actor to focus on
        good actions while staying close to the behavior policy.

        In return, the actor guides critic updates by providing the policy distribution
        used to compute expected Q-values in TD targets (see compute_critic_loss).
        This makes the critic less overoptimistic on out-of-distribution actions by using
        policy-averaged Q-values instead of max Q-values.
        Even if an OOD action has a high Q-value, it contributes less to the expected value
        when π(a|s) is low.
        """
        # (7) TODO(student): update the actor using AWAC
        # θ ← arg max E[log π(a|s) exp(1/λ A(s,a))]
        action_dist = self.actor(observations)

        with torch.no_grad():
            # Ensures neither advantage nor weights track gradients from the critic
            # and prevent gradients flow back to BOTH actor AND critic.
            advantage = self.compute_advantage(observations, actions, action_dist)
            weights = torch.exp(advantage / self.temperature)
            # weights = torch.clamp(weights, max=20.0)
            # weights = weights / (weights.mean() + 1e-8)

        log_probs = action_dist.log_prob(actions)

        """
        log π(a|s) plateaus and actor loss converges once the policy matches
        the dataset actions.
        """
        loss = -(weights * log_probs).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        if not self._debug_printed:
            total_norm = 0.0
            for p in self.actor.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm**0.5
            print(f"Actor Grad Norm: {total_norm}")
            self._debug_printed = True

        # Return metrics for logging and debugging
        return {
            "actor_loss": loss.item(),
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item(),
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),
            "log_prob_mean": log_probs.mean().item(),
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = super().update(
            observations, actions, rewards, next_observations, dones, step
        )

        # Update the actor and merge metrics
        actor_metrics = self.update_actor(observations, actions)
        metrics.update(actor_metrics)

        """
        Debugging:
        If the advantage is 0 and weights is 1, the actor is doing the pure BC.
        
        Since the log-probabilities depend on the actor parameters and 
        the gradient norm is non-zero, the actor is being updated.
        
        Critic chases a moving target forever so the loss of the critic 
        never converges.
        """

        return metrics
