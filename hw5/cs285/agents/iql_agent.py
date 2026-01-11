from turtle import done
from typing import Optional
import torch
from torch import nn
from cs285.agents.awac_agent import AWACAgent

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(AWACAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_value_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_value_critic_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        expectile: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )

        self.value_critic = make_value_critic(observation_shape)
        self.target_value_critic = make_value_critic(observation_shape)
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())

        self.value_critic_optimizer = make_value_critic_optimizer(
            self.value_critic.parameters()
        )
        self.expectile = expectile

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # (11) TODO(student): Compute advantage with IQL
        """
        This function called in super class but in IQL, the actor update
        may not use advantage at all.

        In this way, IQL completely avoids querying OOD actions as it doesn't
        depend on the actor at all.
        IQL is “actor-independent” in advantage computation, making it safer
        offline.
        """
        with torch.no_grad():
            # to stabilize advantage
            qa_values = self.target_critic(observations)
            q_values = qa_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

            # online value network would create non-stationarity.
            vs = self.target_value_critic(observations)

        return q_values - vs

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)

        L_Q(θ) = E[(r(s,a) + gamma V(s') - Q(s,a))²]
        Uses V(s') instead of max Q(s',a')

        Here, IQL also remove the dependency on actor, so the critic update can be conducted
        without an actor update. Thus we can first train a critic and then only train the actor
        at the end.
        """
        # (10) TODO(student): Update Q(s, a) to match targets (based on V)
        qa_values = self.critic(observations)
        q_values = qa_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # Fixed targets because we're trying to make Q fit the targets.
        with torch.no_grad():
            vs_prime = self.target_value_critic(next_observations)
            target_values = rewards + self.discount * (1 - dones.float()) * vs_prime

        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        metrics = {
            "q_loss": self.critic_loss(q_values, target_values).item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "q_grad_norm": grad_norm.item(),
        }

        return metrics

    @staticmethod
    def iql_expectile_loss(
        expectile: float, vs: torch.Tensor, target_qs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        Average L²_τ(μ) = |τ - 1{μ ≤ 0}|μ² over all samples

        Return 1D Ex~X[L2]
        """
        # (8) TODO(student): Compute the expectile loss
        mu = target_qs - vs
        indicator = (mu <= 0).float()  # shape: (batch,)
        return torch.mean(torch.abs(expectile - indicator) * mu**2)

    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the value network V(s) using targets Q(s, a)

        V(s) is the direct output of the value network self.value_critic.

        V is a separate learned function trained via expectile regression
        instead of E_π[Q(s,a)] = Σ π(a|s) * Q(s,a) computed as the
        expected Q-value under the policy in AWAC or policy-dependent one in SAC.

        IQL's V(s) learns an expectile of Q-values from the dataset, acting
        as an optimistic baseline.
        """
        # (9) TODO(student): Compute target values for V(s)
        # V(s) predictions for each sample in the batch
        vs = self.value_critic(observations)

        # (9) TODO(student): Update V(s) using the loss from the IQL paper
        """
        n IQL paper, the expectile is applied over online Q(s,a) for each batch. 
        Using target_critic here is not standard, may slow learning
        """
        with torch.no_grad():
            qa_values = self.critic(observations)
            target_values = qa_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        loss = self.iql_expectile_loss(self.expectile, vs, target_values)

        self.value_critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.value_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.value_critic_optimizer.step()

        return {
            "v_loss": loss.item(),
            "vs_adv": (vs - target_values).mean().item(),
            "vs": vs.mean().item(),
            "target_values": target_values.mean().item(),
            "v_grad_norm": grad_norm.item(),
        }

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s)
        """

        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_v = self.update_v(observations, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = self.update_critic(observations, actions, rewards, next_observations, dones)

        # Update actor and merge metrics
        actor_metrics = self.update_actor(observations, actions)
        metrics.update(actor_metrics)

        if step % self.target_update_period == 0:
            self.update_target_critic()
            self.update_target_value_critic()

        return metrics

    def update_target_value_critic(self):
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
