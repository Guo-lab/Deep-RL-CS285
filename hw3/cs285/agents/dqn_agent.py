from typing import Callable, Optional, Sequence, Tuple

import cs285.infrastructure.pytorch_util as ptu
import numpy as np
import torch
from torch import nn


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],  # factory functions passed in from outside
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

        self._printed_device = False

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # (1) TODO(student): get the action from the critic using an
        # epsilon-greedy strategy
        # With probability epsilon: do something random.
        # With probability 1-epsilon: do something greedy.
        if np.random.random() < epsilon:
            action = torch.randint(0, self.num_actions, (1,))  # one batch
        else:
            q_values = self.critic(observation)
            # max among Q-values for all actions
            action = torch.argmax(q_values, dim=-1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        """ 
        target = r + gamma * (1 - done) * max_a' Q_target(s', a')
        When the epoisode ends at step t, there is no next state s_{t+1}, so the 
        target is just the reward.
        """

        with torch.no_grad():  # Targets are fixed.
            """
            We need Q-values for all actions in the next state (batch_size x num_actions)
            and the action with the highest Q-value in next state (shape: batch_size).
            """
            # (2) TODO(student): compute target values
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # (6) The double Q-learning trick changes how we select the next action
                # for computing the target value.
                # We still use the target network to estimate the action values, but we
                # will select the next action using the online network.
                online_qa_values = self.critic(next_obs)
                next_action = torch.argmax(online_qa_values, dim=-1)

            else:
                next_action = torch.argmax(next_qa_values, dim=-1)

            """
            next_q_values: Select per-sample actions from a batched Q-table.
            The next_qa_values is used to decide, but the next_q_values is used to evaluate.
            """
            next_q_values = next_qa_values.gather(
                dim=1, index=next_action.unsqueeze(1)
            ).squeeze(1)
            target_values = reward + self.discount * (1 - done.float()) * next_q_values

        # (2) TODO(student): train the critic with the target values
        """
        For the current state, train the online critic.
        Computes MSE between predictions (q_values) and targets.
        """
        qa_values = self.critic(obs)
        q_values = qa_values.gather(dim=1, index=action.unsqueeze(1)).squeeze(1)
        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        if not self._printed_device:
            print(obs.device, next_obs.device, action.device, reward.device)
            self._printed_device = True

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # (3) TODO(student): update the critic, and the target if needed
        """
        (i) Update the critic using the sampled batch.
        (ii) Periodically update target network.
        (iii) Return the stats for the critic we are training
        """

        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
