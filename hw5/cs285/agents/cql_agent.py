from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

        # Log CQL parameters for verification
        print(f"[CQL] Initialized with alpha={self.cql_alpha}, temperature={self.cql_temperature}")

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the second term after the DQN loss computation to penalizes
        Q-values of "unseen actions", making Q conservative. To prevent
        the network from overestimating out-of-distribution actions.
        """
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        # (4) TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your
        # CQL implementation
        """
        Compute the soft-maximum (logsumexp) of all Q-values and average this
        Q of any actions.
        The average over dataset actions has been computed by DQN.
        
        By E a~any [Q] - E a~D [Q], Q-values of dataset actions are not penalized, 
        and the Q-values of unseen actions would be focused on.
        """
        cql_regularizer = (
            torch.logsumexp(variables["qa_values"] / self.cql_temperature, dim=-1)
            - variables["q_values"]
        )
        # α * E_batch[ log(Σ_a exp(Q(s,a)/τ)) - Q(s, a_dataset) ]
        loss = loss + self.cql_alpha * cql_regularizer.mean()

        return loss, metrics, variables
