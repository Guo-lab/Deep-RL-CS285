import itertools
import logging

import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
from torch import distributions, nn, optim
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)  # Set default level to INFO
logger = logging.getLogger(__name__)


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # (3) TODO: implement the forward pass of the critic network
        """
        Estimate the value function. This predicts how good it is to be in state s on average.
        So the returned value should be a scalar, an deterministic expectation.
        """
        return self.network(obs)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # (3) TODO: update the critic using the observations and q_values
        """
        The critic itself needs to minimize the difference between its prediction Vϕ(s) 
        and the true target that the critic is trying to regress to (Mean Squared Error).

        E[Gt|st=s,at=a] = Q(s,a)
        """
        expectation = self(obs)
        if expectation.shape != q_values.shape:
            # logger.warning(
            #     f"Value shape {expectation.shape} != q_values shape {q_values.shape}. Squeezing the Tensor."
            # )
            expectation = expectation.squeeze(-1)
        loss = F.mse_loss(expectation, q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline_Loss": ptu.to_numpy(loss),
        }
