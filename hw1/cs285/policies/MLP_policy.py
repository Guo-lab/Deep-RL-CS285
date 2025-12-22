"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
import logging
from typing import Any

import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from torch import distributions, nn, optim
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)  # Set default level to INFO
logger = logging.getLogger(__name__)


def build_mlp(input_size: int, output_size: int, n_layers: int, size: int) -> nn.Module:
    """
    Builds a feedforward neural network

    arguments:
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """

    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        nn_baseline=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.mean_net.to(ptu.device)

        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)

        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate,
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
                            shape (batch_size, ob_dim)  e.g., (100, 27) for Ant-v4
        :return:
            action: sampled action(s) from the policy
            We want to use the full Gaussian distribution, which will maintain
            the notion of uncertainty.
        """

        # (2) TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!

        if not torch.is_tensor(observation):
            observation = ptu.from_numpy(observation)

        # Feed observations (states) into the network to get predicted actions.

        logger.debug("FORWARD PASS. The shape of observations is", observation.shape)

        mu = self.mean_net(observation)
        std = torch.exp(self.logstd)

        # Create a normal distribution with mean mu and standard deviation std.
        dist = distributions.Normal(mu, std)
        return dist

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # (2) TODO: update the policy and return the loss

        # $\pi(a \mid o) = \mathcal{N}(\mu(o), \sigma^2)$
        action_dist = self.forward(observations)

        # Loss is negative log-likelihood loss.
        # How likely the expert actions are under our policy.
        if not torch.is_tensor(actions):
            actions = ptu.from_numpy(actions)
        nll = -action_dist.log_prob(actions)

        loss = nll.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            "Training Loss": ptu.to_numpy(loss),
        }
