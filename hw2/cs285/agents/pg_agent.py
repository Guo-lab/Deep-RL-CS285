import logging
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from cs285.infrastructure import pytorch_util as ptu
from cs285.networks.critics import ValueCritic
from cs285.networks.policies import MLPPolicyPG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """
        The train step for PG involves updating its actor using the given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory.
        The batch size is the total number of samples across all trajectories (i.e. the sum of the
        lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # (5) TODO: flatten the lists of arrays into single arrays, so that the rest of the code
        # can be written in a vectorized way. obs, actions, rewards, terminals, and q_values
        # should all be arrays with a leading dimension of `batch_size` beyond this point.
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        q_values = np.concatenate(q_values, axis=0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # (7) TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update
        # the PG critic/baseline
        if self.critic is not None:
            # (7) TODO: perform `self.baseline_gradient_steps` updates to
            # the critic/baseline network
            for _ in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(obs, q_values)

                info.update(critic_info)

            # info.update(critic_info)
        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """
        Monte Carlo estimation of the Q function.

        Args:
        rewards: a list of NumPy arrays, each array is a sequence of rewards
        for a trajectory

        Returns:
        q_values: a list of NumPy arrays, each array is a sequence of estimated Q values
        for the corresponding trajectory
        """

        q_values = []

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted
            # return for the entire trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # (4) TODO: use the helper function self._discounted_return to calculate the Q-values

            for rew in rewards:
                q_values.append(self._discounted_return(rew))

        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # (4) TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values

            for rew in rewards:
                q_values.append(self._discounted_reward_to_go(rew))

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """
        Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # (6) TODO: if no baseline, then what are the advantages?

            # A(s,a) = Q(s,a) - baseline(s)
            # We have no learned value function V(s).
            # Actions with higher Q-values get reinforced more
            advantages = q_values
        else:
            # (6) TODO: run the critic and use it as a baseline
            values = self.critic(ptu.from_numpy(obs))
            values = ptu.to_numpy(values)
            values = values.squeeze()

            logger.debug(
                f"Values shape: {values.shape}, Q values shape: {q_values.shape}"
            )

            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # (6) TODO: if using a baseline, but not GAE, what are the advantages?

                # Monte Carlo advantage (high variance)
                # That is A(s,a) = Q(s,a) - V(s)
                advantages = q_values - values
            else:
                # (6) TODO: implement GAE (Generalized Advantage Estimation)
                # GAE = discounted sum of Temporal Difference errors
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # (6) TODO: recursively compute advantage estimates starting from timestep T.

                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is
                    # the last in its trajectory, and 0 otherwise.
                    # When the terminal is 1, there is no next state so V(s_{t+1}) should be treated as 0.

                    # At timestep i, you need to:
                    # 1. Compute TD error δ_i: bootstrapped target minus the current estimate
                    # δ_i = r_i + (1 - terminal_i) * γ * V(s_{i+1}) - V(s_i)
                    delta_i = (
                        rewards[i]
                        + (1 - terminals[i]) * self.gamma * values[i + 1]
                        - values[i]
                    )
                    # 2. Compute advantage using recursion
                    # A_i = δ_i + (1 - terminal_i) * γ * λ * A_{i+1}
                    # The earlier advantage estimates depend on the later ones, thus more items
                    # summed up and the same moment delta getting counted more times.
                    advantages[i] = (
                        delta_i
                        + (1 - terminals[i])
                        * self.gamma
                        * self.gae_lambda
                        * advantages[i + 1]
                    )

                # remove dummy advantage
                advantages = advantages[:-1]

        # (6) TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (
                np.std(advantages) + 1e-8
            )

        logger.debug(
            f"Advantages mean: {np.mean(advantages)}, std: {np.std(advantages)}"
        )
        logger.debug(f"Advantages shape: {advantages.shape}")

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        (4) TODO: Implement the trajectory-based Q-value estimator.

        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is
        from 0 to T (and doesn't involve t)!
        """
        entry_value = 0.0
        gamma_power = 1.0

        for reward in rewards:
            entry_value += gamma_power * reward
            gamma_power *= self.gamma

        return [entry_value for _ in rewards]

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        (4) TODO: Compute reward-to-go for each timestep.

        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        discounted_reward_to_go = []
        running_ret = 0.0

        """
        G_t = r_t + gamma r_{t+1} + gamma² r_{t+2} + ...
        which has the recursive form: G_t = gamma G_{t+1} + r_t
        """
        for reward in reversed(rewards):
            running_ret = self.gamma * running_ret + reward
            discounted_reward_to_go.append(running_ret)

        return list(reversed(discounted_reward_to_go))
