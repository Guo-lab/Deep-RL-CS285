from typing import Callable, Optional, Tuple
from xml.etree.ElementTree import tostringlist
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer("obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device))
        self.register_buffer("obs_delta_std", torch.ones(self.ob_dim, device=ptu.device))

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)

        # (2) TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas)
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        """
        Compute ground truth deltas from data
        Normalize inputs and outputs using the global statistics (in update_statistics)
        Forward pass through model
        Compute loss between predicted and actual normalized deltas
        Backprop.
        """
        #  Prepare the data from this batch
        obs_acs = torch.cat([obs, acs], dim=1)  # Use batch data
        deltas = next_obs - obs  # Use batch data

        #  Normalize using GLOBAL statistics
        norm_in = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + 1e-8)
        norm_gt = (deltas - self.obs_delta_mean) / (self.obs_delta_std + 1e-8)

        norm_pred = self.dynamics_models[i](norm_in)
        loss = self.loss_fn(norm_pred, norm_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)

        # (1) TODO(student): update the statistics
        #  Average over (batch, obsDim + acsDim)
        #  ONCE on ALL data in replay buffer.
        self.obs_acs_mean = torch.cat([obs, acs], dim=1).mean(dim=0)
        self.obs_acs_std = torch.cat([obs, acs], dim=1).std(dim=0) + 1e-8
        self.obs_delta_mean = (next_obs - obs).mean(dim=0)
        self.obs_delta_std = (next_obs - obs).std(dim=0) + 1e-8

    @torch.no_grad()
    def get_dynamics_predictions(self, i: int, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)

        # (3) TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        obs_acs = torch.cat([obs, acs], dim=1)
        norm_in = (obs_acs - self.obs_acs_mean) / (self.obs_acs_std + 1e-8)

        pred = self.dynamics_models[i](norm_in)
        unnorm_delta = pred * (self.obs_delta_std + 1e-8) + self.obs_delta_mean

        pred_next_obs = obs + unnorm_delta
        return ptu.to_numpy(pred_next_obs)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # (6) TODO(student): for each batch of actions in in the horizon...
        """
        Takes K action sequences of length H, then for each sequence, predicts 
        N trajectories (one per ensemble model)
        Computes rewards for all trajectories and averages over ensemble to 
        get K reward values
        """
        for acs in action_sequences.transpose(1, 0, 2):
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # (6) TODO(student): predict the next_obs for each rollout
            # HINT: use self.get_dynamics_predictions
            next_obs_list = []
            for i in range(self.ensemble_size):
                next_ob = self.get_dynamics_predictions(i, obs[i], acs)
                next_obs_list.append(next_ob)
            next_obs = np.stack(next_obs_list, axis=0)

            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # (6) TODO(student): get the reward for the current step in each rollout
            # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
            # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
            # respectively, and returns a tuple of `(rewards, dones)`. You can
            # ignore `dones`. You might want to do some reshaping to make
            # `next_obs` and `acs` 2-dimensional.
            """
            Shape the obs to (ensemble_size * mpc_num_action_sequences, ob_dim)
            Repeat the acs's row ensemble_size times.
            """
            reshaped_next_obs = next_obs.reshape(-1, self.ob_dim)
            acs = np.tile(acs, (self.ensemble_size, 1))

            rewards, _ = self.env.get_reward(reshaped_next_obs, acs)
            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]

        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                # (7) TODO(student): implement the CEM algorithm
                # HINT: you need a special case for i == 0 to initialize
                # the elite mean and std
                if i == 0:
                    """
                    Initialize distribution
                    elite_mean and elite_std represent the distribution of action sequences
                    """
                    rewards = self.evaluate_action_sequences(obs, action_sequences)
                    top_J_indices = np.argsort(rewards)[-self.cem_num_elites :]
                    elite_actions = action_sequences[top_J_indices]
                    """
                    The mean and std should also follow the evaluation actions at very beginning.
                    
                    Note that if use top_idx = np.argsort(rewards)[-1:], then it would be
                    same as random shooting one.
                    """
                    elite_mean = elite_actions.mean(axis=0)
                    elite_std = elite_actions.std(axis=0) + 1e-8

                else:
                    # CEM Sample action sequences from current distribution.
                    current_sequences = np.random.normal(
                        elite_mean,
                        elite_std,
                        size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
                    )
                    current_sequences = np.clip(
                        current_sequences,
                        self.env.action_space.low,
                        self.env.action_space.high,
                    )
                    assert current_sequences.shape == (
                        self.mpc_num_action_sequences,
                        self.mpc_horizon,
                        self.ac_dim,
                    )

                    rewards = self.evaluate_action_sequences(obs, current_sequences)
                    assert rewards.shape == (self.mpc_num_action_sequences,)

                    """
                    Choose the J sequences with the highest predicted sum of discounted rewards 
                    as the ”elite” action sequence
                    """
                    top_J_indices = np.argsort(rewards)[-self.cem_num_elites :]
                    elite_actions = current_sequences[top_J_indices]

                    """
                    Fit a diagonal Gaussian with the same mean and variance as the 
                    "elite" action sequences and use this as our action sampling distribution 
                    for the next iteration
                    """
                    new_mean = elite_actions.mean(axis=0)
                    new_std = elite_actions.std(axis=0) + 1e-8
                    # alpha from yaml is 1 by default
                    elite_mean = self.cem_alpha * new_mean + (1 - self.cem_alpha) * elite_mean
                    elite_std = self.cem_alpha * new_std + (1 - self.cem_alpha) * elite_std

            return elite_mean[0]

        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
