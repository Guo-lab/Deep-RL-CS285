from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from cs285.env_configs.dqn_config import basic_dqn_config
import cs285.infrastructure.pytorch_util as ptu
from cs285.networks.mlp_policy import MLPPolicy


def awac_config(
    total_steps: int = 50000,
    discount: float = 0.95,
    temperature: float = 1.0,
    actor_hidden_size: int = 128,
    actor_num_layers: int = 2,
    actor_learning_rate: float = 3e-4,
    **kwargs,
):
    make_actor = lambda obs_shape, num_actions: MLPPolicy(
        num_actions,
        obs_shape[0],
        discrete=True,
        n_layers=actor_num_layers,
        layer_size=actor_hidden_size,
    )
    make_actor_optimizer = lambda params: torch.optim.Adam(params, lr=actor_learning_rate)

    config = basic_dqn_config(total_steps=total_steps, discount=discount, **kwargs)
    config["log_name"] = "{env_name}_awac{temperature}".format(
        env_name=config["env_name"], temperature=temperature
    )
    config["agent"] = "awac"

    config["agent_kwargs"]["temperature"] = temperature
    config["agent_kwargs"]["make_actor"] = make_actor
    config["agent_kwargs"]["make_actor_optimizer"] = make_actor_optimizer

    # Print all agent_kwargs during runtime
    print("\n" + "=" * 80)
    print("AWAC Config - agent_kwargs:")
    print("=" * 80)
    for key, value in config["agent_kwargs"].items():
        if callable(value):
            print(f"  {key:30s}: <callable: {value.__class__.__name__}>")
        else:
            print(f"  {key:30s}: {value}")
    print(f"  {'total_steps':30s}: {total_steps}")
    print(f"  {'actor_hidden_size':30s}: {actor_hidden_size}")
    print(f"  {'actor_num_layers':30s}: {actor_num_layers}")
    print(f"  {'actor_learning_rate':30s}: {actor_learning_rate}")
    print("=" * 80 + "\n")

    return config
