from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from cs285.env_configs.dqn_config import basic_dqn_config
import cs285.infrastructure.pytorch_util as ptu

def cql_config(
    cql_alpha: float = 1.0,
    cql_temperature: float = 1.0,
    total_steps: int = 50000,
    discount: float = 0.95,
    **kwargs,
):
    config = basic_dqn_config(total_steps=total_steps, discount=discount, **kwargs)
    config["log_name"] = "{env_name}_cql{cql_alpha}".format(
        env_name=config["env_name"], cql_alpha=cql_alpha
    )
    config["agent"] = "cql"

    config["agent_kwargs"]["cql_alpha"] = cql_alpha
    config["agent_kwargs"]["cql_temperature"] = cql_temperature

    # Print all agent_kwargs during runtime
    print("\n" + "=" * 80)
    print("CQL Config - agent_kwargs:")
    print("=" * 80)
    for key, value in config["agent_kwargs"].items():
        if callable(value):
            print(f"  {key:30s}: <callable: {value.__class__.__name__}>")
        else:
            print(f"  {key:30s}: {value}")
    print(f"  {'total_steps':30s}: {total_steps}")
    print(f"  {'cql_alpha':30s}: {cql_alpha}")
    print(f"  {'cql_temperature':30s}: {cql_temperature}")
    print("=" * 80 + "\n")

    return config
