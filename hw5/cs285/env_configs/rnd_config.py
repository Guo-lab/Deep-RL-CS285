import torch
from cs285.env_configs.dqn_config import basic_dqn_config
from cs285.infrastructure import pytorch_util as ptu


def rnd_config(
    rnd_weight: float,
    rnd_dim: int = 5,
    rnd_network_hidden_size: int = 400,
    rnd_network_num_layers: int = 2,
    rnd_network_learning_rate: float = 1e-3,
    total_steps: int = 50000,
    discount: float = 0.95,
    **kwargs,
):
    config = basic_dqn_config(total_steps=total_steps, discount=discount, **kwargs)
    config["agent_kwargs"]["rnd_weight"] = rnd_weight
    config["log_name"] = "{env_name}_rnd{rnd_weight}".format(
        env_name=config["env_name"], rnd_weight=rnd_weight
    )
    config["agent"] = "rnd"

    config["agent_kwargs"]["make_rnd_network"] = lambda obs_shape: ptu.build_mlp(
        input_size=obs_shape[0],
        output_size=rnd_dim,
        n_layers=rnd_network_num_layers,
        size=rnd_network_hidden_size,
    )
    config["agent_kwargs"]["make_target_rnd_network"] = lambda obs_shape: ptu.build_mlp(
        input_size=obs_shape[0],
        output_size=rnd_dim,
        n_layers=rnd_network_num_layers,
        size=rnd_network_hidden_size,
    )
    config["agent_kwargs"][
        "make_rnd_network_optimizer"
    ] = lambda params: torch.optim.Adam(params, lr=rnd_network_learning_rate)

    # Print all agent_kwargs during runtime
    print("\n" + "=" * 80)
    print("RND Config - agent_kwargs:")
    print("=" * 80)
    for key, value in config["agent_kwargs"].items():
        if callable(value):
            print(f"  {key:30s}: <callable: {value.__class__.__name__}>")
        else:
            print(f"  {key:30s}: {value}")
    print(f"  {'total_steps':30s}: {total_steps}")
    print(f"  {'rnd_weight':30s}: {rnd_weight}")
    print(f"  {'rnd_dim':30s}: {rnd_dim}")
    print(f"  {'rnd_network_hidden_size':30s}: {rnd_network_hidden_size}")
    print(f"  {'rnd_network_num_layers':30s}: {rnd_network_num_layers}")
    print(f"  {'rnd_network_learning_rate':30s}: {rnd_network_learning_rate}")
    print("=" * 80 + "\n")

    return config
