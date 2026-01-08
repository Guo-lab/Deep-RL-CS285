import argparse
import os
import time
import warnings

import cs285.env_configs
import gym
import numpy as np
import torch
import tqdm
import yaml
from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer
from gym import wrappers
from scripting_utils import make_config, make_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"]

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # (6) TODO(student): Select an action
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()
        else:
            observation = next_observation

        # Train the agent
        if step >= config["training_starts"]:
            # (6) TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)

            update_info = agent.update(
                observations=batch["observations"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                next_observations=batch["next_observations"],
                dones=batch["dones"],
                step=step,
            )

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    # Allow overriding config parameters
    parser.add_argument("--actor_learning_rate", type=float, default=None)
    parser.add_argument("--critic_learning_rate", type=float, default=None)
    parser.add_argument("--target_update_period", type=int, default=None)
    parser.add_argument("--soft_target_update_rate", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--training_starts", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--use_entropy_bonus", type=str2bool, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--actor_gradient_type", type=str, default=None)
    parser.add_argument("--num_actor_samples", type=int, default=None)
    parser.add_argument("--num_critic_updates", type=int, default=None)
    parser.add_argument("--num_critic_networks", type=int, default=None)
    parser.add_argument("--target_critic_backup_type", type=str, default=None)
    parser.add_argument("--discount", type=float, default=None)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    # Build config overrides from command line arguments
    config_overrides = {}

    # Debug: Print parsed value
    print(
        f"DEBUG: args.use_entropy_bonus = {args.use_entropy_bonus} (type: {type(args.use_entropy_bonus)})"
    )

    if args.actor_learning_rate is not None:
        config_overrides["actor_learning_rate"] = args.actor_learning_rate
    if args.critic_learning_rate is not None:
        config_overrides["critic_learning_rate"] = args.critic_learning_rate
    if args.target_update_period is not None:
        config_overrides["target_update_period"] = args.target_update_period
    if args.soft_target_update_rate is not None:
        config_overrides["soft_target_update_rate"] = args.soft_target_update_rate
        print(f"DEBUG: use soft_target_update_rate {args.soft_target_update_rate}")
    if args.total_steps is not None:
        config_overrides["total_steps"] = args.total_steps
    if args.training_starts is not None:
        config_overrides["training_starts"] = args.training_starts
    if args.batch_size is not None:
        config_overrides["batch_size"] = args.batch_size
    if args.use_entropy_bonus is not None:
        print(f"DEBUG: Adding use_entropy_bonus={args.use_entropy_bonus} to overrides")
        config_overrides["use_entropy_bonus"] = args.use_entropy_bonus
    if args.temperature is not None:
        config_overrides["temperature"] = args.temperature
    if args.actor_gradient_type is not None:
        config_overrides["actor_gradient_type"] = args.actor_gradient_type
    if args.num_actor_samples is not None:
        config_overrides["num_actor_samples"] = args.num_actor_samples
    if args.num_critic_updates is not None:
        config_overrides["num_critic_updates"] = args.num_critic_updates
    if args.num_critic_networks is not None:
        config_overrides["num_critic_networks"] = args.num_critic_networks
    if args.target_critic_backup_type is not None:
        config_overrides["target_critic_backup_type"] = args.target_critic_backup_type
    if args.discount is not None:
        config_overrides["discount"] = args.discount

    print(f"DEBUG: config_overrides = {config_overrides}")

    config = make_config(args.config_file, **config_overrides)

    # Debug: Print use_entropy_bonus value
    print(f"DEBUG: use_entropy_bonus = {config['agent_kwargs']['use_entropy_bonus']}")
    print(
        f"DEBUG: soft_target_update_rate = {config['agent_kwargs']['soft_target_update_rate']}"
    )
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
