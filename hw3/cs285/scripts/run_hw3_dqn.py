import argparse
import os
import time

import cs285.env_configs
import gym
import numpy as np
import torch
import tqdm
from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import (MemoryEfficientReplayBuffer,
                                                ReplayBuffer)
from gym import wrappers
from scripting_utils import make_config, make_logger

MAX_NVIDEO = 2

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps

    observation = None

    # Replay buffer
    if len(env.observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(frame_history_len=frame_history_len)
    elif len(env.observation_space.shape) == 1:
        stacked_frames = False
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    """
    Resets at episode boundaries and stores first frame.
    """
    reset_env_training()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        epsilon = exploration_schedule.value(step)

        # (4) TODO(student): Compute action
        action = agent.get_action(observation, epsilon)

        # (4) TODO(student): Step the environment
        next_observation, reward, done, info = env.step(action)

        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)
        done = False if truncated else done

        # (4) TODO(student): Add the data to the replay buffer
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            """
            We have observation, e.g. with shape (4, 84, 84), representing 4 stacked frames,
            so we only need to take the last and the newest frame to the replay buffer.
            """
            replay_buffer.insert(
                action=action,
                reward=reward,
                next_observation=next_observation[-1, ...],
                done=done,
            )
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )

        # Handle episode termination
        if done:
            reset_env_training()

            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation

        # Main DQN training loop
        if step >= config["learning_starts"]:
            # (5) TODO(student): Sample config["batch_size"] samples from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])

            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch)

            # (5) TODO(student): Train the agent. `batch` is a dictionary of numpy arrays,
            update_info = agent.update(
                obs=batch["observations"],
                action=batch["actions"],
                reward=batch["rewards"],
                next_obs=batch["next_observations"],
                done=batch["dones"],
                step=step,
            )

            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
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

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    # Allow overriding config parameters
    parser.add_argument("--learning_rate", "-lr", type=float, default=None)
    parser.add_argument("--target_update_period", type=int, default=None)
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--learning_starts", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder

    # Build config overrides from command line arguments
    config_overrides = {}
    if args.learning_rate is not None:
        config_overrides["learning_rate"] = args.learning_rate
    if args.target_update_period is not None:
        config_overrides["target_update_period"] = args.target_update_period
    if args.clip_grad_norm is not None:
        config_overrides["clip_grad_norm"] = args.clip_grad_norm
    if args.total_steps is not None:
        config_overrides["total_steps"] = args.total_steps
    if args.learning_starts is not None:
        config_overrides["learning_starts"] = args.learning_starts
    if args.batch_size is not None:
        config_overrides["batch_size"] = args.batch_size

    config = make_config(args.config_file, **config_overrides)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
