#!/usr/bin/env python3
"""
Script to plot train_return and eval_return on the same axes with smoothing.
Usage:
python plot_train_vs_eval.py ./data/hw3_dqn_dqn_MsPacmanNoFrameskip-v0_d0.99_tu2000_lr0.0001_doubleq_clip10.0_05-01-2026_23-22-34/
    --output mspacman_train_eval.png --smooth 0.6
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def find_subdirs(parent_dir):
    """Find all subdirectories with event files."""
    return [
        root
        for root, dirs, files in os.walk(parent_dir)
        if any(f.startswith("events.out.tfevents") for f in files)
    ]


def read_metric(logdir, metric):
    """Read a metric from tensorboard."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    if metric not in ea.scalars.Keys():
        return None, None
    events = ea.scalars.Items(metric)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    if weight == 0:
        return values
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def average_runs(dirs, metric):
    """Average a metric across multiple runs."""
    data_list = [read_metric(d, metric) for d in dirs]
    data_list = [(s, v) for s, v in data_list if s is not None]

    if not data_list:
        return None, None, None

    common_steps = sorted(set.intersection(*[set(s) for s, _ in data_list]))
    aligned = [
        [dict(zip(steps, values))[s] for s in common_steps]
        for steps, values in data_list
    ]

    return np.array(common_steps), np.mean(aligned, axis=0), np.std(aligned, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="Directory containing runs")
    parser.add_argument("--output", default="train_vs_eval.png", help="Output file")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.9,
        help="Smoothing weight (0=no smoothing, 0.9=default)",
    )
    args = parser.parse_args()

    # Find runs
    dirs = find_subdirs(args.logdir)
    print(f"Found {len(dirs)} runs in {args.logdir}")

    # Average train and eval
    steps_train, mean_train, std_train = average_runs(dirs, "train_return")
    steps_eval, mean_eval, std_eval = average_runs(dirs, "eval_return")

    # Apply smoothing
    if steps_train is not None:
        mean_train = smooth(mean_train, args.smooth)
        std_train = smooth(std_train, args.smooth)
    if steps_eval is not None:
        mean_eval = smooth(mean_eval, args.smooth)
        std_eval = smooth(std_eval, args.smooth)

    # Plot
    plt.figure(figsize=(10, 6))

    if steps_train is not None:
        plt.plot(
            steps_train,
            mean_train,
            label="Train Return",
            color="orange",
            linewidth=2,
            alpha=0.8,
        )
        plt.fill_between(
            steps_train,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.15,
            color="orange",
        )

    if steps_eval is not None:
        plt.plot(
            steps_eval,
            mean_eval,
            label="Eval Return",
            color="green",
            linewidth=3.5,
            alpha=0.9,
        )
        plt.fill_between(
            steps_eval,
            mean_eval - std_eval,
            mean_eval + std_eval,
            alpha=0.2,
            color="green",
        )

    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Return", fontsize=12)
    plt.title(f"Training vs Evaluation Return (smoothing={args.smooth})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.output}")
