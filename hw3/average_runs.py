#!/usr/bin/env python3
"""
Simple script to average TensorBoard runs.
Usage: python average_runs.py data/my_experiment data/my_experiment_averaged
"""

import sys
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


def read_scalars(logdir):
    """Read all metrics from a run."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    result = {}
    for tag in ea.scalars.Keys():
        events = ea.scalars.Items(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        result[tag] = (steps, values)
    return result


def average_data(data_list):
    """Average data across runs."""
    all_steps = [steps for steps, _ in data_list]
    common_steps = sorted(set.intersection(*[set(s) for s in all_steps]))

    aligned = []
    for steps, values in data_list:
        step_to_value = dict(zip(steps, values))
        aligned.append([step_to_value[s] for s in common_steps])

    mean_values = np.mean(aligned, axis=0)
    return np.array(common_steps), mean_values


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python average_runs.py <input_dir> <output_dir>")
        print("Example: python average_runs.py data/lunarlander data/lunarlander_avg")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Find all subdirectories with event files
    subdirs = []
    for root, dirs, files in os.walk(input_dir):
        if any(f.startswith('events.out.tfevents') for f in files):
            subdirs.append(root)

    print(f"Found {len(subdirs)} runs")

    # Read all data
    all_data = []
    for subdir in subdirs:
        all_data.append(read_scalars(subdir))
        print(f"  Loaded: {subdir}")

    # Get all metrics
    all_tags = set()
    for data in all_data:
        all_tags.update(data.keys())

    # Average and write
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    for tag in all_tags:
        tag_data = [data[tag] for data in all_data if tag in data]
        steps, values = average_data(tag_data)

        for step, value in zip(steps, values):
            writer.add_scalar(tag, value, step)

        print(f"Averaged: {tag}")

    writer.close()
    print(f"\nDone! Saved to: {output_dir}")
