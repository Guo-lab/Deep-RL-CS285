#!/usr/bin/env python
"""
Average TensorBoard event files across multiple seeds and write back to TensorBoard format.

Usage:
    python average_seeds.py --exp_prefix pendulum_default_s --num_seeds 5 --output_name pendulum_default_avg
    python average_seeds.py --exp_prefix pendulum_tuned_s --num_seeds 5 --output_name pendulum_tuned_avg
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


def find_event_files(data_dir, exp_prefix, num_seeds):
    """Find all event files for the given experiment prefix and seeds."""
    event_files = []

    for seed in range(1, num_seeds + 1):
        # Pattern: data/q2_pg_{exp_prefix}{seed}_*
        pattern = os.path.join(data_dir, f"q2_pg_{exp_prefix}{seed}_*")
        matching_dirs = glob.glob(pattern)

        if not matching_dirs:
            print(f"Warning: No directory found for seed {seed} with pattern {pattern}")
            continue

        # Find event file in the directory
        exp_dir = matching_dirs[0]
        events = glob.glob(os.path.join(exp_dir, "events.out.tfevents.*"))

        if events:
            event_files.append(events[0])
            print(f"Found seed {seed}: {events[0]}")
        else:
            print(f"Warning: No event file found in {exp_dir}")

    return event_files


def load_scalar_data(event_file, tags):
    """Load scalar data for specified tags from an event file."""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    data = {}
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            # Store as (step, value) pairs
            data[tag] = [(e.step, e.value) for e in events]
        else:
            print(f"Warning: Tag '{tag}' not found in {event_file}")
            data[tag] = []

    return data


def average_across_seeds(event_files, tags):
    """Average scalar data across multiple seed runs."""
    all_data = defaultdict(list)

    # Load data from all seeds
    for event_file in event_files:
        seed_data = load_scalar_data(event_file, tags)
        for tag, values in seed_data.items():
            all_data[tag].append(values)

    # Average across seeds
    averaged_data = {}

    for tag in tags:
        if not all_data[tag]:
            print(f"Warning: No data for tag '{tag}'")
            continue

        # Find common steps across all seeds
        # Use the first seed as reference for steps
        reference_steps = [step for step, _ in all_data[tag][0]]

        averaged_values = []
        for step in reference_steps:
            # Collect values at this step from all seeds
            values_at_step = []
            for seed_data in all_data[tag]:
                # Find value at this step
                for s, v in seed_data:
                    if s == step:
                        values_at_step.append(v)
                        break

            if values_at_step:
                avg_value = np.mean(values_at_step)
                std_value = np.std(values_at_step)
                averaged_values.append((step, avg_value, std_value))

        averaged_data[tag] = averaged_values

    return averaged_data


def write_averaged_data(averaged_data, output_dir, output_name):
    """Write averaged data to TensorBoard event file."""
    log_dir = os.path.join(output_dir, f"q2_pg_{output_name}")
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    for tag, values in averaged_data.items():
        for step, avg_value, std_value in values:
            # Write mean
            writer.add_scalar(tag, avg_value, step)
            # Also write std as separate metric
            writer.add_scalar(f"{tag}_std", std_value, step)

    writer.close()
    print(f"\nAveraged data written to: {log_dir}")
    print(f"You can view it in TensorBoard alongside individual runs!")


def main():
    parser = argparse.ArgumentParser(
        description="Average TensorBoard events across seeds"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/section6",
        help="Directory containing experiment data",
    )
    parser.add_argument(
        "--exp_prefix",
        type=str,
        required=True,
        help='Experiment name prefix (e.g., "pendulum_default_s")',
    )
    parser.add_argument(
        "--num_seeds", type=int, default=5, help="Number of seeds to average"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help='Name for averaged output (e.g., "pendulum_default_avg")',
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["Eval_AverageReturn", "Train_AverageReturn"],
        help="Metrics to average",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Averaging TensorBoard Events Across Seeds")
    print("=" * 60)
    print(f"Experiment prefix: {args.exp_prefix}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Output name: {args.output_name}")
    print(f"Metrics to average: {args.tags}")
    print()

    # Find event files
    event_files = find_event_files(args.data_dir, args.exp_prefix, args.num_seeds)

    if len(event_files) < args.num_seeds:
        print(f"\nWarning: Found only {len(event_files)} out of {args.num_seeds} seeds")
        if len(event_files) == 0:
            print("No event files found. Exiting.")
            return

    # Average data
    print(f"\nAveraging across {len(event_files)} seeds...")
    averaged_data = average_across_seeds(event_files, args.tags)

    # Write back to TensorBoard
    write_averaged_data(averaged_data, args.data_dir, args.output_name)

    print("\nDone! ✓")
    print(f"\nTo view in TensorBoard:")
    print(f"  tensorboard --logdir {args.data_dir}")


if __name__ == "__main__":
    main()
