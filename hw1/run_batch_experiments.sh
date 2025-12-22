#!/bin/bash
# Simple batch experiment runner for CS285 HW1

set -e

# Environments to test
ENVS=("Ant-v4" "Walker2d-v4" "HalfCheetah-v4" "Hopper-v4")

# Experiment parameters
EVAL_BATCH_SIZES=(500 2000 5000 10000 20000)
TRAIN_STEPS=(100 200 500 1000 2000 4000)

echo "Starting batch experiments..."
echo ""

# Run experiments
for env in "${ENVS[@]}"; do
    echo "=== Running experiments for $env ==="

    # Get short name (ant, walker2d, etc.)
    short_name=$(echo $env | cut -d'-' -f1 | tr '[:upper:]' '[:lower:]')
    expert_policy="cs285/policies/experts/${env%-v4}.pkl"
    expert_data="cs285/expert_data/expert_data_${env}.pkl"

    # Experiment 1: Sweep eval batch sizes
    echo "Sweeping eval_batch_size..."
    for eval_bs in "${EVAL_BATCH_SIZES[@]}"; do
        python cs285/scripts/run_hw1.py \
            --expert_policy_file $expert_policy \
            --env_name $env \
            --exp_name "bc_${short_name}_evalbs${eval_bs}" \
            --n_iter 1 \
            --ep_len 1000 \
            --eval_batch_size $eval_bs \
            --num_agent_train_steps_per_iter 1000 \
            --expert_data $expert_data \
            --video_log_freq 5
    done

    # Experiment 2: Sweep training steps
    echo "Sweeping num_agent_train_steps_per_iter..."
    for train_steps in "${TRAIN_STEPS[@]}"; do
        python cs285/scripts/run_hw1.py \
            --expert_policy_file $expert_policy \
            --env_name $env \
            --exp_name "bc_${short_name}_trainsteps${train_steps}" \
            --n_iter 1 \
            --ep_len 1000 \
            --eval_batch_size 5000 \
            --num_agent_train_steps_per_iter $train_steps \
            --expert_data $expert_data \
            --video_log_freq 5
    done

    echo ""
done

echo "All experiments completed!"
echo "Run: python analyze_results.py"
