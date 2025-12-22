#!/bin/bash
# DAgger experiments for CS285 HW1 Problem 2
# Runs DAgger on multiple environments with multiple seeds

set -e

# Select environments to test (modify as needed based on Problem 1)
ENVS=("Ant-v4" "Hopper-v4" "HalfCheetah-v4" "Walker2d-v4")

# DAgger parameters
N_ITER=10  # Number of DAgger iterations
BATCH_SIZE=1000  # Data collected per iteration
EVAL_BATCH_SIZE=5000  # Evaluation rollouts
TRAIN_STEPS=1000  # Training steps per iteration
N_SEEDS=3  # Number of random seeds for error bars

echo "Starting DAgger experiments for Problem 2..."
echo "Environments: ${ENVS[@]}"
echo "DAgger iterations: $N_ITER"
echo "Seeds per environment: $N_SEEDS"
echo ""

# Run DAgger experiments
for env in "${ENVS[@]}"; do
    echo "=== Running DAgger experiments for $env ==="

    # Get environment-specific paths
    short_name=$(echo $env | cut -d'-' -f1 | tr '[:upper:]' '[:lower:]')
    expert_policy="cs285/policies/experts/${env%-v4}.pkl"
    expert_data="cs285/expert_data/expert_data_${env}.pkl"

    # Run multiple seeds
    for seed in $(seq 1 $N_SEEDS); do
        echo "Running DAgger with seed $seed..."
        python cs285/scripts/run_hw1.py \
            --expert_policy_file $expert_policy \
            --env_name $env \
            --exp_name "dagger_${short_name}_seed${seed}" \
            --do_dagger \
            --n_iter $N_ITER \
            --batch_size $BATCH_SIZE \
            --eval_batch_size $EVAL_BATCH_SIZE \
            --ep_len 1000 \
            --num_agent_train_steps_per_iter $TRAIN_STEPS \
            --expert_data $expert_data \
            --seed $seed \
            --video_log_freq -1 \
            --scalar_log_freq 1
    done

    echo ""
done

echo "DAgger experiments completed!"
echo ""
echo "Next steps:"
echo "1. Run: python analyze_dagger_results.py"
echo "2. This will generate learning curve plots comparing DAgger, BC, and Expert performance"
