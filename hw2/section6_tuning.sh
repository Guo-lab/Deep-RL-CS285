#!/bin/bash

# Section 6: Hyperparameter Tuning for InvertedPendulum-v4
# Goal: Achieve high return (1000) in as few environment steps as possible

set -e  # Exit on error

# GPU to use
GPU_ID=0

echo "========================================="
echo "Section 6: Hyperparameter Tuning"
echo "========================================="

# ============================================
# BASELINE: Default settings (5 seeds)
# ============================================
echo ""
echo "Running BASELINE experiments (default settings)..."
for seed in $(seq 1 3); do
    echo "  Baseline - Seed $seed/5"
    python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
        -n 70 \
        --exp_name pendulum_default_s$seed \
        -rtg --use_baseline -na \
        --batch_size 5000 \
        --seed $seed \
        --which_gpu $GPU_ID
done

# ============================================
# TUNED: Optimized hyperparameters (5 seeds)
# ============================================
echo ""
echo "Running TUNED experiments (optimized hyperparameters)..."
for seed in $(seq 1 3); do
    echo "  Tuned - Seed $seed/5"
    python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
        -n 70 \
        --exp_name pendulum_tuned_s$seed \
        -rtg --use_baseline -na \
        --batch_size 1000 \
        -lr 0.03 \
        --gae_lambda 0.98 \
        -l 2 -s 64 \
        -blr 0.02 -bgs 10 \
        --seed $seed \
        --which_gpu $GPU_ID
done

for seed in $(seq 1 3); do
    python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
        -n 70 \
        --exp_name pendulum_tuned_no_na_s$seed \
        -rtg --use_baseline \
        --batch_size 1000 \
        -lr 0.03 \
        --gae_lambda 0.98 \
        -l 2 -s 64 \
        -blr 0.02 -bgs 10 \
        --seed 1 \
        --which_gpu $GPU_ID
done

# ============================================
# OPTIONAL: Grid search (batch_size x lr)
# Uncomment to run systematic hyperparameter search
# ============================================
echo ""
echo "OPTIONAL: Grid search disabled by default"
echo "Uncomment grid search section below to explore batch_size x learning_rate"

# Grid search parameters
BATCH_SIZES=(500 1000 2000)
LEARNING_RATES=(0.02 0.03 0.05)

for bs in "${BATCH_SIZES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        echo "  Grid search: batch=$bs, lr=$lr (seed 1 only)"
        python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
            -n 70 \
            --exp_name pendulum_grid_b${bs}_lr${lr} \
            -rtg --use_baseline -na \
            --batch_size $bs \
            -lr $lr \
            --gae_lambda 0.98 \
            -l 2 -s 64 \
            -blr 0.02 -bgs 10 \
            --seed 1 \
            --which_gpu $GPU_ID
    done
done
