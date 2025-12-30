#!/bin/bash

# Average TensorBoard events across seeds for Section 6
# Run this AFTER section6_tuning.sh completes

echo "========================================="
echo "Averaging Section 6 Results Across Seeds"
echo "========================================="
echo ""

# Average baseline results (5 seeds)
echo "[1/2] Averaging BASELINE results..."
python average_seeds.py \
    --data_dir data/section6 \
    --exp_prefix "pendulum_default_s" \
    --num_seeds 5 \
    --output_name "pendulum_default_avg" \
    --tags "Eval_AverageReturn" "Train_AverageReturn" "Train_EnvstepsSoFar" "Baseline_Loss"

echo ""

# Average tuned results (5 seeds)
echo "[2/2] Averaging TUNED results..."
python average_seeds.py \
    --exp_prefix "pendulum_tuned_s" \
    --num_seeds 5 \
    --output_name "pendulum_tuned_avg" \
    --tags "Eval_AverageReturn" "Train_AverageReturn" "Train_EnvstepsSoFar" "Baseline_Loss"