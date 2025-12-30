#!/bin/bash

# Section 4: Using a Neural Network Baseline
# Experiment 2: HalfCheetah-v4

# No baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--exp_name cheetah --which_gpu 0

# Baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 5 \
--exp_name cheetah_baseline --which_gpu 0

# Baseline with video logging and -na flag
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 5 \
--exp_name cheetah_baseline_na --which_gpu 0 \
--video_log_freq 10 -na





# Section 4: Grid search over baseline gradient steps (bgs)
# Keep baseline learning rate fixed at 0.01
# Vary bgs to see effect on baseline learning and policy performance

# Baseline gradient steps = 1 (very low)
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 1 --exp_name cheetah_baseline_bgs1

# Baseline gradient steps = 2
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 2 --exp_name cheetah_baseline_bgs2

# Baseline gradient steps = 3
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 3 --exp_name cheetah_baseline_bgs3

# Baseline gradient steps = 5 (default - already run)
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

# # Baseline gradient steps = 10 (increased for comparison)
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --use_baseline -blr 0.01 -bgs 10 --exp_name cheetah_baseline_bgs10







# Section 4: Grid search over baseline learning rate (blr)
# Keep baseline gradient steps fixed at 5
# Vary blr to see effect on baseline learning and policy performance

# Baseline learning rate = 0.001 (very low)
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.001 -bgs 5 --exp_name cheetah_baseline_blr0.001

# Baseline learning rate = 0.005
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.005 -bgs 5 --exp_name cheetah_baseline_blr0.005

# Baseline learning rate = 0.01 (default - already run)
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

# # Baseline learning rate = 0.02 (increased for comparison)
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --use_baseline -blr 0.02 -bgs 5 --exp_name cheetah_baseline_blr0.02

# # Baseline learning rate = 0.05 (very high)
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
# -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
# --use_baseline -blr 0.05 -bgs 5 --exp_name cheetah_baseline_blr0.05
