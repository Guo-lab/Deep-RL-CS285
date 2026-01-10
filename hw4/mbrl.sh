#!/bin/bash

# Check if experiment number is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash mbrl.sh <experiment_number> [additional_args...]"
    echo "Example: bash mbrl.sh 9 -nvid 1"
    echo ""
    echo "Available experiments:"
    echo "  1   - Problem 1: Training Dynamics Model (default config)"
    echo "  1.1 - Problem 1: Training with all hyperparameter variations"
    echo "  2   - Problem 2: MPC Action Selection"
    echo "  3   - Problem 3: Multi-iteration training (3 environments)"
    echo "  4   - Problem 4: Hyperparameter ablations (10 runs)"
    echo "  5   - Problem 5: CEM vs Random Shooting (3 runs)"
    echo "  6   - Problem 6: MBPO - Model-Based Policy Optimization (3 runs)"
    exit 1
fi

EXPERIMENT=$1
# Capture any additional arguments to pass to Python script
EXTRA_ARGS="${@:2}"

# Execute different trials based on experiment number
case $EXPERIMENT in
    1)
        echo "=========================================================="
        echo ""
        echo "Problem 1: Training Dynamics Model"
        echo "Testing on HalfCheetah with 0 iterations. TODO (1)-(5)."
        echo ""
        echo "This trains the dynamics model only (no policy evaluation)"
        echo "Expected: Loss should go below 0.2 by iteration 500"
        echo ""
        echo "----------------------------------------------"

        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_0_iter.yaml
        ;;

    1.1)
        echo "=========================================="
        echo "Testing on HalfCheetah with different hyperparameters"
        echo "------------------------------------------"
        echo ""

        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_0_iter_layers2.yaml
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_0_iter_layers3.yaml
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_0_iter_layers4.yaml
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_0_iter_hidden64.yaml
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_0_iter_hidden128.yaml

        echo ""
        echo "All 6 runs completed! Compare the loss curves to see effects of:"
        echo "  - Number of layers (1, 2, 3, 4)"
        echo "  - Hidden size (32, 64, 128)"
        ;;

    2)
        echo "=========================================================="
        echo ""
        echo "Problem 2: MPC Action Selection"
        echo "Testing MPC with random shooting on Obstacles environment. TODO (6)"
        echo ""
        echo "This verifies that MPC action selection works correctly"
        echo "Expected: eval_return should be greater than -70"
        echo ""
        echo "----------------------------------------------"
        echo ""

        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/obstacles_1_iter.yaml

        ;;

    3)
        echo "=========================================================="
        echo ""
        echo "Problem 3: On-Policy Data Collection & Iterative Training"
        echo "Running multi-iteration MBRL on 3 environments. No new code."
        echo ""
        echo "Expected rewards:"
        echo "  - Obstacles: -25 to -20"
        echo "  - Reacher: -300 to -250"
        echo "  - HalfCheetah: 250-350"
        echo ""
        echo "----------------------------------------------"
        echo ""

        echo "Running Obstacles multi-iteration..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/obstacles_multi_iter.yaml

        echo ""
        echo "Running Reacher multi-iteration..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_multi_iter.yaml

        echo ""
        echo "Running HalfCheetah multi-iteration..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_multi_iter.yaml

        ;;

    4)
        echo "=========================================================="
        echo ""
        echo "Problem 4: Hyperparameter Ablations"
        echo "Testing effect of ensemble size, action sequences, and horizon.  No new code."
        echo ""
        echo "Default values: ensemble=3, actions=1000, horizon=10"
        echo ""
        echo "----------------------------------------------"
        echo ""

        echo "1/10: Baseline (default values)..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation.yaml

        echo ""
        echo "2/10: Ensemble size = 1..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_ensemble1.yaml

        echo ""
        echo "3/10: Ensemble size = 2..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_ensemble2.yaml

        echo ""
        echo "4/10: Ensemble size = 5..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_ensemble5.yaml

        echo ""
        echo "5/10: Action sequences = 500..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_actions500.yaml

        echo ""
        echo "6/10: Action sequences = 1500..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_actions1500.yaml

        echo ""
        echo "7/10: Action sequences = 2000..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_actions2000.yaml

        echo ""
        echo "8/10: Horizon = 5..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_horizon5.yaml

        echo ""
        echo "9/10: Horizon = 15..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_horizon15.yaml

        echo ""
        echo "10/10: Horizon = 20..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/reacher_ablation_horizon20.yaml

        echo ""
        echo "Problem 4 completed! All 10 ablation experiments finished."
        echo "Ensemble sizes tested: 1, 2, 3, 5"
        echo "Action sequences tested: 500, 1000, 1500, 2000"
        echo "Horizons tested: 5, 10, 15, 20"
        ;;

    5)
        echo "=========================================================="
        echo ""
        echo "Problem 5: CEM vs Random Shooting"
        echo "Comparing action selection methods on HalfCheetah. TODO (7)"
        echo ""
        echo "Expected: CEM should reach ~800+ reward"
        echo ""
        echo "----------------------------------------------"
        echo ""

        echo "1/3: Random Shooting baseline (5 iterations)..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_random5iter.yaml

        echo ""
        echo "2/3: CEM with 2 iterations..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_cem2.yaml

        echo ""
        echo "3/3: CEM with 4 iterations..."
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_cem.yaml

        echo ""
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_cem3.yaml

        ;;

    6)
        echo "=========================================================="
        echo ""
        echo "Problem 6: MBPO - Model-Based Policy Optimization"
        echo "Comparing SAC with different amounts of synthetic data. TODO (8)"
        echo ""
        echo "Three configurations:"
        echo "  1. Model-free SAC (rollout_length=0)"
        echo "  2. Dyna-style (rollout_length=1)"
        echo "  3. Full MBPO (rollout_length=10)"
        echo ""
        echo "Expected: Full MBPO should reach ~700+ reward"
        echo ""
        echo "----------------------------------------------"
        echo ""

        YAML_FILE="experiments/sac/halfcheetah_clipq.yaml"

        echo "1/3: Model-free SAC baseline (rollout_length=0)..."
        sed -i 's/mbpo_rollout_length: .*/mbpo_rollout_length: 0/' $YAML_FILE
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_mbpo.yaml \
            --sac_config_file $YAML_FILE

        echo ""
        echo "2/3: Dyna-style (rollout_length=1)..."
        sed -i 's/mbpo_rollout_length: .*/mbpo_rollout_length: 1/' $YAML_FILE
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_mbpo.yaml \
            --sac_config_file $YAML_FILE

        echo ""
        echo "3/3: Full MBPO (rollout_length=10)..."
        sed -i 's/mbpo_rollout_length: .*/mbpo_rollout_length: 10/' $YAML_FILE
        python cs285/scripts/run_hw4.py \
            -cfg experiments/mpc/halfcheetah_mbpo.yaml \
            --sac_config_file $YAML_FILE

        echo ""
        echo "Rollout lengths tested: 0 (model-free), 1 (Dyna), 10 (MBPO)"

        # Restore to default
        sed -i 's/mbpo_rollout_length: .*/mbpo_rollout_length: 0/' $YAML_FILE
        ;;


    dead)
        echo "=========================================="
        echo ""
        if [ -n "$EXTRA_ARGS" ]; then
            echo "Additional arguments: $EXTRA_ARGS"
        fi

        echo "Running "
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/humanoid.yaml \
            $EXTRA_ARGS

        ;;

    *)
        echo "Error: Invalid experiment number '$EXPERIMENT'"
        echo "Available experiments: 1, 1.1, 2, 3, 4, 5, 6"
        exit 1
        ;;
esac

echo ""
echo "Experiment $EXPERIMENT completed!"


# So now you should see hw4.pdf in the root folder, and please     
#   help me complete the code in codebase cs285. I do  
#   not need answers, just give me hints to guide me walk through 
#   step by step.                                                
#                 
#   I would have questions during the process.  
#   Think before doing anything and tell me your plan first. 