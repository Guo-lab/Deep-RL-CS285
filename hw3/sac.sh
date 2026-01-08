#!/bin/bash

# Check if experiment number is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash sac.sh <experiment_number> [additional_args...]"
    echo "Example: bash sac.sh 9 -nvid 1"
    echo ""
    echo "Available experiments:"
    echo "  1 - Section 2.1.1: Bootstrapping (Pendulum-v1)"
    echo "  2 - Section 2.1.2: Entropy Bonus (Pendulum-v1)"
    echo "  3 - Section 2.1.3: REINFORCE Testing (InvertedPendulum-v4)"
    echo "  4 - Section 2.1.3: REINFORCE Deliverables (HalfCheetah-v4)"
    echo "  5 - Section 2.1.4: REPARAMETRIZE Testing (InvertedPendulum-v4)"
    echo "  6 - Section 2.1.4: REPARAMETRIZE Deliverable (HalfCheetah-v4)"
    echo "  7 - Section 2.1.4: REPARAMETRIZE Deliverable (Humanoid-v4)"
    echo "  8 - Section 2.1.5: Stabilizing Target Values (Hopper-v4)"
    echo "  9 - Section 2.1.5: Stabilizing Target Values (Humanoid-v4)"
    exit 1
fi

EXPERIMENT=$1
# Capture any additional arguments to pass to Python script
EXTRA_ARGS="${@:2}"

# Execute different trials based on experiment number
case $EXPERIMENT in
    1)
        echo "=========================================="
        echo "Section 2.1.1: Bootstrapping"
        echo "Testing critic updates on Pendulum-v1"
        echo "------------------------------------------"
        echo ""
        echo "This test verifies:"
        echo "  Critic network updates with target values"
        echo ""
        echo "Expected behavior:"
        echo "  Critic loss should decrease over time and Q-values should stabilize"
        echo ""
        python cs285/scripts/run_hw3_sac.py \
            --use_entropy_bonus False \
            --actor_gradient_type skip \
            --training_starts 5000 \
            -cfg experiments/sac/sanity_pendulum.yaml

        ;;

    2)
        echo "=========================================="
        echo "Section 2.1.2: Entropy Bonus and Soft Actor-Critic"
        echo "Testing entropy on Pendulum-v1"
        echo "------------------------------------------"
        echo ""
        python cs285/scripts/run_hw3_sac.py \
            --use_entropy_bonus True \
            --actor_gradient_type skip \
            --training_starts 10000 \
            -cfg experiments/sac/sanity_pendulum.yaml

        ;;

    3)
        echo "=========================================="
        echo "Section 2.1.3: Actor with REINFORCE - Testing"
        echo "Testing reward close to 1000 on InvertedPendulum-v4"
        echo "------------------------------------------"
        echo ""
        # Lower temperature to reduce exploration after learning
        # Lower actor LR for more stable updates
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/sanity_invertedpendulum_reinforce.yaml \
            --actor_learning_rate 0.0003 \
            --temperature 0.1 \
            --log_interval 10000

        ;;

    4)
        echo "=========================================="
        echo "Section 2.1.3: Actor with REINFORCE - Deliverables"
        echo "Training HalfCheetah-v4 with REINFORCE-1 and REINFORCE-10"
        echo "------------------------------------------"
        echo ""
        echo "Training with REINFORCE-1 (single action sample)..."
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/halfcheetah_reinforce1.yaml

        echo ""
        echo "Training with REINFORCE-10 (10 action samples)..."
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/halfcheetah_reinforce10.yaml

        ;;

    5)
        echo "=========================================="
        echo "Section 2.1.4: Actor with REPARAMETRIZE - Testing"
        echo "Testing reward close to 1000 on InvertedPendulum-v4"
        echo "------------------------------------------"
        echo ""
        echo "This test verifies:"
        echo "  REPARAMETRIZE gradient estimator works correctly"
        echo "  Should achieve similar reward to REINFORCE (~1000)"
        echo ""
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/sanity_invertedpendulum_reparametrize.yaml

        ;;

    6)
        echo "=========================================="
        echo "Section 2.1.4: Actor with REPARAMETRIZE - Deliverable"
        echo "Training HalfCheetah-v4 with REPARAMETRIZE"
        echo "------------------------------------------"
        echo ""
        echo "This will be compared with REINFORCE-1 and REINFORCE-10"
        echo "Expected: Lower variance, better sample efficiency than REINFORCE"
        echo ""
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/halfcheetah_reparametrize.yaml

        ;;

    7)
        echo "=========================================="
        echo "Section 2.1.4: Actor with REPARAMETRIZE - Deliverable"
        echo "Training Humanoid-v4 with SAC"
        echo "------------------------------------------"
        echo ""
        echo "NOTE: This is a long training run (5M steps)"
        echo "You can truncate after 500K steps for the assignment"
        echo ""
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/humanoid_sac.yaml

        ;;

    8)
        echo "=========================================="
        echo "Section 2.1.5: Stabilizing Target Values"
        echo "Training Hopper-v4 with Single-Q, Double-Q, and Clipped Double-Q"
        echo "------------------------------------------"
        echo ""
        echo "This will compare overestimation bias mitigation strategies"
        echo "Expected: Clipped Double-Q (min) should reduce overestimation"
        echo ""

        echo "Running Single-Q (1 critic network)..."
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/hopper.yaml

        echo ""
        echo "Running Double-Q (2 critics with rotation)..."
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/hopper_doubleq.yaml

        echo ""
        echo "Running Clipped Double-Q (2 critics with min)..."
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/hopper_clipq.yaml

        ;;

    9)
        echo "=========================================="
        echo "Section 2.1.5: Stabilizing Target Values"
        echo "Training Humanoid-v4 with the best configuration."
        echo "------------------------------------------"
        echo ""
        echo "NOTE: These are long training runs (5M steps each)"
        echo "Optional: Truncate after 500K steps for the assignment"
        echo ""
        if [ -n "$EXTRA_ARGS" ]; then
            echo "Additional arguments: $EXTRA_ARGS"
        fi

        echo "Running Clipped Double-Q (2 critics with min)..."
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/humanoid.yaml \
            $EXTRA_ARGS

        ;;

    10)
        ehco "REDQ: "
        echo ""
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/hopper_redq.yaml

        ;;

    *)
        echo "Error: Invalid experiment number '$EXPERIMENT'"
        echo "Available experiments: 1, 2, 3, 4, 5, 6, 7, 8, 9"
        exit 1
        ;;
esac

echo ""
echo "Experiment $EXPERIMENT completed!"
