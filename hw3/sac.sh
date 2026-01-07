#!/bin/bash

# Check if experiment number is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash sac.sh <experiment_number>"
    echo "Available experiments:"
    echo "  1 - Section 2.1.1: Bootstrapping (Pendulum-v1)"
    echo "  2 - Section 2.1.2: Entropy Bonus (Pendulum-v1)"
    echo "  3 - Section 2.1.3: REINFORCE Testing (InvertedPendulum-v4)"
    echo "  4 - Section 2.1.3: REINFORCE Deliverables (HalfCheetah-v4)"
    exit 1
fi

EXPERIMENT=$1

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
            -cfg experiments/sac/halfcheetah_reinforce1.yaml\
            --log_interval 10000

        echo ""
        echo "Training with REINFORCE-10 (10 action samples)..."
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/halfcheetah_reinforce10.yaml\
            --log_interval 10000

        ;;

    *)
        echo "Error: Invalid experiment number '$EXPERIMENT'"
        echo "Available experiments: 1, 2, 3, 4"
        exit 1
        ;;
esac

echo ""
echo "Experiment $EXPERIMENT completed!"
