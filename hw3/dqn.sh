#!/bin/bash

# Check if experiment number is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash scripts.sh <experiment_number>"
    echo "Available experiments:"
    echo "  1 - CartPole DQN with different learning rates (0.05 and 0.001)"
    echo "  2 - LunarLander DQN with multiple seeds (1, 2, 3)"
    echo "  3 - LunarLander Double-Q with multiple seeds (1, 2, 3)"
    echo "  4 - MsPacman Atari game with multiple seeds (1, 2, 3)"
    exit 1
fi

EXPERIMENT=$1

# Execute different trials based on experiment number
case $EXPERIMENT in
    1)
        echo "Running Experiment 1: CartPole DQN with different learning rates..."
        echo "================================"
        echo "Trial 1a: Learning rate 0.05"
        echo "--------------------------------"
        python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml --learning_rate 0.05

        echo ""
        echo "================================"
        echo "Trial 1b: Learning rate 0.001"
        echo "--------------------------------"
        python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml --learning_rate 0.001
        ;;

    2)
        echo "Running Experiment 2: LunarLander DQN with multiple seeds..."
        seeds=(1 2 3)
        for seed in "${seeds[@]}"; do
            python cs285/scripts/run_hw3_dqn.py \
                -cfg experiments/dqn/lunarlander.yaml \
                --seed "$seed" \
                --clip_grad_norm 10.0
            sleep 2
        done
        ;;

    3)
        echo "Running Experiment 3: LunarLander Double-Q with multiple seeds..."
        seeds=(1 2 3)
        for seed in "${seeds[@]}"; do
            python cs285/scripts/run_hw3_dqn.py \
                -cfg experiments/dqn/lunarlander_doubleq.yaml \
                --seed "$seed" \
                --clip_grad_norm 10.0
            sleep 2
        done
        ;;
    
        # For tiny networks, (CartPole / LunarLander)
        # Decreasing the target_update_period causes the target network to be updated more frequently. 
        # As a result, the Q-targets used to compute the temporal-difference (TD) errors are fresher 
        # and less stale. 
        # This leads to smaller and more stable TD errors, which in turn produce gradients with smaller 
        # magnitude during backpropagation. 
        # Since gradient clipping is applied in DQN, smaller gradients reduce the computational work 
        # required for clipping and backpropagation. 
        # Consequently, the wall-clock time per training step decreases. 
        # In short, more frequent target updates indirectly speed up each step by producing more stable 
        # and smaller gradients.


    4)
        echo "Running Experiment 4: MsPacman Atari game..."
        python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml --seed 1
        python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml --seed 2
        python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml --seed 3
        ;;

    *)
        echo "Error: Invalid experiment number '$EXPERIMENT'"
        echo "Available experiments: 1, 2, 3, 4"
        exit 1
        ;;
esac

echo "Experiment $EXPERIMENT completed!"
