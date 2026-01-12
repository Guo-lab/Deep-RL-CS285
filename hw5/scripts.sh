#!/bin/bash

# Check if experiment number is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash scripts.sh <experiment_number> [additional_args...]"
    echo "Example: bash scripts.sh 9 -nvid 1"
    echo ""
    echo "Available experiments:"
    echo "  1   - 3.1 Random Policy"
    echo "  2   - 3.2 Random Network Distillation"
    echo "  3   - 4.1 CQL vs DQN"
    echo "  4   - 4.1 CQL Alpha Sweep (α = 0, 0.1, 0.5, 1.0, 5.0, 10.0)"
    echo "  5   - 4.2 AWAC and IQL Comparison"
    echo "  6   - RND Exploration on Medium with different dataset sizes"
    echo "  7   - 4.3 Data Ablations - CQL on Medium with different dataset sizes"
    echo "  8   - 5.0 Online Fine-tuning on Hard with CQL, AWAC, DQN, and IQL"
    exit 1
fi

EXPERIMENT=$1
# Capture any additional arguments to pass to Python script
EXTRA_ARGS="${@:2}"

# Execute different trials based on experiment number
case $EXPERIMENT in
    1)
        echo "=========================================================="
        echo "3.1 Random Policy @ TODO (1)"
        echo "----------------------------------------------"

        python cs285/scripts/run_hw5_explore.py \
            -cfg experiments/exploration/pointmass_easy_random.yaml \
            --dataset_dir datasets/
        python cs285/scripts/run_hw5_explore.py \
            -cfg experiments/exploration/pointmass_medium_random.yaml \
            --dataset_dir datasets/
        python cs285/scripts/run_hw5_explore.py \
            -cfg experiments/exploration/pointmass_hard_random.yaml \
            --dataset_dir datasets/

        ;;

    2)
        echo "========================================================="
        echo "3.2 Random Network Distillation @ TODO (2) (3)"
        echo "------------------------------------------"
        echo ""
        python cs285/scripts/run_hw5_explore.py \
            -cfg experiments/exploration/pointmass_easy_rnd.yaml \
            --dataset_dir datasets/
        python cs285/scripts/run_hw5_explore.py \
            -cfg experiments/exploration/pointmass_medium_rnd.yaml \
            --dataset_dir datasets/
        python cs285/scripts/run_hw5_explore.py \
            -cfg experiments/exploration/pointmass_hard_rnd.yaml \
            --dataset_dir datasets/

        ;;

    3)
        echo "========================================================="
        echo "4.1 CQL vs DQN @ TODO (4)"
        echo "------------------------------------------"
        echo ""
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_easy_cql.yaml \
            --dataset_dir datasets
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_easy_dqn.yaml \
            --dataset_dir datasets
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_medium_cql.yaml \
            --dataset_dir datasets
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_medium_dqn.yaml \
            --dataset_dir datasets

        ;;

    4)
        echo "========================================================="
        echo "4.1 CQL Alpha Sweep (Medium environment)"
        echo "------------------------------------------"
        echo "Testing alpha values: 0, 0.1, 0.5, 1.0, 5.0, 10.0"
        echo ""

        CONFIG_FILE="experiments/offline/pointmass_medium_cql.yaml"
        BACKUP_FILE="experiments/offline/pointmass_medium_cql.yaml.backup"

        # Backup original config
        cp $CONFIG_FILE $BACKUP_FILE
        echo "Backed up original config to $BACKUP_FILE"
        echo ""

        for alpha in 0 0.1 0.5 1.0 5.0 10.0; do
            echo "------------------------------------------------------"
            echo "Running CQL with alpha=$alpha"
            echo "------------------------------------------------------"

            # Replace cql_alpha value in YAML
            sed -i "s/^cql_alpha: .*/cql_alpha: $alpha/" $CONFIG_FILE
            sed -i "s/^exp_name: .*/exp_name: \"pointmass_medium_cql_alpha${alpha}\"/" $CONFIG_FILE

            # Run experiment
            python ./cs285/scripts/run_hw5_offline.py \
                -cfg $CONFIG_FILE \
                --dataset_dir datasets

            echo ""
        done

        # Restore original config
        mv $BACKUP_FILE $CONFIG_FILE
        echo "Restored original config"

        ;;

    5)
        echo "========================================================="
        echo "4.2 AWAC and IQL Comparison @ TODO (5) - (7) and (8) - (11)"
        echo "------------------------------------------"
        echo "Running AWAC and IQL on available environments"
        echo ""

        # Run AWAC on Hard environment
        echo "Running AWAC on Hard environment..."
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_hard_awac.yaml \
            --dataset_dir datasets

        # Run IQL on Hard environment
        echo "Running IQL on Hard environment..."
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_hard_iql.yaml \
            --dataset_dir datasets

        echo "Running AWAC on Medium environment..."
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_medium_awac.yaml \
            --dataset_dir datasets

        echo "Running IQL on Medium environment..."
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_medium_iql.yaml \
            --dataset_dir datasets

        echo "Running AWAC on Easy environment..."
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_easy_awac.yaml \
            --dataset_dir datasets

        echo "Running IQL on Easy environment..."
        python ./cs285/scripts/run_hw5_offline.py \
            -cfg experiments/offline/pointmass_easy_iql.yaml \
            --dataset_dir datasets

        ;;

    6)
        echo "========================================================="
        echo "RND Exploration on Medium maze with different dataset sizes"
        echo "------------------------------------------"
        echo "Testing dataset sizes: 1000, 5000, 10000, 20000"
        echo ""

        python cs285/scripts/run_hw5_explore.py \
                -cfg experiments/exploration/pointmass_medium_rnd_step1000.yaml \
                --dataset_dir datasets/

        python cs285/scripts/run_hw5_explore.py \
                -cfg experiments/exploration/pointmass_medium_rnd_step5000.yaml \
                --dataset_dir datasets/

        python cs285/scripts/run_hw5_explore.py \
                -cfg experiments/exploration/pointmass_medium_rnd_step10000.yaml \
                --dataset_dir datasets/

        python cs285/scripts/run_hw5_explore.py \
                -cfg experiments/exploration/pointmass_medium_rnd_step20000.yaml \
                --dataset_dir datasets/
        ;;

    7)
        echo "========================================================="
        echo "4.3 Data Ablations - CQL on Medium maze with different dataset sizes"
        echo "------------------------------------------"
        echo "Testing dataset sizes: 1000, 5000, 10000, 20000"
        echo ""

        for size in 1000 5000 10000 20000; do
            echo "Running CQL with ${size} samples on Medium maze..."
            python ./cs285/scripts/run_hw5_offline.py \
                -cfg experiments/offline/pointmass_medium_cql${size}.yaml \
                --dataset_dir datasets
        done
        ;;

    8)
        echo "========================================================="
        echo "5.0 Online Fine-tuning on Hard environment"
        echo "------------------------------------------"
        echo "Running CQL, AWAC, DQN, and IQL with offline→online finetuning"
        echo ""

        python ./cs285/scripts/run_hw5_finetune.py \
                -cfg experiments/finetuning/pointmass_hard_awac_finetune.yaml \
                --dataset_dir datasets

        python ./cs285/scripts/run_hw5_finetune.py \
                -cfg experiments/finetuning/pointmass_hard_cql_finetune.yaml \
                --dataset_dir datasets

        python ./cs285/scripts/run_hw5_finetune.py \
                -cfg experiments/finetuning/pointmass_hard_dqn_finetune.yaml \
                --dataset_dir datasets

        python ./cs285/scripts/run_hw5_finetune.py \
                -cfg experiments/finetuning/pointmass_hard_iql_finetune.yaml \
                --dataset_dir datasets
        ;;

    dead)
        echo "=========================================="
        echo ""
        if [ -n "$EXTRA_ARGS" ]; then
            echo "Additional arguments: $EXTRA_ARGS"
        fi
        python cs285/scripts/run_hw3_sac.py \
            -cfg experiments/sac/humanoid.yaml \
            $EXTRA_ARGS
        ;;

    *)
        echo "Error: Invalid experiment number '$EXPERIMENT'"
        echo "Available experiments: 1-8"
        echo "Run 'bash scripts.sh' without arguments to see the full list"
        exit 1
        ;;
esac

echo ""
echo "Experiment $EXPERIMENT completed!"
