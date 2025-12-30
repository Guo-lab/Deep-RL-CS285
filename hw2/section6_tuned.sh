for seed in $(seq 1 5); do
    python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
    --exp_name pendulum_default_s$seed \
    -rtg --use_baseline -na \
    --batch_size 5000 \
    --seed $seed
done

# # section6_na_tuned_v2

# for seed in $(seq 1 3); do
#     echo "  Tuned - Seed $seed/3"
#     python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
#         -n 70 \
#         --exp_name pendulum_tuned_s$seed \
#         -rtg --use_baseline -na \
#         --batch_size 2000 \
#         -lr 0.015 \
#         --gae_lambda 0.97 \
#         -l 2 -s 64 \
#         -blr 0.02 -bgs 10 \
#         --seed $seed \
#         --which_gpu 0
# done


# # section6_na_tuned_v3

# for seed in $(seq 1 3); do
#     echo "  Tuned - Seed $seed/3"
#     python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
#         -n 70 \
#         --exp_name pendulum_tuned_s$seed \
#         -rtg --use_baseline -na \
#         --batch_size 2500 \
#         -lr 0.012 \
#         --gae_lambda 0.97 \
#         -l 2 -s 64 \
#         -blr 0.02 -bgs 10 \
#         --seed $seed \
#         --which_gpu 0
# done


# # section6_na_tuned_v4

# for seed in $(seq 1 3); do
#     echo "  Tuned - Seed $seed/3"
#     python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
#         -n 70 \
#         --exp_name pendulum_tuned_s$seed \
#         -rtg --use_baseline -na \
#         --batch_size 2500 \
#         -lr 0.011 \
#         --gae_lambda 0.97 \
#         -l 2 -s 64 \
#         -blr 0.03 -bgs 13 \
#         --seed $seed \
#         --which_gpu 0
# done



# # section6_na_tuned_v5

# for seed in $(seq 1 3); do
#     echo "  Tuned - Seed $seed/3"
#     python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
#         -n 70 \
#         --exp_name pendulum_tuned_s$seed \
#         -rtg --use_baseline -na \
#         --batch_size 2000 \
#         -lr 0.012 \
#         --gae_lambda 0.97 \
#         -l 2 -s 64 \
#         -blr 0.05 -bgs 18 \
#         --seed $seed \
#         --which_gpu 0
# done


# # section6_na_tuned_v6

# for seed in $(seq 1 3); do
#     echo "  Tuned - Seed $seed/3"
#     python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
#         -n 70 \
#         --exp_name pendulum_tuned_s$seed \
#         -rtg --use_baseline -na \
#         --batch_size 1500 \
#         -lr 0.018 \
#         --gae_lambda 0.96 \
#         -l 2 -s 64 \
#         -blr 0.022 -bgs 12 \
#         --seed $seed \
#         --which_gpu 0
# done


# section6_na_tuned_v7

for seed in $(seq 1 3); do
    echo "  Tuned - Seed $seed/3"
    python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
        -n 70 \
        --exp_name pendulum_tuned_s$seed \
        -rtg --use_baseline -na \
        --batch_size 1200 \
        -lr 0.018 \
        --gae_lambda 0.96 \
        -l 2 -s 64 \
        -blr 0.035 -bgs 18 \
        --seed $seed \
        --which_gpu 0
done