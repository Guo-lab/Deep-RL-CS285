# CUDA_LAUNCH_BLOCKING=1 python cs285/scripts/run_hw2.py \
#     --env_name Humanoid-v4 --ep_len 1000 \
#     --discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001 \
#     --baseline_gradient_steps 50 \
#     -na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
#     --exp_name humanoid --video_log_freq 100



CUDA_LAUNCH_BLOCKING=1 python cs285/scripts/run_hw2.py \
    --env_name Humanoid-v4 --ep_len 1000 \
    --discount 0.99 -n 1000 -l 4 -s 512 -b 50000 -lr 0.0002 \
    --baseline_gradient_steps 120 \
    -na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
    --exp_name humanoid_tuned --video_log_freq 200