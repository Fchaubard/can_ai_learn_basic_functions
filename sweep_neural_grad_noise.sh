#!/bin/bash

# Kill existing screen sessions that start with "neural_noise_"
screen -ls | grep '\.neural_noise_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit

log_normal_mus=(0.0)
log_normal_sigmas=(0.0001 0.001 0.01 0.05 0.1)

for mu in "${log_normal_mus[@]}"; do
    for sigma in "${log_normal_sigmas[@]}"; do
        run_name="neural_noise_mu${mu}_sigma${sigma}"
        screen -dmS "$run_name" bash -c "export WANDB_RUN_NAME=${run_name}; \
        python learn_addition.py --lr 5e-5 --micro_batch_size 8 \
        --max_iters 100000 --training_at_100_patience 100 \
        --log_normal_gradient_noise \
        --log_normal_mu ${mu} --log_normal_sigma ${sigma}"
        sleep 30 # this allows the GPU mem to fill up
    done
done
