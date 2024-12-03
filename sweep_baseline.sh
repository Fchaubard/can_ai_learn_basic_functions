#!/bin/bash

# Kill existing screen sessions that start with "baseline_"
screen -ls | grep '\.baseline_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit

learning_rates=(1e-5 5e-5 1e-4)
micro_batch_sizes=(4 8 16)

for lr in "${learning_rates[@]}"; do
    for bs in "${micro_batch_sizes[@]}"; do
        run_name="baseline_lr${lr}_bs${bs}"
        screen -dmS "$run_name" bash -c "export WANDB_RUN_NAME=${run_name}; \
        python learn_addition.py --lr ${lr} --micro_batch_size ${bs} \
        --max_iters 100000 --training_at_100_patience 100"
        sleep 30 # this allows the GPU mem to fill up
    done
done
