#!/bin/bash

# Kill existing screen sessions that start with "mezo_"
screen -ls | grep '\.mezo_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit

learning_rates=(1e-5 5e-5 1e-4)
mezo_epsilons=(0.0001 0.001 0.01)

for lr in "${learning_rates[@]}"; do
    for epsilon in "${mezo_epsilons[@]}"; do
        run_name="mezo_lr${lr}_epsilon${epsilon}"
        screen -dmS "$run_name" bash -c "export WANDB_RUN_NAME=${run_name}; \
        python learn_addition.py --lr ${lr} --micro_batch_size 8 \
        --max_iters 100000 --training_at_100_patience 100 \
        --mezo --mezo_epsilon ${epsilon}"
        sleep 30 # this allows the GPU mem to fill up
    done
done
