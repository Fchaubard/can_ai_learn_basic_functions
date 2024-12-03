#!/bin/bash

# Kill existing screen sessions that start with "wd_sched_"
screen -ls | grep '\.wd_sched_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit

weight_decay_ks=(1.0 1.1 1.2 1.3)

for k in "${weight_decay_ks[@]}"; do
    run_name="wd_sched_k${k}"
    screen -dmS "$run_name" bash -c "export WANDB_RUN_NAME=${run_name}; \
    python learn_addition.py --lr 5e-5 --micro_batch_size 8 \
    --max_iters 100000 --training_at_100_patience 100 \
    --weight_decay_schedule --weight_decay_k ${k}"
    sleep 30 # this allows the GPU mem to fill up
done
