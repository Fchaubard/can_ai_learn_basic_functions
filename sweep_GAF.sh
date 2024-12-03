#!/bin/bash

# Kill existing screen sessions that start with "gaf_"
screen -ls | grep '\.gaf_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit

for tau in $(seq 0.9 0.02 1.1); do
    tau=$(printf "%.2f" $tau)
    run_name="gaf_tau${tau}"
    screen -dmS "$run_name" bash -c "export WANDB_RUN_NAME=${run_name}; \
    python learn_addition.py --lr 5e-5 --micro_batch_size 8 \
    --max_iters 100000 --training_at_100_patience 100 \
    --gaf --gaf_tau ${tau}"
    sleep 30 # this allows the GPU mem to fill up
done
