#!/bin/bash

# Kill existing screen sessions that start with "klsparsity_"
screen -ls | grep '\.klsparsity_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit

klsparsity_pis=(0.01 0.05 0.1)
klsparsity_lambdas=(0.000001 0.00001 0.0001 0.001 0.01)

for pi in "${klsparsity_pis[@]}"; do
    for lam in "${klsparsity_lambdas[@]}"; do
        run_name="klsparsity_pi${pi}_lambda${lam}"
        screen -dmS "$run_name" bash -c "export WANDB_RUN_NAME=${run_name}; \
        python learn_addition.py --lr 5e-5 --micro_batch_size 8 \
        --max_iters 100000 --training_at_100_patience 100 \
        --klsparsity --klsparsity_pi ${pi} \
        --klsparsity_lambda ${lam}"
        sleep 30 # this allows the GPU mem to fill up
    done
done
