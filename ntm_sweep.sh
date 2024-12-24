#!/bin/bash

##############################################################################
# run_sweep_screens.sh
#
# Example shell script to:
# 1) Kill existing screen sessions that start with "NTM_"
# 2) Launch new screen sessions for each hyperparam combination
#
# Usage:
#   chmod +x ntm_sweep.sh
#   ./ntm_sweep.sh
##############################################################################

# 1) Kill existing screen sessions that match "NTM_"
echo "[INFO] Killing existing screens named 'NTM_*'..."
screen -ls | grep '\.NTM_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

# 2) Define hyperparameter values
TASKS=("copy" "repeat_copy" "associative_recall")
HIDDEN_SIZES=(64 128)
MEMORY_SIZES=(64 128)
HEAD_SIZES=(32 64)
LEARNING_RATES=(0.001 0.0001)
OPTIMIZERS=("adam" "mezo")

MAX_ITERS=10000
BATCH_SIZE=16
SEQ_LEN=10
LOG_INTERVAL=500

WANDB_PROJ="NTM-ICML-Screens"
RUN_PREFIX="NTM_"  # screen session name prefix

# 3) Loop over all hyperparameter combinations
for TASK in "${TASKS[@]}"; do
  for HS in "${HIDDEN_SIZES[@]}"; do
    for MEM in "${MEMORY_SIZES[@]}"; do
      for HEAD_SZ in "${HEAD_SIZES[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
          for OPT in "${OPTIMIZERS[@]}"; do
            # Construct a unique run name and screen session name
            RUN_NAME="${RUN_PREFIX}${TASK}_hs${HS}_mem${MEM}_head${HEAD_SZ}_lr${LR}_${OPT}"
            
            echo "[INFO] Launching screen session: $RUN_NAME"
            
            # 4) Launch the sweep run in a new screen session
            screen -dmS "$RUN_NAME" bash -c "
              echo \"[INFO] Starting run: $RUN_NAME\";
              export WANDB_RUN_NAME=$RUN_NAME;
              python train_ntm.py \
                --task $TASK \
                --seq_len $SEQ_LEN \
                --hidden_size $HS \
                --memory_size $MEM \
                --head_size $HEAD_SZ \
                --num_heads 1 \
                --batch_size $BATCH_SIZE \
                --max_iters $MAX_ITERS \
                --log_interval $LOG_INTERVAL \
                --learning_rate $LR \
                --optimizer $OPT \
                --wandb_proj $WANDB_PROJ;

              echo \"[INFO] Finished run: $RUN_NAME\";
              exec bash
            "

            # Optional sleep to let GPU memory settle
            sleep 5
          done
        done
      done
    done
  done
done

echo "[INFO] Done launching all screen runs."
